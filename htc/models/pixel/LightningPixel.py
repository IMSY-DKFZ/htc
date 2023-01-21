# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from htc.models.common.class_weights import calculate_class_weights
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.common.utils import get_n_classes
from htc.models.image.DatasetImage import DatasetImage
from htc.models.pixel.DatasetPixelStream import DatasetPixelStream
from htc.tissue_atlas.median_pixel.DatasetMedianPixel import DatasetMedianPixel


class LightningPixel(HTCLightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        name = self.config.get("model/model_name", "ModelPixel")
        if importlib.util.find_spec(f"htc.models.pixel.{name}") is not None:
            module = importlib.import_module(f"htc.models.pixel.{name}")
            ModelClass = getattr(module, name)

            self.model = ModelClass(self.config)

        if self.config["model/class_weight_method"] and self.dataset_train is not None:
            weights = calculate_class_weights(self.config, *self.dataset_train.label_counts())
        else:
            weights = torch.ones(get_n_classes(self.config))

        self.ce_loss_weighted = nn.CrossEntropyLoss(weight=weights)

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        train = kwargs["train"]
        config = kwargs["config"]
        if config["input/median_spectra"]:
            return DatasetMedianPixel(**kwargs) if train else DatasetImage(**kwargs)
        else:
            return DatasetPixelStream(**kwargs) if train else DatasetImage(**kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.config["input/median_spectra"] or self.config["input/simulated"]:
            return super().train_dataloader()
        else:
            return StreamDataLoader(self.dataset_train, self.config)

    def val_dataloader(self, **kwargs) -> list[DataLoader]:
        # Only one image fits into GPU memory
        return super().val_dataloader(batch_size=1, prefetch_factor=2, **kwargs)

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader(batch_size=1, prefetch_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["class"]

    def _predict_images(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["features"]

        # Get an output from each pixel by reshaping the image to a list of pixels and then reshape back again to an image
        batches = features.shape[0]
        spatial = features.shape[1:3]
        channels = features.shape[3]

        # x.shape = [1, 480, 640, 100]
        features = features.reshape(-1, channels)  # List of pixels: x.shape = [307200, 100]
        logits = self(features)  # Model prediction per pixel: x.shape = [307200, 15]
        logits = logits.reshape(batches, *spatial, -1)  # Go back to an image: x.shape = [1, 480, 640, 15]
        logits = logits.permute(0, 3, 1, 2)  # Channel first format (NCHW): x.shape = [1, 15, 480, 640]

        return {"class": logits}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        if self.config["input/specs_threshold"]:
            # Exclude specular highlight pixels
            valid_pixels = batch["features"][~batch["specs"]]
            valid_labels = batch["labels"][~batch["specs"]]
        else:
            valid_pixels = batch["features"]
            valid_labels = batch["labels"]

        prediction = self(valid_pixels)
        loss = self.ce_loss_weighted(prediction, valid_labels)

        self.log("train/ce_loss", loss, on_epoch=True)

        return {"loss": loss, "img_indices": torch.unique(batch["image_index"])}

    def encode_images(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = batch["features"]
        channels = x.shape[3]

        x = x.reshape(-1, channels)

        return self.model.encode(x)
