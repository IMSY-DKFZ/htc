# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from htc.models.common.class_weights import calculate_class_weights
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.loss import KLDivLossWeighted
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.common.utils import get_n_classes
from htc.models.superpixel_classification.DatasetSuperpixelImage import DatasetSuperpixelImage
from htc.models.superpixel_classification.DatasetSuperpixelStream import DatasetSuperpixelStream


class LightningSuperpixelClassification(HTCLightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        name = self.config.get("model/model_name", "ModelSuperpixelClassification")
        if importlib.util.find_spec(f"htc.models.superpixel_classification.{name}") is not None:
            module = importlib.import_module(f"htc.models.superpixel_classification.{name}")
            ModelClass = getattr(module, name)

            self.model = ModelClass(self.config)

        if self.config["model/class_weight_method"] and self.dataset_train is not None:
            weights = calculate_class_weights(
                self.config, *self.dataset_train.label_counts()
            )  # Calculate class weights to overcome class imbalances
        else:
            weights = torch.ones(get_n_classes(self.config))

        self.kl_loss_weighted = KLDivLossWeighted(weight=weights)

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        return DatasetSuperpixelStream(**kwargs) if kwargs["train"] else DatasetSuperpixelImage(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return StreamDataLoader(self.dataset_train, self.config)

    def val_dataloader(self, **kwargs) -> list[DataLoader]:
        return super().val_dataloader(batch_size=1, prefetch_factor=2, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _predict_images(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert batch["valid_pixels"].shape[0] == 1, "Can only handle one image at a time"
        img_height = batch["valid_pixels"].shape[1]
        img_width = batch["valid_pixels"].shape[2]

        predictions = self(batch["features"].squeeze(dim=0))
        predictions = predictions.repeat_interleave(
            batch["spxs_sizes"].squeeze(dim=0), dim=0
        )  # Repeat each classification value according to the superpixel size; [307200, 19]
        predictions = predictions.permute(1, 0)  # [19, 307200]

        image_predictions = torch.empty(
            (1, predictions.shape[0], img_height, img_width), dtype=predictions.dtype, device=self.device
        )
        image_predictions[0, :, batch["spxs_indices_rows"], batch["spxs_indices_cols"]] = predictions.unsqueeze(dim=1)

        return {"class": image_predictions}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        predictions = self(batch["features"])

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.kl_loss_weighted(predictions, batch["weak_labels"])
        self.log("train/kl_loss", loss, on_epoch=True)

        return {"loss": loss, "img_indices": batch["image_index"]}
