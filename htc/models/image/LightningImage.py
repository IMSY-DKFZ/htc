# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from scipy import interpolate
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from htc.models.common.class_weights import calculate_class_weights
from htc.models.common.HierarchicalSampler import HierarchicalSampler
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.loss import SuperpixelLoss
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.common.torch_helpers import smooth_one_hot
from htc.models.common.utils import get_n_classes
from htc.models.image.DatasetImageBatch import DatasetImageBatch
from htc.settings import settings
from htc.utils.type_from_string import type_from_string


class LightningImage(HTCLightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        name = self.config["model/model_name"]
        if name is not None:
            if ">" in name:
                ModelClass = type_from_string(name)
            else:
                module = importlib.import_module(f"htc.models.image.{name}")
                ModelClass = getattr(module, name)

            self.model = ModelClass(self.config)

        if self.config["model/class_weight_method"] and self.dataset_train is not None:
            class_weights = calculate_class_weights(self.config, *self.dataset_train.label_counts())
        else:
            class_weights = torch.ones(get_n_classes(self.config))
        self.register_buffer("class_weights", class_weights, persistent=False)

        self.ce_loss_weighted = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=self.config.get("optimization/label_smoothing_ce", 0)
        )

        self.dice_loss = DiceLoss(reduction="none", softmax=True, batch=True)
        if hasattr(self.dice_loss, "class_weight"):
            # MONAI >=1.3.0 uses a class weight buffer which breaks loading of old checkpoints
            # Since we have our own class weighting anyway, we simple remove the buffer
            self.dice_loss.class_weight = None

        if "optimization/spx_loss_weight" in self.config:
            assert "input/superpixels" in self.config, (
                "Superpixels are missing in the input specification but they are required for the superpixel loss"
            )
            self.spx_loss = SuperpixelLoss()

            if type(self.config["optimization/spx_loss_weight"]) == dict:
                # spx_loss_weight is specified as a function of the current epoch
                epochs = self.config["optimization/spx_loss_weight/epochs"]
                weights = self.config["optimization/spx_loss_weight/weights"]
                self.spx_loss_weight_f = interpolate.interp1d(
                    epochs, weights, fill_value=(weights[0], weights[-1]), bounds_error=False
                )
            else:
                # Constant value returned as array to be consistent with the interpolation output
                self.spx_loss_weight_f = lambda _: np.array(self.config["optimization/spx_loss_weight"])

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        if kwargs["train"]:
            if kwargs["config"]["input/hierarchical_sampling"]:
                sampler = HierarchicalSampler(kwargs["paths"], kwargs["config"])
            else:
                sampler = RandomSampler(
                    kwargs["paths"], replacement=True, num_samples=kwargs["config"]["input/epoch_size"]
                )

            return DatasetImageBatch(sampler=sampler, **kwargs)
        else:
            # We want every image from the validation/test dataset
            sampler = list(range(len(kwargs["paths"])))
            return DatasetImageBatch(sampler=sampler, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return StreamDataLoader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return [StreamDataLoader(dataset_val) for dataset_val in self.datasets_val]

    def test_dataloader(self) -> DataLoader:
        return StreamDataLoader(self.dataset_test)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["features"]
        x = x.permute(0, 3, 1, 2)  # Input dimension for UNet needs to be [N, C, H, W]

        # x.stride() = (30720000, 1, 64000, 100), i.e. channel last format
        return self.model(x)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, return_valid_tensors: bool = False
    ) -> dict[str, torch.Tensor]:
        ce_loss_weight = self.config.get("optimization/ce_loss_weight", 1.0)
        dice_loss_weight = self.config.get("optimization/dice_loss_weight", 1.0)

        if "optimization/spx_loss_weight" in self.config:
            # .item is necessary here as otherwise lightning does not handle the logging properly
            spx_loss_weight = self.spx_loss_weight_f(self.current_epoch).item()
            self.log("train/spx_loss_weight", spx_loss_weight)
        else:
            spx_loss_weight = 0

        predictions = self(batch)
        if type(predictions) == dict:
            predictions = predictions["class"]  # [BCHW]
        n_classes = predictions.size(1)

        labels = batch["labels"]
        valid_pixels = batch["valid_pixels"]

        if labels.ndim < batch["features"].ndim:
            used_labels = labels[valid_pixels].unique()

            # We need to replace the invalid labels with a "valid" one for the one-hot encoding (but the values won't be used)
            labels = labels.masked_fill(~valid_pixels, 0)
            if self.config["optimization/label_smoothing"]:
                labels = smooth_one_hot(
                    labels, n_classes=n_classes, smoothing=self.config["optimization/label_smoothing"]
                )  # [BHWC]
            else:
                labels = F.one_hot(labels, num_classes=n_classes).to(torch.float16)  # [BHWC]
        else:
            # E.g. in case an augmentation already applied the one-hot encoding
            used_labels = batch["labels_original"][valid_pixels].unique()

        # Calculate the losses only for the valid pixels
        # Keep the class dimension
        valid_predictions = predictions.permute(0, 2, 3, 1)[valid_pixels]  # (samples, class)
        valid_labels = labels[valid_pixels]  # (samples, class)
        assert valid_predictions.shape == valid_labels.shape, "Invalid shape"

        n_invalid = (~valid_predictions.isfinite()).sum()
        if n_invalid > 0:
            valid_predictions.nan_to_num_()
            settings.log.warning(
                f"Found {n_invalid} invalid values in prediction of the annotated area"
                f" ({self.current_epoch = }, {self.global_step = })"
            )
            settings.log_once.warning(
                "nan_to_num will be applied to the predictions but please note that this is only a workaround and no"
                " real solution. It is very likely that the model does not learn properly (this message is not shown"
                " again)"
            )

        # Cross Entropy loss
        ce_loss = self.ce_loss_weighted(valid_predictions, valid_labels)

        self.log("train/ce_loss", ce_loss, on_epoch=True)
        loss_sum = ce_loss_weight * ce_loss

        # Superpixel loss
        if spx_loss_weight > 0:
            spxs_vec = []
            for b in range(batch["spxs"].shape[0]):
                # The superpixel id must be unique per batch for easier calculation of the gini coefficient
                # Here, we avoid this problem by shifting the indices to the next free range in the index space, e.g. 1, 2 --> 1001, 1002
                spxs_vec.append(batch["spxs"][b].flatten() + b * self.config["input/superpixels/n_segments"])
            spxs_vec = torch.cat(spxs_vec)  # [M]
            prediction_vec = predictions.transpose(1, 0).flatten(start_dim=1).transpose(1, 0)  # [N, 19]

            spx_loss = self.spx_loss(prediction_vec, spxs_vec)
            loss_sum += spx_loss_weight * spx_loss
            self.log("train/spx_loss", spx_loss, on_epoch=True)

        # Dice loss
        dice_loss = self.dice_loss(input=valid_predictions, target=valid_labels)  # Dice per class [C]

        # We use only the classes available in the batch to calculate the dice
        used_weights = self.class_weights[used_labels]
        # Only use dice values from classes which really occurred in the images, e.g. [4] and weight the class dices
        dice_loss = dice_loss[used_labels] * used_weights
        dice_loss = dice_loss.sum() / used_weights.sum()  # Weighted average
        self.log("train/dice_loss", dice_loss, on_epoch=True)
        loss_sum += dice_loss_weight * dice_loss

        res = {}
        loss_weights = sum([ce_loss_weight, dice_loss_weight, spx_loss_weight])
        if return_valid_tensors:
            res["valid_predictions"] = valid_predictions
            res["valid_labels"] = valid_labels
            res["loss_sum"] = loss_sum
            res["loss_weights"] = loss_weights

        # Normalize the loss (weighted average)
        res["loss"] = loss_sum / loss_weights
        return res
