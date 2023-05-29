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
from htc.models.common.loss import KLDivLossWeighted, SuperpixelLoss
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.common.torch_helpers import smooth_one_hot
from htc.models.common.utils import get_n_classes
from htc.models.image.DatasetImageStream import DatasetImageStream
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
            weights = calculate_class_weights(
                self.config, *self.dataset_train.label_counts()
            )  # Calculate class weights to overcome class imbalances
        else:
            weights = torch.ones(get_n_classes(self.config))

        if self.config["optimization/label_smoothing"]:
            self.ce_loss_weighted = KLDivLossWeighted(weight=weights)
        else:
            self.ce_loss_weighted = nn.CrossEntropyLoss(weight=weights)

        self.dice_loss = DiceLoss(reduction="none", to_onehot_y=True, softmax=True, batch=True)
        if "optimization/spx_loss_weight" in self.config:
            assert (
                "input/superpixels" in self.config
            ), "Superpixels are missing in the input specification but they are required for the superpixel loss"
            self.spx_loss = SuperpixelLoss()

            if type(self.config["optimization/spx_loss_weight"]) == dict:
                # spx_loss_weight is specified as a function of the current epoch
                epochs = self.config["optimization/spx_loss_weight/epochs"]
                weights = self.config["optimization/spx_loss_weight/weights"]
                self.spx_loss_weight_f = interpolate.interp1d(
                    epochs, weights, fill_value=(weights[0], weights[-1]), bounds_error=False
                )
            else:
                self.spx_loss_weight_f = lambda _: np.array(
                    self.config["optimization/spx_loss_weight"]
                )  # Also an array to be consistent with the interpolation output

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        if kwargs["train"]:
            if kwargs["config"]["input/hierarchical_sampling"]:
                sampler = HierarchicalSampler(kwargs["paths"], kwargs["config"])
            else:
                sampler = RandomSampler(
                    kwargs["paths"], replacement=True, num_samples=kwargs["config"]["input/epoch_size"]
                )

            return DatasetImageStream(sampler=sampler, **kwargs)
        else:
            # We want every image from the validation/test dataset
            sampler = list(range(len(kwargs["paths"])))
            return DatasetImageStream(sampler=sampler, **kwargs)

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int, return_valid_tensors: bool = False) -> dict:
        ce_loss_weight = self.config.get("optimization/ce_loss_weight", 1.0)
        dice_loss_weight = self.config.get("optimization/dice_loss_weight", 1.0)

        if "optimization/spx_loss_weight" in self.config:
            spx_loss_weight = self.spx_loss_weight_f(
                self.current_epoch
            ).item()  # .item is necessary here as otherwise lightning does not handle the logging properly
            self.log("train/spx_loss_weight", spx_loss_weight)
        else:
            spx_loss_weight = 0

        labels = batch["labels"]
        predictions = self(batch)
        if type(predictions) == dict:
            predictions = predictions["class"]

        # Calculate the losses only for the valid pixels. This is a bit complicated since we need to preserve the logits dimension (we discard the spatial dimension in this process)
        valid_pixels_mask = (
            batch["valid_pixels"].unsqueeze(dim=1).expand(-1, predictions.shape[1], -1, -1)
        )  # Bring the mask to the shape [3, 19, 480, 640] (same as prediction)
        valid_predictions = predictions.transpose(1, 0)[
            valid_pixels_mask.transpose(1, 0)
        ]  # Apply the mask but put the logits dimension in front [19*N] (N = number of valid pixels which remain). This ensures that we can easily reshape the logits dimension back
        valid_predictions = valid_predictions.reshape(predictions.shape[1], -1).transpose(
            1, 0
        )  # Reshape the logits dimension back and put the batch dimension back to the front [N, 19]
        valid_labels = labels[batch["valid_pixels"]]
        assert valid_predictions.shape[0] == valid_labels.shape[0], "Invalid shape in the batch dimension"
        assert valid_predictions.shape[1] == predictions.shape[1], "Invalid shape in the logits dimension"

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
        if self.config["optimization/label_smoothing"]:
            valid_prediction_log = F.log_softmax(valid_predictions, dim=1)
            valid_labels_smooth = smooth_one_hot(
                valid_labels, n_classes=valid_predictions.size(1), smoothing=self.config["optimization/label_smoothing"]
            )
            ce_loss = self.ce_loss_weighted(valid_prediction_log, valid_labels_smooth)
        else:
            ce_loss = self.ce_loss_weighted(valid_predictions, valid_labels)

        self.log(
            "train/ce_loss", ce_loss, on_epoch=True
        )  # Automatically aggregated and averaged by Lightning for all epochs
        loss = ce_loss_weight * ce_loss

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
            loss += spx_loss_weight * spx_loss
            self.log("train/spx_loss", spx_loss, on_epoch=True)

        # Dice loss
        valid_predictions = valid_predictions.unsqueeze(dim=-1)  # All pixels are put into the batch dimension [N, C, 1]
        valid_labels = valid_labels.unsqueeze(dim=-1).unsqueeze(dim=-1)  # Same for the labels [N, 1, 1]
        dice_loss = self.dice_loss(input=valid_predictions, target=valid_labels)  # Dice per class [C]

        # We use only the classes available in the batch to calculate the dice
        used_labels = valid_labels.unique()
        used_weights = self.ce_loss_weighted.weight[used_labels]
        dice_loss = (
            dice_loss[used_labels] * used_weights
        )  # Only use dice values from classes which really occurred in the images, e.g. [4] and weight the class dices
        dice_loss = dice_loss.sum() / used_weights.sum()  # Weighted average
        self.log("train/dice_loss", dice_loss, on_epoch=True)
        loss += dice_loss_weight * dice_loss

        # Normalize the loss (weighted average)
        loss /= sum([ce_loss_weight, dice_loss_weight, spx_loss_weight])

        res = {"loss": loss}

        if return_valid_tensors:
            res["valid_predictions"] = valid_predictions
            res["valid_labels"] = valid_labels

        return res
