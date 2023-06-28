# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
from monai.losses import FocalLoss

from htc import LightningImage


class LightningImageThoracic(LightningImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Focal loss for showcasing purposes
        self.focal_loss = FocalLoss()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        # Batch with training data:
        # - features: L1 normalized spectral data (5, 480, 640, 100)
        # - labels: segmentation mask with the combined mask from all annotators (5, 480, 640)
        # - valid_pixels: annotated pixels from all annotators (5, 480, 640)

        # LightningImage already computes the dice and cross entropy loss
        # return_valid_tensors=True gives us access to the valid pixels in the prediction/labels. This is handy if we want to compute the loss only on the pixels where we have at least one annotation
        res = super().training_step(batch, batch_idx, return_valid_tensors=True)

        # img_model["valid_predictions"].shape = (N, 2)
        # img_model["valid_labels"].shape = (N, 2), already one-hot encoded
        focal_loss = self.focal_loss(res["valid_predictions"], res["valid_labels"])
        self.log("train/focal_loss", focal_loss, on_epoch=True)

        # Combine the focal loss with the existing cross entropy and dice loss
        focal_loss_weight = self.config.get("optimization/focal_loss_weight", 1.0)
        loss_sum = res["loss_sum"] + focal_loss * focal_loss_weight

        # Weighted average of all losses
        return loss_sum / (res["loss_weights"] + focal_loss_weight)
