# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.median_pixel.LightningMedianPixel import LightningMedianPixel
from htc_projects.sepsis_icu.models.SepsisEvaluationMixin import SepsisEvaluationMixin


class LightningMedianPixelSepsis(SepsisEvaluationMixin, LightningMedianPixel):
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        predictions = self(batch)
        image_labels = batch["image_labels"]

        ce_loss = self.ce_loss_weighted(predictions, image_labels)
        self.log("train/ce_loss", ce_loss, on_epoch=True)

        return {"loss": ce_loss}
