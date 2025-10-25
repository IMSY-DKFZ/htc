# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.image.LightningImage import LightningImage
from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification
from htc_projects.sepsis_icu.models.SepsisEvaluationMixin import SepsisEvaluationMixin


class LightningImageSepsis(SepsisEvaluationMixin, LightningImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ModelSuperpixelClassification(self.config, fold_name=self.fold_name)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["features"].permute(0, 3, 1, 2)

        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        predictions = self(batch)
        image_labels = batch["image_labels"]

        ce_loss = self.ce_loss_weighted(predictions, image_labels)
        self.log("train/ce_loss", ce_loss, on_epoch=True)

        return {"loss": ce_loss}
