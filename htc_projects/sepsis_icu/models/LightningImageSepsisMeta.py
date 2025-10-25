# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc_projects.sepsis_icu.models.LightningImageSepsis import LightningImageSepsis
from htc_projects.sepsis_icu.models.ModelImageMeta import ModelImageMeta


class LightningImageSepsisMeta(LightningImageSepsis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ModelImageMeta(self.config, fold_name=self.fold_name)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["features"].permute(0, 3, 1, 2)

        return self.model(x, batch["meta"])
