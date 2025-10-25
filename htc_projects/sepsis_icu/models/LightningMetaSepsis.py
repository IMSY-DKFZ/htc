# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.common.HTCDataset import HTCDataset
from htc_projects.sepsis_icu.models.DatasetMeta import DatasetMeta
from htc_projects.sepsis_icu.models.LightningMedianPixelSepsis import LightningMedianPixelSepsis
from htc_projects.sepsis_icu.models.ModelTableMeta import ModelTableMeta


class LightningMetaSepsis(LightningMedianPixelSepsis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ModelTableMeta(self.config)

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        return DatasetMeta(**kwargs)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["meta"])
