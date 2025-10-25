# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

import htc.models.common.functions
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import get_n_classes
from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification
from htc.utils.Config import Config
from htc_projects.sepsis_icu.models.ModelTableMeta import ModelTableMeta


class ModelImageMeta(HTCModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

        per_model_output_size = config["model/per_model_output_size"]
        self.image = ModelSuperpixelClassification(self.config, n_classes=per_model_output_size, **kwargs)
        self.image_norm = nn.BatchNorm1d(num_features=per_model_output_size)
        self.meta = ModelTableMeta(self.config, n_classes=per_model_output_size, **kwargs)
        self.meta_norm = nn.BatchNorm1d(num_features=per_model_output_size)

        self.F = htc.models.common.functions.activation_functions[self.config["model/activation_function"]]
        NormalizationLayer = (
            getattr(nn, f"{self.config['model/normalization']}1d")
            if self.config["model/normalization"]
            else nn.Identity
        )
        DropoutLayer = nn.Dropout if self.config["model/dropout"] else nn.Identity

        self.fc = nn.Linear(in_features=2 * per_model_output_size, out_features=per_model_output_size)
        self.fc_norm = NormalizationLayer(num_features=self.fc.out_features)
        self.fc_dropout = DropoutLayer(self.config["model/dropout"])

        self.head = nn.Linear(in_features=self.fc.out_features, out_features=get_n_classes(config))

    def forward(self, x_image: torch.Tensor, x_meta: torch.Tensor) -> torch.Tensor:
        x_image = self.image(x_image)
        x_image = self.image_norm(x_image)
        x_meta = self.meta(x_meta)
        x_meta = self.meta_norm(x_meta)

        x = torch.cat([x_image, x_meta], dim=1)
        x = self.fc_dropout(self.F(self.fc_norm(self.fc(x)), inplace=True))
        x = self.head(x)

        return x
