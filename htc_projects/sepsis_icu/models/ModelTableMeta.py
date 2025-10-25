# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

import htc.models.common.functions
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import get_n_classes
from htc.utils.Config import Config


class ModelTableMeta(HTCModel):
    def __init__(self, config: Config, n_classes: int = None, **kwargs):
        super().__init__(config, **kwargs)
        if n_classes is None:
            n_classes = get_n_classes(self.config)
        n_input_features = len(self.config["input/meta/attributes"])
        normalization = self.config["model/meta/normalization"]
        dropout = self.config["model/meta/dropout"]

        self.F = htc.models.common.functions.activation_functions[self.config["model/meta/activation_function"]]
        NormalizationLayer = getattr(nn, f"{normalization}1d") if normalization else nn.Identity
        DropoutLayer = nn.Dropout if dropout else nn.Identity

        self.input_normalization = nn.BatchNorm1d(n_input_features)

        self.fc1 = nn.Linear(in_features=n_input_features, out_features=self.config["model/meta/layer1_size"])
        self.fc1_norm = NormalizationLayer(num_features=self.fc1.out_features)
        self.fc1_dropout = DropoutLayer(dropout)

        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=self.config["model/meta/layer2_size"])
        self.fc2_norm = NormalizationLayer(num_features=self.fc2.out_features)
        self.fc2_dropout = DropoutLayer(dropout)

        self.head = nn.Linear(in_features=self.fc2.out_features, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_normalization(x)
        x = self.fc1_dropout(self.F(self.fc1_norm(self.fc1(x)), inplace=True))
        x = self.fc2_dropout(self.F(self.fc2_norm(self.fc2(x)), inplace=True))
        x = self.head(x)

        return x
