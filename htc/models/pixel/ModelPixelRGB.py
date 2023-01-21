# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

import htc.models.common.functions
from htc.models.common.Heads import Heads
from htc.models.common.HTCModel import HTCModel
from htc.utils.Config import Config


class ModelPixelRGB(HTCModel):
    def __init__(self, config: Config):
        super().__init__(config)

        self.F = htc.models.common.functions.activation_functions[self.config["model/activation_function"]]

        if self.config["model/normalization"]:
            NormalizationLayer = getattr(nn, f'{self.config["model/normalization"]}1d')
        else:
            NormalizationLayer = nn.Identity

        if self.config["model/dropout"]:
            DropoutLayer = nn.Dropout
        else:
            DropoutLayer = nn.Identity

        # FNN
        self.fc1 = nn.Linear(in_features=self.config["input/n_channels"], out_features=200)
        self.fc1_norm = NormalizationLayer(num_features=self.fc1.out_features)
        self.fc1_dropout = DropoutLayer(self.config["model/dropout"])

        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=100)
        self.fc2_norm = NormalizationLayer(num_features=self.fc2.out_features)
        self.fc2_dropout = DropoutLayer(self.config["model/dropout"])

        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=50)
        self.fc3_norm = NormalizationLayer(num_features=self.fc3.out_features)
        self.fc3_dropout = DropoutLayer(self.config["model/dropout"])

        # Heads
        self.heads = Heads(self.config, features_dim=self.fc3.out_features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.squeeze()

        x = self.fc1_dropout(self.F(self.fc1_norm(self.fc1(x)), inplace=True))
        x = self.fc2_dropout(self.F(self.fc2_norm(self.fc2(x)), inplace=True))
        x = self.fc3_dropout(self.F(self.fc3_norm(self.fc3(x)), inplace=True))

        logits = self.heads(x)

        return logits
