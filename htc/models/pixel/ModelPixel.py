# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

import htc.models.common.functions
from htc.models.common.Heads import Heads
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import model_input_channels
from htc.utils.Config import Config


class ModelPixel(HTCModel):
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

        channels_base = self.config.get("model/channels_base", 64)

        # CNN
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels_base, kernel_size=5)
        self.conv1_norm = NormalizationLayer(num_features=self.conv1.out_channels)
        self.pool1 = nn.AvgPool1d(kernel_size=2, count_include_pad=False)

        self.conv2 = nn.Conv1d(in_channels=self.conv1.out_channels, out_channels=channels_base // 2, kernel_size=5)
        self.conv2_norm = NormalizationLayer(num_features=self.conv2.out_channels)
        self.pool2 = nn.AvgPool1d(kernel_size=2, count_include_pad=False)

        self.conv3 = nn.Conv1d(in_channels=self.conv2.out_channels, out_channels=channels_base // 4, kernel_size=5)
        self.conv3_norm = NormalizationLayer(num_features=self.conv3.out_channels)
        self.pool3 = nn.AvgPool1d(kernel_size=2, count_include_pad=False)

        # FNN
        # The adaptive pooling layer ensures that the output of the conv layers always has the same length
        # This allows a different number of channels to be used as input which could be helpful for pretraining
        # If the input already has the correct input size, the adaptive layer does not change the conv output
        in_dim = self._conv_output_features(model_input_channels(self.config))
        self.adaptive_conv_reduction = nn.AdaptiveAvgPool1d(in_dim)

        self.fc1 = nn.Linear(in_features=in_dim, out_features=100)
        self.fc1_norm = NormalizationLayer(num_features=self.fc1.out_features)
        self.fc1_dropout = DropoutLayer(self.config["model/dropout"])

        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=50)
        self.fc2_norm = NormalizationLayer(num_features=self.fc2.out_features)
        self.fc2_dropout = DropoutLayer(self.config["model/dropout"])

        # Heads
        self.heads = Heads(self.config, features_dim=self.fc2.out_features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encode(x)
        logits = self.heads(x)

        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(dim=1)  # Single channel

        x = self.F(self.conv1_norm(self.conv1(x)), inplace=True)
        x = self.pool1(x)
        x = self.F(self.conv2_norm(self.conv2(x)), inplace=True)
        x = self.pool2(x)
        x = self.F(self.conv3_norm(self.conv3(x)), inplace=True)
        x = self.pool3(x)
        # torch.Size([batch_size, 16, 9])

        shape = x.shape[1:]  # Everything except batch
        x = x.view(-1, torch.prod(torch.tensor(shape)))
        x = self.adaptive_conv_reduction(x)

        x = self.fc1_dropout(self.F(self.fc1_norm(self.fc1(x)), inplace=True))
        x = self.fc2_dropout(self.F(self.fc2_norm(self.fc2(x)), inplace=True))

        return x

    def _conv_output_features(self, input_channels: int, layers: torch.nn.Sequential = None) -> int:
        """
        Helper function to determine the output channels after the convolution operations.

        Args:
            input_channels: Number of channels in the input.
            layers: Sequential list of layers to apply to the input.

        Returns: Number of channels which remain after the last convolution layer.
        """
        with torch.no_grad():
            x = torch.rand(1, 1, input_channels)

            if layers is None:
                x = self.pool1(self.conv1(x))
                x = self.pool2(self.conv2(x))
                x = self.pool3(self.conv3(x))
            else:
                x = layers(x)

            x = x.squeeze(dim=0)

            return x.numel()
