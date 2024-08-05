# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

import htc.models.common.functions
from htc.models.common.ChannelWeights import ChannelWeights
from htc.models.common.utils import model_input_channels
from htc.utils.Config import Config


class HSI3dChannel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if self.config["model/channel_preprocessing/channel_weights"]:
            self.channel_weights = ChannelWeights(config)
        else:
            self.channel_weights = nn.Identity()

        self.F = htc.models.common.functions.activation_functions_module[
            self.config["model/channel_preprocessing/activation_function"]
        ]

        if self.config["model/channel_preprocessing/normalization"]:
            NormalizationLayer = getattr(nn, f'{config["model/channel_preprocessing/normalization"]}3d')
        else:
            NormalizationLayer = nn.Identity

        # Defining the 3D convolutions is a bit complicated since there are multiple issues
        # - There is no easy parameter to set to get a specific number of output channels after the 3D convolution. This depends on the number of layers, kernel size, stride, etc.
        # - The memory consumption of these operations is crucial as there are many activations to be stored. This makes e.g. a higher stride value in the beginning necessary so that the channel dimension is reduced very early

        remaining_channels = self.config.get("model/channel_preprocessing/remaining_channels", 40)
        if remaining_channels == 5:
            conv1 = nn.Conv3d(1, 10, (10, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(2, 1, 1))
            conv1_norm = NormalizationLayer(num_features=conv1.out_channels)

            conv2 = nn.Conv3d(
                conv1.out_channels, 5, (10, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(2, 1, 1)
            )
            conv2_norm = NormalizationLayer(num_features=conv2.out_channels)

            conv3 = nn.Conv3d(
                conv2.out_channels, 1, (10, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(2, 1, 1)
            )
            conv3_norm = NormalizationLayer(num_features=conv3.out_channels)

            self.layers = nn.Sequential(conv1, conv1_norm, conv2, conv2_norm, conv3, conv3_norm)
        elif remaining_channels == 16:
            conv1 = nn.Conv3d(1, 5, kernel_size=(5, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(2, 1, 1))
            conv1_norm = NormalizationLayer(num_features=conv1.out_channels)
            pool1 = nn.AvgPool3d(kernel_size=(2, 1, 1), count_include_pad=False)

            conv2 = nn.Conv3d(
                conv1.out_channels,
                5,
                kernel_size=(5, 3, 3),
                padding=(0, 1, 1),
                padding_mode="circular",
                stride=(1, 1, 1),
            )
            conv2_norm = NormalizationLayer(num_features=conv2.out_channels)

            conv3 = nn.Conv3d(
                conv2.out_channels,
                1,
                kernel_size=(5, 3, 3),
                padding=(0, 1, 1),
                padding_mode="circular",
                stride=(1, 1, 1),
            )
            conv3_norm = NormalizationLayer(num_features=conv3.out_channels)

            self.layers = nn.Sequential(conv1, conv1_norm, pool1, conv2, conv2_norm, conv3, conv3_norm)
        elif remaining_channels == 38:
            conv1 = nn.Conv3d(1, 5, (10, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(2, 1, 1))
            conv1_norm = NormalizationLayer(num_features=conv1.out_channels)

            conv2 = nn.Conv3d(
                conv1.out_channels, 2, (5, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(1, 1, 1)
            )
            conv2_norm = NormalizationLayer(num_features=conv2.out_channels)

            conv3 = nn.Conv3d(
                conv2.out_channels, 1, (5, 3, 3), padding=(0, 1, 1), padding_mode="circular", stride=(1, 1, 1)
            )
            conv3_norm = NormalizationLayer(num_features=conv3.out_channels)

            self.layers = nn.Sequential(
                conv1,
                conv1_norm,
                self.F(inplace=True),
                conv2,
                conv2_norm,
                self.F(inplace=True),
                conv3,
                conv3_norm,
                self.F(inplace=True),
            )
        elif remaining_channels == 40:
            layer_sizes = [1, 10, 5, 1]
            strides = [None, 2, 1, 1]
            layers = []
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Conv3d(layer_sizes[i - 1], layer_sizes[i], (5, 1, 1), stride=(strides[i], 1, 1)))
                layers.append(NormalizationLayer(num_features=layer_sizes[i]))
                layers.append(self.F(inplace=True))

            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError(f"The specified number of remaining channels {remaining_channels} is not supported")

    def output_channels(self) -> int:
        """
        Returns: Number of output channels which remain after the 3D convolution.
        """
        with torch.no_grad():
            return self(torch.rand(2, model_input_channels(self.config), 2, 2)).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [3, 100, 480, 640]
        x = self.channel_weights(x)

        x = x.unsqueeze(dim=1)
        x = self.layers(x)
        x = x.squeeze(dim=1)

        # x.shape = [3, remaining_channels, 480, 640]
        return x
