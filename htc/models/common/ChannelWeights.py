# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from htc.utils.Config import Config


class ChannelWeights(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.channel_weights = nn.Parameter(
            torch.ones(self.config["input/n_channels"])
        )  # Start with equal contribution to each channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weights = self.channel_weights

        # Broadcast shape to x
        idx_channel_dim = list(x.shape).index(self.config["input/n_channels"])
        for i in range(len(x.shape)):
            if i < idx_channel_dim:
                channel_weights = channel_weights.unsqueeze(dim=0)
            elif i > idx_channel_dim:
                channel_weights = channel_weights.unsqueeze(dim=-1)

        x = channel_weights * x

        return x
