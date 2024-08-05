# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from htc.models.common.HSI3dChannel import HSI3dChannel
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import get_n_classes, model_input_channels
from htc.utils.Config import Config


class ModelImage(HTCModel):
    def __init__(self, config: Config, channels: int = None):
        super().__init__(config)

        if self.config["model/input_channels"]:
            channels = self.config["model/input_channels"]

        if self.config["model/channel_preprocessing"]:
            self.channel_preprocessing = HSI3dChannel(self.config)
            channels = self.channel_preprocessing.output_channels()
        else:
            self.channel_preprocessing = nn.Identity()
            channels = model_input_channels(self.config) if channels is None else channels

        ArchitectureClass = getattr(smp, self.config["model/architecture_name"])
        self.architecture = ArchitectureClass(
            classes=get_n_classes(self.config), in_channels=channels, **self.config.get("model/architecture_kwargs", {})
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_preprocessing(x)
        x = self.architecture(x)

        return x
