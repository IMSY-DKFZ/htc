# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from htc.models.common.HSI3dChannel import HSI3dChannel
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import get_n_classes
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
            channels = self.config["input/n_channels"] if channels is None else channels

        # Standardize each image in the batch
        if self.config["model/image_normalization"]:
            # We use GroupNorm with only one group instead of LayerNorm since for the former we do not need to know the image size
            # Higher epsilon give more accurate results for small numbers (e.g. L1 normalized data)
            self.image_normalization = nn.GroupNorm(1, channels, eps=1e-8)
        else:
            self.image_normalization = nn.Identity()

        ArchitectureClass = getattr(smp, self.config["model/architecture_name"])
        self.architecture = ArchitectureClass(
            classes=get_n_classes(self.config), in_channels=channels, **self.config.get("model/architecture_kwargs", {})
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_normalization(x)
        x = self.channel_preprocessing(x)
        x = self.architecture(x)

        return x
