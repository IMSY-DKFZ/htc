# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from htc.models.common.HSI3dChannel import HSI3dChannel
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import get_n_classes, model_input_channels
from htc.utils.Config import Config


class UNetClassification(smp.Unet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Skipping the decoder of the UNet is basically a classification network
        self.decoder = None

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads."""
        features = self.encoder(x)

        labels = self.classification_head(features[-1])
        return labels


class ModelSuperpixelClassification(HTCModel):
    def __init__(self, config: Config, n_classes: int = None, **kwargs):
        super().__init__(config, **kwargs)
        if n_classes is None:
            n_classes = get_n_classes(self.config)

        if self.config["model/channel_preprocessing"]:
            self.channel_preprocessing = HSI3dChannel(self.config)
            channels = self.channel_preprocessing.output_channels()
        else:
            self.channel_preprocessing = nn.Identity()
            channels = model_input_channels(self.config)

        self.architecture = UNetClassification(
            self.config["model/encoder"],
            encoder_weights=self.config["model/encoder_weights"],  # or None if no pretraining required
            classes=n_classes,
            in_channels=channels,
            aux_params=dict(pooling="avg", dropout=self.config["model/dropout"], classes=n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_preprocessing(x)
        x = self.architecture(x)

        return x
