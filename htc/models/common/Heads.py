# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from htc.models.common.utils import get_n_classes
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper


class Heads(nn.Module):
    def __init__(self, config: Config, features_dim: torch.Size | int):
        """
        Adds multiple heads which calculate its prediction based on the features of a network.

        Args:
            config: The configuration object.
            features_dim: The feature dimension of the network (e.g. number of neurons of the last layer in a FNN).
        """
        super().__init__()

        self.heads = nn.ModuleDict()
        for head_name in config.get("model/heads", ["class"]):
            if head_name == "class":
                self.heads[head_name] = nn.Linear(in_features=features_dim, out_features=get_n_classes(config))
            elif head_name in ["camera_index", "subject_index", "species_index"]:
                self.heads[head_name] = nn.Linear(
                    in_features=features_dim, out_features=len(DomainMapper.from_config(config)[head_name])
                )
            else:
                raise ValueError(f"Invalid head name {head_name}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = {name: head(x) for name, head in self.heads.items()}

        return logits
