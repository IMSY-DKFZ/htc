# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch.nn as nn
import torch.nn.functional as F


def linear_activation(x, *args, **kwargs):
    return x


activation_functions = {
    "leaky_relu": F.leaky_relu,
    "relu": F.relu,
    "elu": F.elu,
    "selu": F.selu,
    "tanh": F.tanh,
    "mish": F.mish,
    "linear": linear_activation,
    None: linear_activation,
}

activation_functions_module = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
    "linear": nn.Identity,
    None: nn.Identity,
}
