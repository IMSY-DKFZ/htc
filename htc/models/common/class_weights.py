# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.common.utils import get_n_classes
from htc.utils.Config import Config


def calculate_class_weights(
    config: Config, label_indices: torch.Tensor = None, label_counts: torch.Tensor = None
) -> torch.Tensor:
    """
    Implements different class weight calculation methods.

    Args:
        config: Configuration object which must contain at least a model/class_weight_method key.
        label_indices: Vector with all labels used during training.
        label_counts: Vector with corresponding counts for each label.

    Returns:
        Estimated class weights normalized to a probability vector.
    """
    n_classes = get_n_classes(config)

    if config["model/class_weight_method"] == "1" or not config["model/class_weight_method"]:
        return torch.ones(n_classes)

    assert (
        label_indices is not None and label_counts is not None
    ), "Cannot calculate class weights without label information"
    assert len(label_indices) == len(label_counts), "Label ids and counts must match"

    n_pixels = label_counts.sum()
    class_weights = torch.zeros(n_classes)

    # n = total counts
    # m = counts for the current label
    if config["model/class_weight_method"] == "(n-m)∕n":  # ATTENTION: unicode char 2215 (∕) used to avoid path issues
        class_weights[label_indices] = (n_pixels - label_counts.float()) / n_pixels
        class_weights = class_weights / torch.sum(class_weights)
    elif config["model/class_weight_method"] == "1∕m":
        class_weights[label_indices] = 1 / label_counts.float()
        class_weights = class_weights / torch.sum(class_weights)
    elif config["model/class_weight_method"] == "softmin":
        counts_normalized = label_counts.float() / n_pixels
        exp_counts = torch.exp(config["model/softmin_scaling"] * counts_normalized)
        class_weights[label_indices] = exp_counts / exp_counts.sum()
    else:
        raise ValueError(f'{config["model/class_weight_method"]} is not a valid class weight calculation method')

    return class_weights
