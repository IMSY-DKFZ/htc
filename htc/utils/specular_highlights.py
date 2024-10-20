# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
from kornia.color import rgb_to_lab

from htc.tivita.DataPath import DataPath


def specular_highlights_mask_lab(path: DataPath, threshold: int) -> torch.Tensor:
    """
    Detect specular highlights in an image based on a threshold in the LAB color space.

    Args:
        path: The path to the image.
        threshold: The threshold value for determining the presence of specular highlights.

    Returns: A binary mask indicating the presence of specular highlights (True = specular highlight is present).
    """
    rgb = torch.from_numpy(path.read_rgb_reconstructed()) / 255
    spec_mask = rgb_to_lab(rgb.permute(2, 0, 1))[0] > threshold

    return spec_mask
