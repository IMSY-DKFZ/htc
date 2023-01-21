# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def read_tivita_rgb(path: Path, target_shape: tuple = (480, 640)) -> np.ndarray:
    """
    Load and crop the Tivita RGB image (the one which has the heading "RGB Image").

    >>> from htc.settings import settings
    >>> from htc.tivita.DataPath import DataPath
    >>> paths = DataPath.iterate(settings.data_dirs['PATH_Tivita_multiorgan_semantic'])
    >>> path = next(iter(paths))
    >>> rgb = path.read_rgb_reconstructed()
    >>> rgb.shape
    (480, 640, 3)
    >>> rgb.dtype
    dtype('uint8')

    Args:
        path: Path to the RGB file (e.g. "[...]/2020_07_20_18_17_26/2020_07_20_18_17_26_RGB-Image.png").
        target_shape: expected shape of the read RGB image (without the black borders.)

    Returns:
        The cropped RGB image with shape (480, 640, 3) and values in [0; 255].
    """
    assert path.exists() and path.is_file(), f"RGB file {path} does not exist or is not a file"

    image = np.array(Image.open(path))
    if image.shape[:2] != target_shape:
        # Check that the borders are really "black"
        back_color = 15

        if image.shape[:2] == (513, 646):  # for Tivita first generation cameras
            assert np.all(
                image[0:30, 90:] == back_color
            ), f"{path}: Top border is not black"  # We do not check the "RGB Image" heading
            assert np.all(image[:, -3:] == back_color), f"{path}: Right border is not black"
            assert np.all(image[-3:, :] == back_color), f"{path}: Bottom border is not black"
            assert np.all(image[30:, :3] == back_color), f"{path}: Left border is not black"
            image = image[30:-3, 3:-3]  # Crop the image to remove the black border

        elif image.shape[:2] == (550, 680):  # for Tivita Surgery 2.0 camera
            assert np.all(
                image[0:50, 0:230] == back_color
            ), f"{path}: Top border is not black"  # We do not check the "RGB Image" heading
            assert np.all(image[-20:, :] == back_color), f"{path}: Bottom border is not black"
            assert np.all(image[:, :20] == back_color), f"{path}: Left border is not black"
            assert np.all(image[:, -20:] == back_color), f"{path}: Right border is not black"
            image = image[50:-20, 20:-20]

        else:
            raise ValueError(f"Unknown format of RGB images for {path}")

    assert image.shape[:2] == target_shape, f"{path}: The cropped RGB image does not have the correct shape"
    assert image.shape[2] == 3, f"{path}: The RGB image should have three color channels"

    return image


def hsi_to_rgb(cube: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Converts an HSI cube to an RGB image.

    Args:
        cube: The HSI cube (height, width, channels).
        gamma: Value for the gamma correction applied to the RGB image. The default parameter is estimated in the RGBConversion notebook and was confirmed by TIVITA.

    Returns:
        np.ndarray: RGB image (height, width, channels).
    """
    from htc.tivita.functions_official import calc_rgb

    # Create the lookup table for the gamma correction
    LUT_gamma = np.empty(256, np.uint8)
    for i in range(len(LUT_gamma)):
        LUT_gamma[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    assert len(LUT_gamma) == 256, "The LUT lookup table must provide a value for each intensity (0, 1, ..., 255)"
    assert np.min(LUT_gamma) >= 0 and np.max(LUT_gamma) <= 255, "Invalid intensity value in the LUT table"

    rgb = calc_rgb(cube, LUT_gamma)
    rgb = np.rot90(rgb, k=-1, axes=(0, 1))
    assert rgb.shape[:2] == cube.shape[:2], "Invalid shape for the RGB image"
    assert rgb.shape[2] == 3, "Invalid color dimension"
    assert rgb.dtype == np.uint8, "Invalid type for the RGB image"

    return rgb
