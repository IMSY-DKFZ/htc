# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np


def tivita_wavelengths() -> np.array:
    """
    Returns: The wavelength for each channel of a Tivita image.
    """
    # "The spatial resolution can principally be in the megapixel range, but is reduced to a standard range of 640 × 480 pixel in practice, the     spectral resolution is approx. 5 nm in the range from 500 to 1000 nm, generating 100 spectral values"
    # https://pubmed.ncbi.nlm.nih.gov/29522415/
    return np.linspace(500, 1000, 100)


def read_tivita_hsi(path: Path, normalization: int | None = None) -> np.ndarray:
    """
    Load the Tivita data cube as Numpy array.

    Note: If possible, it is recommended to use the DataPath class instead of this (low-level) function. There, you can use the `read_cube()` method to read the cube.

    >>> from htc.settings import settings
    >>> from htc.tivita.DataPath import DataPath
    >>> paths = DataPath.iterate(settings.data_dirs["PATH_Tivita_multiorgan_semantic"])
    >>> path = next(iter(paths))
    >>> cube = path.read_cube()
    >>> cube.shape
    (480, 640, 100)
    >>> cube.dtype
    dtype('float32')

    Args:
        path: Path to the cube file (e.g. "[...]/2020_07_20_18_17_26/2020_07_20_18_17_26_SpecCube.dat").
        normalization: If not None, apply normalization to the data with the given order (e.g. normalization=1 equals L1 normalization, normalization=2 is L2 normalization).

    Returns: Cube data as array of shape (480, 640, 100).
    """
    assert path.exists() and path.is_file(), f"Data cube {path} does not exist or is not a file"

    shape = np.fromfile(path, dtype=">i", count=3)  # Read shape of HSI cube
    cube = np.fromfile(
        path, dtype=">f", offset=12
    )  # Read 1D array in big-endian binary format and ignore first 12 bytes which encode the shape
    cube = cube.reshape(*shape)  # Reshape to data cube
    cube = np.flip(cube, axis=1)  # Flip y-axis to match RGB image coordinates

    cube = np.swapaxes(cube, 0, 1)  # Consistent image shape (height, width)
    cube = cube.astype(np.float32)  # Consistently convert to little-endian

    if normalization is not None:
        cube = cube / np.linalg.norm(cube, ord=normalization, axis=2, keepdims=True)
        cube = np.nan_to_num(cube, copy=False)

    return cube


def read_tivita_dark(path: Path) -> np.ndarray:
    """
    Load the dark pattern file which is used by Tivita cameras for white and dark correction.

    The dark pattern is averaged over the acquisition time and does therefore not have dark values for each pixel position, i.e. the 640 dimension is missing.

    >>> from htc.settings import settings
    >>> dark = read_tivita_dark(settings.data_dirs.studies / "white_balances/0615-00023/2021_12_15_08_32_23_DarkPattern.dpic")
    >>> dark.shape
    (480, 100)

    Args:
        path: Path to the dark pattern file (e.g. 2021_12_15_08_32_23_DarkPattern.dpic).

    Returns: Dark pattern with double values and shape (480, 100).
    """
    assert path.exists() and path.is_file(), f"The path {path} does not exist or is not a file"

    shape = np.fromfile(path, dtype=">i", count=2)
    dark = np.fromfile(path, dtype=">d", offset=8)
    dark = dark.reshape(*shape)
    dark = dark.astype(np.float64)  # Consistently convert to little-endian

    return dark
