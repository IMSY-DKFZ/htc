# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Union

from matplotlib.colors import LinearSegmentedColormap, to_hex


def tivita_colorscale(library: str) -> Union[list, LinearSegmentedColormap]:
    """
    Returns the official colormap as used in the Tivita parameter images.

    Args:
        library: The library for which the colormap is needed. Either `plotly` or `matplotlib`.

    Returns: The colormap.
    """
    # Colors obtained via from the original Tivita images via color picker
    cmap_list = [
        (0, (0, 0, 101 / 255)),
        (0.1, (0, 0, 175 / 255)),
        (0.2, (0, 0, 253 / 255)),
        (0.3, (0, 124 / 255, 130 / 255)),
        (0.4, (0, 253 / 255, 1 / 255)),
        (0.5, (124 / 255, 1, 0)),
        (0.6, (251 / 255, 1, 0)),
        (0.7, (1, 130 / 255, 0)),
        (0.8, (1, 4 / 255, 0)),
        (0.9, (179 / 255, 0, 0)),
        (1.0, (102 / 255, 0, 0)),
    ]

    if library == "plotly":
        return [(pos, to_hex(c)) for pos, c in cmap_list]
    elif library == "matplotlib":
        return LinearSegmentedColormap.from_list("tivita", cmap_list, gamma=1)
    else:
        raise ValueError(f"Invalid library: {library}")
