# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_font(font_size: int = 16, font_family: str = "serif") -> None:
    """
    Sets the font properties for matplotlib to use the Libertinus font.

    Args:
        font_size: Default font size which is the baseline for relative font sizes.
        font_family: Whether serif or sans-serif font should be used.
    """
    # Register all font files in the directory of this script file (https://stackoverflow.com/a/43647344/2762258)
    font_files = font_manager.findSystemFonts(fontpaths=[Path(__file__).parent])
    for font in font_files:
        font_manager.fontManager.addfont(font)

    # Set the font settings
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.serif"] = ["Libertinus Serif"]
    plt.rcParams["font.sans-serif"] = ["Libertinus Sans"]
    plt.rcParams["font.cursive"] = ["Libertinus Serif"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Libertinus Serif"
    plt.rcParams["mathtext.it"] = "Libertinus Serif:italic"
    plt.rcParams["mathtext.bf"] = "Libertinus Serif:bold"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size - 2
    plt.rcParams["ytick.labelsize"] = font_size - 2

    # Make sure the font is embedded in the pdf file (https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib/22809802)
    plt.rc("pdf", fonttype=42)
