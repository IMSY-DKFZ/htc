# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

# Example from the website: https://spectralverse-heidelberg.org/xeno-spectral/
#
from htc import DataPath

if __name__ == "__main__":
    # You can load every image based on its unique name
    path = DataPath.from_image_name("P190#2023_10_27_14_33_29")

    # HSI cube format: (height, width, channels)
    assert path.read_cube().shape == (480, 640, 100)

    # Semantic annotation
    assert path.read_segmentation("semantic#primary").shape == (480, 640)

    # Retrieve arbitrary meta information (like the perfusion state)
    assert path.meta("perfusion_state") == "malperfused"

    # Or the species name
    assert DataPath.from_image_name("R047#2025_01_26_12_14_55").meta("species_name") == "rat"
