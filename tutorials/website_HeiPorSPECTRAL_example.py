# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

# Example from the website: https://heiporspectral.org/
#
import numpy as np

from htc import DataPath, LabelMapping

if __name__ == "__main__":
    # You can load every image based on its unique name
    path = DataPath.from_image_name("P086#2021_04_15_09_22_02")

    # HSI cube format: (height, width, channels)
    assert path.read_cube().shape == (480, 640, 100)

    # Annotated region of the selected annotator
    mask = path.read_segmentation("polygon#annotator1")
    assert mask.shape == (480, 640)

    # Additional meta information about the image
    assert path.meta("label_meta") == {"spleen": {"situs": 1, "angle": 0, "repetition": 1}}

    # Tivita parameter images are available as well
    sto2 = path.compute_sto2()
    assert sto2.shape == (480, 640)

    # Example: average StO2 value of the annotated spleen area for annotator1
    # The dataset_settings.json file defines the global name to index mapping
    spleen_index = LabelMapping.from_path(path).name_to_index("spleen")
    assert round(np.mean(sto2[mask == spleen_index]), 2) == 0.44
