# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np

from htc.utils.LabelMapping import LabelMapping


class DataPathArray:
    def __init__(self, cube: np.ndarray, seg: np.ndarray = None, label_mapping: LabelMapping = None):
        self.cube = cube
        self.seg = seg
        self.dataset_settings = {"spatial_shape": cube.shape[:2]}
        if label_mapping is not None:
            self.dataset_settings["label_mapping"] = label_mapping.mapping_name_index
            self.dataset_settings["last_valid_label_index"] = label_mapping.last_valid_label_index

    def image_name(self) -> str:
        return "test_array"

    def image_name_annotations(self) -> str:
        return "test_array@test_annotations"

    def read_cube(self, *args, **kwargs) -> np.ndarray:
        return self.cube

    def read_rgb_reconstructed(self, *args, **kwargs) -> np.ndarray:
        return self.cube[..., :3]

    def read_segmentation(self, *args, **kwargs) -> np.ndarray | None:
        return self.seg
