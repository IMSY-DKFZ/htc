# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from htc import settings
from htc.utils.LabelMapping import LabelMapping


class DataPathArray:
    def __init__(
        self,
        cube: np.ndarray,
        seg: np.ndarray = None,
        label_mapping: LabelMapping = None,
        create_dataset: bool = False,
        path: Path = None,
        subject_name: str = None,
        timestamp: str = None,
    ):
        self.cube = cube
        self.seg = seg
        self.dataset_settings = {"spatial_shape": cube.shape[:2]}
        self.data_dir = "TestArrayDataset"
        self.subject_name = subject_name
        self.timestamp = timestamp
        self.path = path

        if label_mapping is not None:
            self.dataset_settings["label_mapping"] = label_mapping.mapping_name_index
            self.dataset_settings["last_valid_label_index"] = label_mapping.last_valid_label_index

        if create_dataset:
            if any([self.path is None, self.subject_name is None, self.timestamp is None]):
                settings.log.error(
                    "Please provide path, subject_name and timestamp arguments when the create_dataset argument has"
                    " been set"
                )
            else:
                self._create_folders(self.path)

    def image_name(self) -> str:
        return f"{self.subject_name}#{self.timestamp}"

    def image_name_annotations(self) -> str:
        return "test_array@test_annotations"

    def read_cube(self, *args, **kwargs) -> np.ndarray:
        return self.cube

    def read_rgb_reconstructed(self, *args, **kwargs) -> np.ndarray:
        return self.cube[..., :3]

    def read_segmentation(self, *args, **kwargs) -> np.ndarray | None:
        return self.seg

    @staticmethod
    def annotation_names(return_settings_default=True) -> list[str]:
        return []

    def _create_folders(self, path: Path) -> None:
        # Create a dataset structure for testing with mock examples
        data_dir = path / "data"
        data_dir.mkdir()
        self.data_dir = data_dir

        subjects_dir = data_dir / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)

        (subjects_dir / self.subject_name).mkdir(parents=True, exist_ok=True)
        (subjects_dir / self.subject_name / self.timestamp).mkdir(parents=True, exist_ok=True)

        intermediates_dir = path / "intermediates"
        nearest_neighbor_indices_path = intermediates_dir / "nearest_neighbor_indices"
        nearest_neighbor_indices_path.mkdir(parents=True, exist_ok=True)

        (intermediates_dir / "tables").mkdir(parents=True, exist_ok=True)
