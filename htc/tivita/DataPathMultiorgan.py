# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Union

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings


# We use a decorator to wrap some of the path functions. This is important for the files
# which are stored in the overlap folder because then the image data is stored in the semantic
# dataset (due to multiple annotations)
def use_semantic_path(method: Callable) -> Callable:
    @functools.wraps(method)
    def _use_semantic_path(self):
        if self.is_overlap:
            image_dir_old = self.image_dir
            image_dir_new = (
                settings.data_dirs["PATH_Tivita_multiorgan_semantic"] / "subjects" / self.subject_name / self.timestamp
            )
            assert image_dir_new.exists(), f"Cannot find the path {image_dir_new}"

            self.image_dir = image_dir_new
            res = method(self)
            self.image_dir = image_dir_old
        else:
            res = method(self)

        return res

    return _use_semantic_path


class DataPathMultiorgan(DataPath):
    def __init__(self, *args, **kwargs):
        """
        Constructs a data path for a multi-organ image and is usually used in conjunction with one of our data repositories (e.g. 2021_02_05_Tivita_multiorgan_semantic).

        This class expects the dataset to have the following structure:
        ```
        .
        ├── dataset_settings.json
        └── subjects/
            ├── subject_name/
            │   ├── image_folder/
            │   │   ├── *_SpecCube.dat
            │   │   └── [more image files]
            │   └── [more images]
            └── [more subjects]
        ```
        """
        super().__init__(*args, **kwargs)
        self.subject_name = self.image_dir.parent.name

        # For some files we have additional masks (e.g. overlap)
        self.parent_folder = self.image_dir.parents[2].name
        self.is_overlap = any(self.parent_folder.startswith(x) for x in ["overlap"])

    def build_path(self, base_folder: Path) -> Path:
        return base_folder / self.subject_name / self.timestamp

    def image_name(self) -> str:
        name = f"{self.subject_name}#{self.timestamp}"
        if self.is_overlap:
            name += f"#{self.parent_folder}"

        return name

    def image_name_parts(self) -> list[str]:
        parts = ["subject_name", "timestamp"]
        if self.is_overlap:
            parts.append("overlap")

        return parts

    @use_semantic_path
    def cube_path(self) -> Path:
        return super().cube_path()

    @use_semantic_path
    def camera_meta_path(self) -> Path:
        return super().camera_meta_path()

    @use_semantic_path
    def rgb_path_reconstructed(self) -> Path:
        return super().rgb_path_reconstructed()

    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[["DataPath"], bool]],
        annotation_name: Union[str, list[str]],
    ) -> Iterator["DataPathMultiorgan"]:
        dataset_settings = DatasetSettings(data_dir / "dataset_settings.json")
        intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)

        # Multi-organ data
        for subject_name_path in sorted(data_dir.glob("subjects/*")):
            for image_dir in sorted(subject_name_path.iterdir()):
                path = DataPathMultiorgan(image_dir, data_dir, intermediates_dir, dataset_settings, annotation_name)
                if all(f(path) for f in filters):
                    yield path
