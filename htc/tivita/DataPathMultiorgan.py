# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import functools
from collections.abc import Callable, Iterator
from pathlib import Path

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings


# We use a decorator to wrap some of the path functions. This is important for the files
# which are stored in the overlap folder because then the image data is stored in a different
# dataset (due to multiple annotations)
def use_overlap_path(method: Callable) -> Callable:
    @functools.wraps(method)
    def _use_overlap_path(self):
        if self.is_overlap:
            image_dir_old = self.image_dir
            potential_data_dirs = [
                settings.data_dirs["PATH_Tivita_multiorgan_semantic"],
                settings.data_dirs["PATH_Tivita_multiorgan_masks"],
            ]
            image_dir_new_found = False

            for potential_data_dir in potential_data_dirs:
                image_dir_new = potential_data_dir / "subjects" / self.subject_name / self.timestamp

                if image_dir_new.exists():
                    image_dir_new_found = True
                    break

            assert (
                image_dir_new_found
            ), f"Cannot find the overlap image name in any of the potential dataset dirs {potential_data_dirs}"

            self.image_dir = image_dir_new
            res = method(self)
            self.image_dir = image_dir_old
        else:
            res = method(self)

        return res

    return _use_overlap_path


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

    @use_overlap_path
    def cube_path(self) -> Path:
        return super().cube_path()

    @use_overlap_path
    def camera_meta_path(self) -> Path:
        return super().camera_meta_path()

    @use_overlap_path
    def rgb_path_reconstructed(self) -> Path:
        return super().rgb_path_reconstructed()

    @staticmethod
    def iterate(
        data_dir: str | Path,
        filters: list[Callable[["DataPathMultiorgan"], bool]] = None,
        annotation_name: str | list[str] = None,
    ) -> Iterator["DataPathMultiorgan"]:
        data_dir, filters, annotation_name = DataPath._iterate_parse_inputs(data_dir, filters, annotation_name)

        if (data_dir / "subjects").exists():
            dataset_settings = DatasetSettings(data_dir / "dataset_settings.json")
            intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)

            # Multi-organ data
            for subject_name_path in sorted(data_dir.glob("subjects/*")):
                for image_dir in sorted(subject_name_path.iterdir()):
                    path = DataPathMultiorgan(image_dir, data_dir, intermediates_dir, dataset_settings, annotation_name)
                    if all(f(path) for f in filters):
                        yield path
        else:
            yield from DataPathTivita.iterate(data_dir, filters, annotation_name)
