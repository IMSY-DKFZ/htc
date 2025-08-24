# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Self

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathTivita(DataPath):
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructs a generic data path for any kind of Tivita hyperspectral image folder.
        """
        super().__init__(*args, **kwargs)
        self.attributes = list(self.image_dir.relative_to(self.data_dir).parts[:-1])

    def build_path(self, base_folder: Path) -> Path:
        return base_folder / "/".join([*self.attributes, self.timestamp])

    @classmethod
    def iterate(
        cls,
        data_dir: str | Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: str | list[str] = None,
    ) -> Iterator[Self]:
        data_dir, filters, annotation_name = DataPath._iterate_parse_inputs(data_dir, filters, annotation_name)

        # Settings of the dataset (shapes etc.) can be referenced by the DataPaths
        intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)

        dataset_settings_dict = {}
        parent_paths = list(data_dir.parents)
        parent_paths.reverse()
        for p in parent_paths:
            if (p / "dataset_settings.json").exists():
                dataset_settings_dict[p] = DatasetSettings(p / "dataset_settings.json")

        # Keep a list of used image folders in case a folder contains both a cube file and a tiv archive
        used_folders = set()
        for root, dirs, files in os.walk(data_dir):
            dirs.sort()  # Recurse in sorted order
            if "dataset_settings.json" in files:
                dataset_settings_dict[root] = DatasetSettings(Path(root) / "dataset_settings.json")
            for f in sorted(files):
                if f.endswith(("SpecCube.dat", ".tiv")) and root not in used_folders:
                    if len(dataset_settings_dict) == 0:
                        dataset_settings = None
                    else:
                        dataset_settings = list(dataset_settings_dict.values())[
                            -1
                        ]  # last dict item should be closest to path
                    path = cls(Path(root), data_dir, intermediates_dir, dataset_settings, annotation_name)
                    if all(f(path) for f in filters):
                        yield path
                    used_folders.add(root)
