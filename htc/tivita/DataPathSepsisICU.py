# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Self

import numpy as np

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathSepsisICU(DataPath):
    def __init__(self, *args, **kwargs):
        """
        Constructs a data path for the sepsis ICU data.

        This class expects the dataset to have the following structure:
        ```
        .
        ├── data/
        │    ├── calibrations/
        │    │   ├── colorchecker
        │    │   │   └── [...]
        │    │   └── white
        │    │   │   └── [...]
        │    ├── subjects/
        │    │   └── [date]/
        │    │       ├── image_folder/
        │    │       │   ├── *_SpecCube.dat
        │    │       │   └── [more image files]
        │    │       └── [more image_folder]
        │    └── dataset_settings.json
        └── intermediates/
            └── [...]
        ```
        """
        super().__init__(*args, **kwargs)
        assert self.patient_meta_path() is not None, f"Patient meta data file missing for {self}"
        self.subject_name = self.patient_meta_path().name.removesuffix(".xml")
        if "calibration" in self.subject_name:
            self.image_cat = "calibrations"
        else:
            self.image_cat = "subjects"

    def image_name(self) -> str:
        return f"{self.subject_name}#{self.timestamp}"

    def image_name_parts(self) -> list[str]:
        return ["subject_name", "timestamp"]

    def subject_number(self) -> int | float:
        if self.image_cat == "subjects":
            return int(self.subject_name[1:])
        else:
            return np.nan

    @classmethod
    def iterate(
        cls,
        data_dir: str | Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: str | list[str] = None,
    ) -> Iterator[Self]:
        data_dir, filters, annotation_name = DataPath._iterate_parse_inputs(data_dir, filters, annotation_name)

        if (data_dir / "calibrations").exists() and (data_dir / "subjects").exists():
            subject_dir = data_dir / "subjects"
            yield from cls.iterate(subject_dir, filters, annotation_name)
            cal_dir = data_dir / "calibrations"
            yield from cls.iterate(cal_dir, filters, annotation_name)
        elif data_dir.name == "calibrations":
            white_dir = data_dir / "white"
            yield from cls.iterate(white_dir, filters, annotation_name)
            colorchecker_dir = data_dir / "colorchecker_classic_video_passport"
            yield from cls.iterate(colorchecker_dir, filters, annotation_name)
            probands_dir = data_dir / "probands"
            yield from cls.iterate(probands_dir, filters, annotation_name)
        else:
            if data_dir.name == "subjects":
                root_data_dir = data_dir.parent
                dataset_settings = DatasetSettings(root_data_dir / "dataset_settings.json")
                intermediates_dir = settings.datasets.find_intermediates_dir(root_data_dir)

                for date_dir in sorted(data_dir.iterdir()):
                    for image_dir in sorted(date_dir.iterdir()):
                        path = cls(image_dir, root_data_dir, intermediates_dir, dataset_settings, annotation_name)
                        if all(f(path) for f in filters):
                            yield path
            else:
                image_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
                if len(image_dirs) > 0:
                    root_data_dir = data_dir.parent.parent
                    dataset_settings = DatasetSettings(root_data_dir / "dataset_settings.json")
                    intermediates_dir = settings.datasets.find_intermediates_dir(root_data_dir)

                    for image_dir in image_dirs:
                        path = cls(image_dir, root_data_dir, intermediates_dir, dataset_settings, annotation_name)
                        if all(f(path) for f in filters):
                            yield path
                else:
                    yield from DataPathTivita.iterate(data_dir, filters, annotation_name)
