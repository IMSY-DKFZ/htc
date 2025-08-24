# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Self

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathSepsis(DataPath):
    def __init__(self, *args, **kwargs):
        """
        Constructs a data path for the sepsis data.

        This class expects the dataset to have the following structure:
        ```
        .
        ├── dataset_settings.json
        ├── hand_posture_study/
        │   └── healthy/
        │       ├── subject_name/
        │       │   ├── location/
        │       │   │   ├── image_folder/
        │       │   │   │   ├── *_SpecCube.dat
        │       │   │   │   └── [more image files]
        │       │   │   └── [more images]
        │       │   └── [more locations]
        │       └── [more subjects]
        └── sepsis_study/
            ├── healthy/
            │   └── [...]
            ├── pancreas/
            │   └── [...]
            └── sepsis/
                └── subject_name/
                    ├── timepoint/
                    │   ├── location/
                    │   │   ├── image_folder/
                    │   │   │   ├── *_SpecCube.dat
                    │   │   │   └── [more image files]
                    │   │   └── [more locations]
                    │   └── [more timepoints]
                    └── [more subjects]
        ```
        """
        super().__init__(*args, **kwargs)

        # Find the path to the study folder
        study_dir = None
        for parent in self.image_dir.parents:
            if parent.name in ["hand_posture_study", "sepsis_study"]:
                study_dir = parent
                break
        assert study_dir is not None, "Could not find the study directory"

        self.study_dir = study_dir

        parts = self.image_dir.relative_to(self.study_dir).parts
        assert 4 <= len(parts) <= 5, f"Cannot extract the image parts from the path {self.image_dir}"

        self.health_status = parts[0]
        self.subject_name = parts[1]
        if self.health_status == "healthy":
            self.timepoint = "None"
            self.location = parts[2]
            timestamp_parts = parts[3]
        else:
            self.timepoint = parts[2]
            self.location = parts[3]
            timestamp_parts = parts[4]

        assert self.timestamp == timestamp_parts and self.location == self.image_dir.parent.name

    def build_path(self, base_folder: Path) -> Path:
        if self.timepoint == "None":
            return base_folder / self.health_status / self.subject_name / self.location / self.timestamp
        else:
            return (
                base_folder / self.health_status / self.subject_name / self.timepoint / self.location / self.timestamp
            )

    def image_name(self) -> str:
        return f"{self.health_status}#{self.subject_name}#{self.timepoint}#{self.location}#{self.timestamp}"

    def image_name_parts(self) -> list[str]:
        return list(self.image_name_typed().keys())

    def image_name_typed(self) -> dict[str, str | bool]:
        return {
            "health_status": self.health_status,
            "subject_name": self.subject_name,
            "timepoint": self.timepoint,
            "location": self.location,
            "timestamp": self.timestamp,
        }

    @classmethod
    def iterate(
        cls,
        data_dir: str | Path,
        filters: list[Callable[[Self], bool]] = None,
        annotation_name: str | list[str] = None,
    ) -> Iterator[Self]:
        data_dir, filters, annotation_name = DataPath._iterate_parse_inputs(data_dir, filters, annotation_name)

        if data_dir.name == "data":
            yield from cls.iterate(data_dir / "hand_posture_study", filters, annotation_name)
            yield from cls.iterate(data_dir / "sepsis_study", filters, annotation_name)
        else:
            root_data_dir = data_dir.parent
            dataset_settings = DatasetSettings(root_data_dir / "dataset_settings.json")
            intermediates_dir = settings.datasets.find_intermediates_dir(root_data_dir)

            if data_dir.name == "hand_posture_study":
                for subject_name_path in sorted((data_dir / "healthy").iterdir()):
                    for location_path in sorted(subject_name_path.iterdir()):
                        for image_dir in sorted(location_path.iterdir()):
                            path = cls(image_dir, root_data_dir, intermediates_dir, dataset_settings, annotation_name)
                            if all(f(path) for f in filters):
                                yield path
            elif data_dir.name == "sepsis_study":
                for health_status_path in sorted(data_dir.iterdir()):
                    if health_status_path.name in ["annotations", "meta"]:
                        continue

                    for subject_name_path in sorted(health_status_path.iterdir()):
                        if health_status_path.name == "healthy":
                            for location_path in sorted(subject_name_path.iterdir()):
                                for image_dir in sorted(location_path.iterdir()):
                                    path = cls(
                                        image_dir, root_data_dir, intermediates_dir, dataset_settings, annotation_name
                                    )
                                    if all(f(path) for f in filters):
                                        yield path
                        else:
                            for timepoint_path in sorted(subject_name_path.iterdir()):
                                for location_path in sorted(timepoint_path.iterdir()):
                                    for image_dir in sorted(location_path.iterdir()):
                                        path = cls(
                                            image_dir,
                                            root_data_dir,
                                            intermediates_dir,
                                            dataset_settings,
                                            annotation_name,
                                        )
                                        if all(f(path) for f in filters):
                                            yield path
            else:
                yield from DataPathTivita.iterate(data_dir, filters, annotation_name)
