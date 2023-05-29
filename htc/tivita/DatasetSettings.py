# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from htc.utils.type_from_string import type_from_string


class DatasetSettings:
    def __init__(self, path_or_data: Union[str, Path, dict]):
        """
        Settings of the dataset (label_mapping, shape, etc.) defined in the dataset_settings.json file of the data folder. The data is not loaded when constructing this object but only when the settings data is accessed for the first time (lazy loading).

        The dataset settings can be conveniently accessed via data paths:
        >>> from htc import DataPath
        >>> path = DataPath.from_image_name("P041#2019_12_14_12_00_16")
        >>> path.dataset_settings["dataset_name"]
        '2021_02_05_Tivita_multiorgan_semantic'
        >>> path.dataset_settings["shape"]
        (480, 640, 100)

        It is also possible to load the settings explicitly based on the path to the data directory:
        >>> from htc.settings import settings
        >>> dsettings = DatasetSettings(settings.data_dirs.semantic)
        >>> dsettings["shape"]
        (480, 640, 100)

        Args:
            path_or_data: Path (or string) to the JSON file containing the dataset settings or path to the data directory which contains the JSON file (in which case the name of the file must be dataset_settings.json). Alternatively, you can pass your settings directly as a dict.
        """
        if isinstance(path_or_data, str):
            path_or_data = Path(path_or_data)

        if type(path_or_data) == dict:
            self._data = path_or_data
            self._data_conversions()
            self._path = None
        else:
            self._data = None
            self._path = path_or_data

    def __repr__(self) -> str:
        res = (
            "Settings for the dataset"
            f" {self.settings_path.parent.name if self.settings_path is not None else '(no path available)'}\n"
        )
        res += "The following settings are available:\n"
        res += f"{list(self.data.keys())}"

        return res

    def __eq__(self, other: "DatasetSettings") -> bool:
        if self._data is None and other._data is None:
            return self.settings_path == other.settings_path
        else:
            return self.data == other.data

    def __getitem__(self, key: str) -> Any:
        assert key in self.data, f"Cannot find {key} in the dataset settings\n{self.settings_path = }\n{self.data = }"
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data[key] if key in self.data else default

    def __contains__(self, key: str) -> bool:
        return key in self.data

    @property
    def settings_path(self) -> Union[None, Path]:
        """
        Returns: The Path to the dataset_settings.json file if it exists or None if not.
        """
        if self._path is None:
            return None
        else:
            if self._path.exists():
                p = self._path
                if self._path.is_dir():
                    p /= "dataset_settings.json"

                return p if p.exists() else None
            else:
                return None

    @property
    def data(self) -> dict:
        if self._data is None:
            if self.settings_path is None:
                self._data = {}
            else:
                with self.settings_path.open(encoding="utf-8") as f:
                    self._data = json.load(f)

                self._data_conversions()

        return self._data

    def data_path_class(self) -> Union[type, None]:
        """
        Tries to infer the appropriate data path class for the current dataset. Ideally, this is defined in the dataset_settings.json file with a key data_path_class referring to a valid data path class (e.g. htc.tivita.DataPathMultiorgan>DataPathMultiorgan). If this is not the case, it tries to infer the data path class based on the dataset name or based on the files in the folder.

        Returns: Data path type or None if no match could be found.
        """
        if "data_path_class" in self:
            DataPathClass = type_from_string(self["data_path_class"])
        elif "multiorgan" in self.get("dataset_name", ""):
            from htc.tivita.DataPathMultiorgan import DataPathMultiorgan

            DataPathClass = DataPathMultiorgan
        elif "sepsis" in self.get("dataset_name", ""):
            from htc.tivita.DataPathSepsis import DataPathSepsis

            DataPathClass = DataPathSepsis
        elif self._path is not None:
            # Try to infer the data path class from the files in the directory
            if self._path.is_file() or not self._path.exists():
                dataset_dir = self._path.parent
            else:
                dataset_dir = self._path
            assert dataset_dir.exists() and dataset_dir.is_dir(), f"The dataset directory {dataset_dir} does not exist"

            files = sorted(dataset_dir.iterdir())
            if any(f.name.startswith("Cat") for f in files):
                from htc.tivita.DataPathTissueAtlas import DataPathTissueAtlas

                DataPathClass = DataPathTissueAtlas
            elif any(f.name.endswith("subjects") for f in files):
                from htc.tivita.DataPathMultiorgan import DataPathMultiorgan

                DataPathClass = DataPathMultiorgan
            elif any(f.name == "sepsis_study" for f in files):
                from htc.tivita.DataPathSepsis import DataPathSepsis

                DataPathClass = DataPathSepsis
            elif any(f.stem == "image_references" for f in files):
                from htc.tivita.DataPathReference import DataPathReference

                DataPathClass = DataPathReference
            else:
                DataPathClass = None
        else:
            DataPathClass = None

        return DataPathClass

    def pixels_image(self) -> int:
        """
        Returns: Number of pixels of one image in the dataset.
        """
        assert "shape" in self.data, "No shape information available in the dataset settings"
        return int(np.prod(self.data["spatial_shape"]))

    def _data_conversions(self) -> None:
        if "shape" in self._data:
            self._data["shape"] = tuple(self._data["shape"])
            if "shape_names" in self._data:
                names = self._data["shape_names"]
                assert (
                    "height" in names and "width" in names
                ), f"shape_names must at least include height and width (got: {names})"
                self._data["spatial_shape"] = (
                    self._data["shape"][names.index("height")],
                    self._data["shape"][names.index("width")],
                )
