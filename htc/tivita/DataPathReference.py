# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Union

import pandas as pd

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathReference(DataPath):
    _references_cache = None

    def __init__(self, network_path: Path, dataset_name: str, *args, **kwargs) -> None:
        """
        Constructs a a data path object which has only a reference to an image folder on the network drive.
        The information about the images is stored in a image_references table in the dataset folder.

        Note: This implementation relies on unique timestamps.

        Args:
            network_path: Relative path to the image on the network drive.
            dataset_name: Name of the dataset where the image comes from.
        """
        self.network_path = network_path
        super().__init__(settings.data_dirs.network_data / self.network_path, *args, **kwargs)
        self.dataset_name = dataset_name

    def image_name(self) -> str:
        return f"ref#{self.dataset_name}#{self.timestamp}"

    def image_name_parts(self):
        return list(self.image_name_typed())

    def image_name_typed(self) -> dict[str, Any]:
        return {
            "ref_identifier": "ref",
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
        }

    def build_path(self, base_folder: Path) -> Path:
        return base_folder / self.network_path

    @staticmethod
    def _cache() -> dict:
        if DataPathReference._references_cache is None:
            DataPathReference._references_cache = {}

            data_dir = settings.data_dirs.unsorted
            df = pd.read_feather(data_dir / "image_references.feather")
            dsettings = DatasetSettings(data_dir / "dataset_settings.json")

            for image_name, network_path, dataset_name in zip(df["image_name"], df["network_path"], df["dataset_name"]):
                DataPathReference._references_cache[image_name] = {
                    "network_path": network_path,
                    "dataset_name": dataset_name,
                    "dsettings": dsettings,
                    "data_dir": settings.data_dirs.network_data / dataset_name / "data",
                    "intermediates_dir": settings.data_dirs.network_data / dataset_name / "intermediates",
                }

        return DataPathReference._references_cache

    @staticmethod
    def from_image_name(image_name: str, annotation_name: Union[str, list[str]]) -> "DataPathReference":
        cache = DataPathReference._cache()
        assert image_name in cache, f"Could not find the image {image_name} in the reference table"

        match = cache[image_name]
        return DataPathReference(
            match["network_path"],
            match["dataset_name"],
            match["data_dir"],
            match["intermediates_dir"],
            match["dsettings"],
            annotation_name,
        )

    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[["DataPath"], bool]],
        annotation_name: Union[str, list[str]],
    ) -> Iterator["DataPathReference"]:
        dataset_settings = DatasetSettings(data_dir / "dataset_settings.json")
        df_references = pd.read_feather(data_dir / "image_references.feather")
        intermediates_dir = settings.data_dirs.find_intermediates_dir(data_dir)

        for i, row in df_references.iterrows():
            path = DataPathReference(
                row["network_path"], row["dataset_name"], data_dir, intermediates_dir, dataset_settings, annotation_name
            )
            if all([f(path) for f in filters]):
                yield path
