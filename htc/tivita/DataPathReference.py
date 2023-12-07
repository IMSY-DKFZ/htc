# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.blosc_compression import decompress_file


class DataPathReference(DataPath):
    _references_cache = None

    def __init__(self, network_path: Path, dataset_name: str, *args, **kwargs):
        """
        Constructs a a data path object which has only a reference to an image folder on the network drive.
        The information about the images is stored in a image_references table in the dataset folder.

        Note: This implementation relies on unique timestamps.

        Args:
            network_path: Relative path to the image on the network drive.
            dataset_name: Name of the dataset where the image comes from.
        """
        self.network_path = network_path
        self.dataset_name = dataset_name

        if settings.datasets.network_data is None:
            super().__init__(None, *args, **kwargs)
            self.timestamp = self.network_path.name
        else:
            local_dataset = settings.datasets[self.dataset_name]
            if local_dataset is not None:
                # If the dataset is locally available (e.g. Tivita studies), then use it (always faster than the network drive)
                image_dir = settings.datasets[self.dataset_name]["path_dataset"] / Path(self.network_path).relative_to(
                    self.dataset_name
                )
                assert image_dir.exists(), (
                    f"Could not find the image directory {image_dir} locally but the dataset is available. This could"
                    f" mean that the dataset {self.dataset_name} is not in sync with the network drive"
                )

                kwargs["data_dir"] = local_dataset["path_data"]
                kwargs["intermediates_dir"] = local_dataset["path_intermediates"]
            else:
                image_dir = settings.datasets.network_data / self.network_path

            super().__init__(image_dir, *args, **kwargs)

    def image_name(self) -> str:
        return f"ref#{self.dataset_name}#{self.timestamp}"

    def image_name_parts(self) -> list[str]:
        return list(self.image_name_typed())

    def image_name_typed(self) -> dict[str, Any]:
        return {
            "ref_identifier": "ref",
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
        }

    def build_path(self, base_folder: Path) -> Path:
        return base_folder / self.network_path

    def rgb_path_reconstructed(self) -> Path:
        if self.image_dir is None:
            # image_dir may not be available if no network directory is set (e.g. on the cluster)
            assert (
                self.intermediates_dir is not None
            ), "Either the network drive or the intermediates directory must be set to get the RGB image path"
            return self.intermediates_dir / "preprocessing" / "rgb_reconstructed" / f"{self.image_name()}.blosc"
        else:
            return super().rgb_path_reconstructed()

    def read_rgb_reconstructed(
        self,
    ) -> np.ndarray:
        if self.image_dir is None:
            return decompress_file(self.rgb_path_reconstructed())
        else:
            return super().read_rgb_reconstructed()

    @staticmethod
    def _cache() -> dict:
        if DataPathReference._references_cache is None:
            DataPathReference._references_cache = {}

            unsorted_dir = settings.data_dirs.unsorted
            df = pd.read_feather(unsorted_dir / "image_references.feather")
            dsettings = DatasetSettings(unsorted_dir / "dataset_settings.json")

            # The unsorted dataset has its own intermediates but the data dir always references the original dataset
            intermediates_dir = settings.datasets.find_intermediates_dir(unsorted_dir)

            for image_name, network_path, dataset_name in zip(df["image_name"], df["network_path"], df["dataset_name"]):
                if settings.datasets.network_data is None:
                    data_dir = None
                else:
                    data_dir = settings.datasets.network_data / dataset_name / "data"

                DataPathReference._references_cache[image_name] = {
                    "network_path": Path(network_path),
                    "dataset_name": dataset_name,
                    "dsettings": dsettings,
                    "data_dir": data_dir,
                    "intermediates_dir": intermediates_dir,
                }

        return DataPathReference._references_cache

    @staticmethod
    def image_name_exists(image_name: str) -> bool:
        return image_name in DataPathReference._cache()

    @staticmethod
    def from_image_name(image_name: str, annotation_name: Union[str, list[str]]) -> Self:
        cache = DataPathReference._cache()
        assert image_name in cache, f"Could not find the image {image_name} in the reference table"

        match = cache[image_name]
        return DataPathReference(
            network_path=match["network_path"],
            dataset_name=match["dataset_name"],
            data_dir=match["data_dir"],
            intermediates_dir=match["intermediates_dir"],
            dataset_settings=match["dsettings"],
            annotation_name_default=annotation_name,
        )

    @staticmethod
    def iterate(
        data_dir: Path,
        filters: list[Callable[[Self], bool]],
        annotation_name: Union[str, list[str]],
    ) -> Iterator["DataPathReference"]:
        dataset_settings = DatasetSettings(data_dir / "dataset_settings.json")
        df_references = pd.read_feather(data_dir / "image_references.feather")
        intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)

        for i, row in df_references.iterrows():
            path = DataPathReference(
                network_path=row["network_path"],
                dataset_name=row["dataset_name"],
                data_dir=data_dir,
                intermediates_dir=intermediates_dir,
                dataset_settings=dataset_settings,
                annotation_name_default=annotation_name,
            )
            if all(f(path) for f in filters):
                yield path
