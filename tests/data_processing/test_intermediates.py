# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
from collections.abc import Callable

import pytest

from htc.settings import settings
from htc.tivita.DataPath import DataPath


class TestIntermediates:
    @pytest.mark.parametrize(
        "dataset_name",
        [
            "2020_11_24_Tivita_sepsis_study",
            "2021_02_05_Tivita_multiorgan_semantic",
            "2021_02_05_Tivita_multiorgan_masks",
            "2021_07_26_Tivita_multiorgan_human",
            "2023_04_22_Tivita_multiorgan_kidney",
        ],
    )
    def test_preprocessing(
        self, dataset_name: str, check_sepsis_data_accessible: Callable, check_human_data_accessible: Callable
    ) -> None:
        if dataset_name == "2020_11_24_Tivita_sepsis_study":
            check_sepsis_data_accessible()
        elif dataset_name == "2021_07_26_Tivita_multiorgan_human":
            check_human_data_accessible()

        if (
            dataset_name == "2021_02_05_Tivita_multiorgan_masks"
            or dataset_name == "2023_04_22_Tivita_multiorgan_kidney"
        ):
            paths = list(DataPath.iterate(settings.data_dirs[dataset_name]))
            paths += list(DataPath.iterate(settings.data_dirs[dataset_name] / "overlap"))
        elif dataset_name == "2021_02_05_Tivita_multiorgan_semantic":
            paths = list(DataPath.iterate(settings.data_dirs[dataset_name]))
            paths += list(DataPath.iterate(settings.data_dirs[dataset_name] / "context_experiments"))
        else:
            paths = list(DataPath.iterate(settings.data_dirs[dataset_name]))

        assert len(paths) > 0

        intermediates_dataset = settings.intermediates_dir_all.find_location(dataset_name)
        assert dataset_name in str(intermediates_dataset)

        # Check that every precomputed file actually belongs to the dataset
        path_names = {p.image_name() for p in paths if not p.image_name().endswith("#overlap")}
        for folder in (intermediates_dataset / "preprocessing").iterdir():
            folder_names = {f.stem for f in folder.iterdir()}
            assert folder_names.issubset(path_names)

    def test_uniqueness(self) -> None:
        all_paths = set()
        for name, entry in settings.datasets:
            # os.walk is much faster than rglob("*") here because we already know what is a file and what is a dir
            path_intermediates = str(entry["path_intermediates"])
            for root, dirs, files in os.walk(path_intermediates):
                for file in files:
                    # Uniqueness is not important for html files (e.g., views)
                    if file.endswith(".html"):
                        continue

                    # Using strings instead of path objects here is much faster
                    path = os.path.relpath(os.path.join(root, file), path_intermediates)  # noqa: PTH118
                    assert path not in all_paths, (
                        f"The file {file} of the dataset {name} is not unique across all intermediate folders."
                        f" Possible locations:\n{settings.intermediates_dir_all / path!r}"
                    )
                    all_paths.add(path)

        assert len(all_paths) > 0, "No intermediate files found."
