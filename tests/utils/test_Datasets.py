# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from pathlib import Path

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from htc.settings import settings
from htc.utils.Datasets import Datasets


class TestDatasets:
    def test_dirs_access(self) -> None:
        assert "2021_02_05_Tivita_multiorgan_masks/data" in str(settings.data_dirs["PATH_Tivita_multiorgan_masks"])
        assert "2021_02_05_Tivita_multiorgan_masks/data" in str(
            settings.data_dirs["2021_02_05_Tivita_multiorgan_masks"]
        )
        assert "2021_02_05_Tivita_multiorgan_masks/data" in str(settings.data_dirs["masks"])

        assert settings.data_dirs["2021"] is None
        assert "2021_02_05_Tivita_multiorgan_masks/data" in str(settings.data_dirs.masks)

    def test_path_to_env(self) -> None:
        assert settings.datasets.path_to_env(settings.data_dirs.masks) == "PATH_Tivita_multiorgan_masks"
        assert settings.datasets.path_to_env("some/nonexistent/path") is None

    def test_local_only(self, caplog: LogCaptureFixture) -> None:
        tmp_datasets = copy.deepcopy(settings.datasets)
        tmp_datasets.add_dir("env_name_for_test_location", "network_folder_name_for_test_location")

        assert tmp_datasets.get("env_name_for_test_location", local_only=True) is None
        assert tmp_datasets.get("network_folder_name_for_test_location", local_only=True) is None
        assert len(caplog.records) == 0

    def test_contains(self, caplog: LogCaptureFixture) -> None:
        assert "2021_02_05_Tivita_multiorgan_masks" in settings.data_dirs
        assert "nonexistent_name" not in settings.data_dirs
        assert len(caplog.records) == 0

    def test_find_intermediates_dir(self) -> None:
        intermediates_dir = settings.intermediates_dirs.masks
        assert settings.datasets.find_intermediates_dir(settings.data_dirs.masks) == intermediates_dir
        assert settings.datasets.find_intermediates_dir(settings.data_dirs.masks.parent) == intermediates_dir
        assert settings.datasets.find_intermediates_dir("some/path/with/data") == Path("some/path/with/intermediates")
        assert settings.datasets.find_intermediates_dir("some/path/with/data/subfolder/again") == Path(
            "some/path/with/intermediates"
        )

    def test_shortcut_match(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("HTC_TEST_PATH_1", str(tmp_path / "1"))
        monkeypatch.setenv("HTC_TEST_PATH_2", str(tmp_path / "2"))

        data_dirs = Datasets()
        data_dirs.add_dir("HTC_TEST_PATH_1", "1", shortcut="a")
        data_dirs.add_dir("HTC_TEST_PATH_2", "2", shortcut="ab")
        assert data_dirs.a["path_data"] == tmp_path / "1" / "data"
        assert data_dirs.ab["path_data"] == tmp_path / "2" / "data"

    @pytest.mark.parametrize("dataset_name", ["PATH_Tivita_test_dataset", "PATH_TIVITA_TEST_DATASET"])
    def test_additional_dataset(self, dataset_name: str, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(settings, "_datasets", None)
        monkeypatch.setenv(dataset_name, f"{tmp_path}:shortcut=test")

        tmp_data = tmp_path / "data"
        tmp_data.mkdir(parents=True, exist_ok=True)

        assert settings.data_dirs["PATH_Tivita_test_dataset"] == tmp_data
        assert settings.data_dirs[tmp_path.name] == tmp_data
        assert settings.data_dirs.test == tmp_data
        assert settings.data_dirs["test_dataset"] == tmp_data
        assert settings.data_dirs["TEST_DATASET"] == tmp_data, "Casing should be irrelevant"

        # Shortcut is optional
        monkeypatch.setattr(settings, "_datasets", None)
        monkeypatch.setenv(dataset_name, f"{tmp_path}")

        assert settings.data_dirs["PATH_Tivita_test_dataset"] == tmp_data
        assert settings.data_dirs[tmp_path.name] == tmp_data
        assert settings.data_dirs.test is None

    def test_parse_path_options(self) -> None:
        path, options = Datasets.parse_path_options("~/htc/Tivita_my_dataset:shortcut=my_shortcut")
        assert path == Path("~/htc/Tivita_my_dataset")
        assert options == {"shortcut": "my_shortcut"}

        path, options = Datasets.parse_path_options("~/htc/Tivita_my_dataset")
        assert path == Path("~/htc/Tivita_my_dataset")
        assert options == {}

        path, options = Datasets.parse_path_options("C:/htc/Tivita_my_dataset:shortcut=my_shortcut")
        assert path == Path("C:/htc/Tivita_my_dataset")
        assert options == {"shortcut": "my_shortcut"}

        path, options = Datasets.parse_path_options("C:/htc/Tivita_my_dataset")
        assert path == Path("C:/htc/Tivita_my_dataset")
        assert options == {}
