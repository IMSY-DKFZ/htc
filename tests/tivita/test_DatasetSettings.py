# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

from htc.settings import settings
from htc.tivita.DatasetSettings import DatasetSettings


class TestDatasetSettings:
    def test_construction(self) -> None:
        dataset_settings = DatasetSettings(settings.data_dirs.semantic / "dataset_settings.json")
        assert dataset_settings["shape"] == (480, 640, 100)
        assert dataset_settings.pixels_image() == 480 * 640

        dataset_settings2 = DatasetSettings(settings.data_dirs.semantic)
        assert dataset_settings == dataset_settings2

    def test_delayed_loading(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "dataset_settings.json"
        dataset_settings = DatasetSettings(settings_path)
        assert dataset_settings._data is None

        shutil.copy2(settings.data_dirs.semantic / "dataset_settings.json", settings_path)
        assert dataset_settings["shape"] == (480, 640, 100)

        assert DatasetSettings(tmp_path / "nonexistent") == DatasetSettings(tmp_path / "nonexistent")

    def test_subdataset(self) -> None:
        dataset_settings1 = DatasetSettings(settings.data_dirs["2022_10_24_Tivita_sepsis_ICU"])
        assert dataset_settings1["dataset_name"] == "2022_10_24_Tivita_sepsis_ICU"
        assert not (settings.data_dirs["2022_10_24_Tivita_sepsis_ICU#subjects"] / "dataset_settings.json").exists()
        dataset_settings2 = DatasetSettings(settings.data_dirs["2022_10_24_Tivita_sepsis_ICU#subjects"])
        assert dataset_settings1 == dataset_settings2
