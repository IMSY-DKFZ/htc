# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings


class TestDataPathTivita:
    def test_tivita(self, tmp_path: Path) -> None:
        dirs = [
            tmp_path / "2021_11_23_17_23_51",
            tmp_path / "top/2021_11_24_17_23_51",
            tmp_path / "top/mid/2021_11_25_17_23_51",
            tmp_path / "top/mid/2021_11_26_17_23_51",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{d.name}_SpecCube.dat").write_text("blub")

        shutil.copy2(settings.data_dirs.studies / "dataset_settings.json", tmp_path / "dataset_settings.json")

        paths = list(DataPath.iterate(tmp_path))
        assert len(paths) == len(dirs)
        assert all(isinstance(p, DataPathTivita) for p in paths)
        assert paths[0].timestamp == "2021_11_23_17_23_51"
        assert paths[1].timestamp == "2021_11_24_17_23_51"
        assert paths[2].timestamp == "2021_11_25_17_23_51"
        assert paths[3].timestamp == "2021_11_26_17_23_51"
        assert paths[0]() == dirs[0]
        assert paths[1]() == dirs[1]
        assert paths[2]() == dirs[2]
        assert paths[3]() == dirs[3]
        assert paths[0].image_name() == "unknown#2021_11_23_17_23_51"
        assert paths[1].image_name() == "unknown#2021_11_24_17_23_51"
        assert paths[2].image_name() == "unknown#2021_11_25_17_23_51"
        assert paths[3].image_name() == "unknown#2021_11_26_17_23_51"
        assert paths[0].attributes == []
        assert paths[1].attributes == ["top"]
        assert paths[2].attributes == ["top", "mid"]
        assert paths[3].attributes == ["top", "mid"]
        base_path = Path("/test")
        assert paths[0].build_path(base_path) == Path("/test/2021_11_23_17_23_51")
        assert paths[1].build_path(base_path) == Path("/test/top/2021_11_24_17_23_51")
        assert paths[2].build_path(base_path) == Path("/test/top/mid/2021_11_25_17_23_51")
        assert paths[3].build_path(base_path) == Path("/test/top/mid/2021_11_26_17_23_51")

        mid_attributes = lambda p: "mid" in p.attributes
        paths = list(DataPath.iterate(tmp_path, filters=[mid_attributes]))
        assert len(paths) == 2
        assert paths[0].timestamp == "2021_11_25_17_23_51"
        assert paths[1].timestamp == "2021_11_26_17_23_51"

    def test_dataset_settings_finding(self):
        test_set = {
            "2023_08_01_skin_Surgery2_timecourse/2023_08_01_17_15_37": DatasetSettings(
                settings.data_dirs.studies / "2023_08_01_skin_Surgery2_timecourse/dataset_settings.json"
            ),
            "2023_07_17_whitetile_Surgery2_timecourse/2023_07_17_17_23_36": DatasetSettings(
                settings.data_dirs.studies / "dataset_settings.json"
            ),
            "2023_02_09_colorchecker_MIC1_TivitaMini/cc_cyan/2023_02_08_10_53_22": DatasetSettings(
                settings.data_dirs.studies / "2023_02_09_colorchecker_MIC1_TivitaMini/dataset_settings.json"
            ),
        }

        # test init
        for path, true_dataset_settings in test_set.items():
            path = settings.data_dirs.studies / path
            dataset_settings = DataPath(path).dataset_settings
            assert dataset_settings == true_dataset_settings

        # test iterate
        for path, true_dataset_settings in test_set.items():
            path = settings.data_dirs.studies / path
            data_dir = path.parent
            paths = list(DataPath.iterate(data_dir))
            dataset_settings = paths[0].dataset_settings
            assert dataset_settings == true_dataset_settings
