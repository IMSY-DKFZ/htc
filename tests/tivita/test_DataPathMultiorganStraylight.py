# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathMultiorganStraylight import DataPathMultiorganStraylight


class TestDataPathMultiorganStraylight:
    def test_basics(self) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.rat / "straylight_experiments"))
        paths_white = list(DataPath.iterate(settings.data_dirs.rat / "straylight_experiments" / "calibration_white"))
        paths_subjects = list(DataPath.iterate(settings.data_dirs.rat / "straylight_experiments" / "subjects"))

        assert paths == paths_white + paths_subjects
        assert all(isinstance(p, DataPathMultiorganStraylight) for p in paths)
        assert paths[0].image_name() == "calibration_white#2023_11_14_08_55_10#0202-00118#OR-right"
        assert paths[0].meta("calibration_subject") == "R027"
        assert paths[0].straylight == "OR-right"

        for p in paths_white:
            assert p.image_cat == "calibration_white"
            assert p.subject_name == "calibration_white"

        for p in paths_subjects:
            assert p.image_cat == "subjects"
            assert p.subject_name.startswith("R")
