# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathSepsisICU import DataPathSepsisICU


class TestDataPathSepsisICU:
    def test_image_cat(self, check_sepsis_ICU_data_accessible: Callable) -> None:
        check_sepsis_ICU_data_accessible()

        dpath = DataPathSepsisICU(settings.data_dirs.sepsis_ICU / "calibrations/white/2022_10_24_19_56_53")
        assert dpath.image_cat == "calibrations"

    def test_iterate(self, check_sepsis_ICU_data_accessible: Callable) -> None:
        check_sepsis_ICU_data_accessible()

        paths = list(DataPath.iterate(settings.data_dirs.sepsis_ICU))
        timestamps = [p.timestamp for p in paths]
        assert len(timestamps) == len(set(timestamps))

        paths_colorchecker = list(
            DataPath.iterate(settings.data_dirs.sepsis_ICU / "calibrations" / "colorchecker_classic_video_passport")
        )
        assert all(isinstance(path, DataPathSepsisICU) for path in paths_colorchecker)
        paths_white = list(DataPath.iterate(settings.data_dirs.sepsis_ICU / "calibrations" / "white"))
        assert all(isinstance(path, DataPathSepsisICU) for path in paths_white)
        paths_calibration_subjects = list(DataPath.iterate(settings.data_dirs.sepsis_ICU / "calibrations" / "probands"))
        assert all(isinstance(path, DataPathSepsisICU) for path in paths_calibration_subjects)

        paths_calibrations = list(DataPath.iterate(settings.data_dirs.sepsis_ICU / "calibrations"))
        assert len(paths_calibrations) == len(paths_colorchecker) + len(paths_white) + len(paths_calibration_subjects)
        assert all(isinstance(path, DataPathSepsisICU) for path in paths_calibrations)

        paths_subjects = list(DataPath.iterate(settings.data_dirs.sepsis_ICU / "subjects"))
        assert len(paths) == len(paths_subjects) + len(paths_calibrations)
        assert all(isinstance(path, DataPathSepsisICU) for path in paths_subjects)
