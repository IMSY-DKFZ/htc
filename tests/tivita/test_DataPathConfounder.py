# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathConfounder import DataPathConfounder


class TestDataPathConfounder:
    def test_subdirs(self):
        # test reading paths from different subdirectories
        path = DataPathConfounder(
            settings.data_dirs.confounder
            / "straylight_subjects"
            / "subjects"
            / "no_straylight"
            / "CS1"
            / "2024_01_06_15_00_11"
        )
        assert path.image_name() == "CS1#2024_01_06_15_00_11#0615-00036#no_straylight"

        path = DataPathConfounder(
            settings.data_dirs.confounder
            / "straylight_subjects"
            / "calibration_white"
            / "ceiling"
            / "2024_01_08_13_44_01"
        )
        assert path.image_name() == "calibration_white#2024_01_08_13_44_01#0202-00118#ceiling"

        path = DataPathConfounder(
            settings.data_dirs.confounder / "sensor_temperature_subjects" / "subjects" / "CS4" / "2024_01_08_14_23_15"
        )
        assert path.image_name() == "CS4#2024_01_08_14_23_15#0202-00118#no_straylight"

        path = DataPathConfounder(
            settings.data_dirs.confounder / "sensor_temperature_subjects" / "calibration_white" / "2024_01_09_12_54_31"
        )
        assert path.image_name() == "calibration_white#2024_01_09_12_54_31#0615-00036#no_straylight"

        path = DataPathConfounder(
            settings.data_dirs.confounder / "device_shift_subjects" / "0615-00023" / "CS5" / "2024_01_10_19_47_41"
        )
        assert path.image_name() == "CS5#2024_01_10_19_47_41#0615-00023#no_straylight"

        path = DataPathConfounder(
            settings.data_dirs.confounder
            / "combined_colorcheckers"
            / "colorchecker_IMSY3"
            / "OR-right"
            / "0615-00036"
            / "repetition2"
            / "2024_01_07_15_18_01"
        )
        assert path.image_name() == "colorchecker_IMSY3#2024_01_07_15_18_01#0615-00036#OR-right"

        path = DataPathConfounder(
            settings.data_dirs.confounder
            / "combined_colorcheckers"
            / "calibration_white"
            / "OR-situs"
            / "0202-00118"
            / "2024_01_04_15_17_42"
        )
        assert path.image_name() == "calibration_white#2024_01_04_15_17_42#0202-00118#OR-situs"

    def test_iterate(self):
        paths_all = list(DataPath.iterate(settings.data_dirs.confounder))
        assert all(isinstance(p, DataPathConfounder) for p in paths_all)

        paths_straylight = list(DataPath.iterate(settings.data_dirs.confounder / "straylight_subjects"))
        assert len(paths_straylight) > 1
        assert all(isinstance(p, DataPathConfounder) for p in paths_straylight)

        paths_sensor_temperature = list(DataPath.iterate(settings.data_dirs.confounder / "sensor_temperature_subjects"))
        assert len(paths_sensor_temperature) > 1
        assert all(isinstance(p, DataPathConfounder) for p in paths_sensor_temperature)

        paths_device_shift = list(DataPath.iterate(settings.data_dirs.confounder / "device_shift_subjects"))
        assert len(paths_device_shift) > 1
        assert all(isinstance(p, DataPathConfounder) for p in paths_device_shift)

        paths_combined = list(DataPath.iterate(settings.data_dirs.confounder / "combined_colorcheckers"))
        assert len(paths_combined) > 1
        assert all(isinstance(p, DataPathConfounder) for p in paths_combined)

        assert len(paths_all) == len(paths_straylight) + len(paths_sensor_temperature) + len(paths_device_shift) + len(
            paths_combined
        )

    def test_sensor_temperature(self) -> None:
        path = DataPath.from_image_name("calibration_white#2024_01_09_12_54_31#0615-00036#no_straylight")

        sensor_temp_mean = (
            path.meta("Temperaturen_HSI-Sensor Temp. vor Scan") + path.meta("Temperaturen_HSI-Sensor Temp. nach Scan")
        ) / 2
        assert path.meta("sensor_temperature") == pytest.approx(sensor_temp_mean)

        light_temp_mean = (
            path.meta("Temperaturen_LED Temp. vor Scan") + path.meta("Temperaturen_LED Temp. nach Scan")
        ) / 2
        assert path.meta("light_temperature") == pytest.approx(light_temp_mean)
