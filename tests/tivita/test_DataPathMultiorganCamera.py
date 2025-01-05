# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathMultiorganCamera import DataPathMultiorganCamera


class TestDataPathMultiorganCamera:
    def test_basics(self) -> None:
        path = next(DataPath.iterate(settings.data_dirs.rat))
        path_halogen = next(DataPath.iterate(settings.data_dirs.rat / "0202-00118"))
        path_led = next(DataPath.iterate(settings.data_dirs.rat / "0615-00036"))

        assert path == path_halogen
        assert isinstance(path, DataPathMultiorganCamera)
        assert path.image_name() == "R002#2023_09_19_10_14_28#0202-00118"
        assert path.subject_name == "R002"
        assert path.timestamp == "2023_09_19_10_14_28"
        assert path_led.image_name() == "R002#2023_09_19_10_20_11#0615-00036"
        assert path_led.subject_name == "R002"
        assert path_led.timestamp == "2023_09_19_10_20_11"

        assert path.Camera_CamID == path.meta("Camera_CamID")
        assert path_led.Camera_CamID == path_led.meta("Camera_CamID")
