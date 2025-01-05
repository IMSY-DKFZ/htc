# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np

from htc.settings import settings
from htc.tivita.DataPath import DataPath


class TestDataPathSepsis:
    def test_iterate(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        paths = list(DataPath.iterate(settings.data_dirs.sepsis))
        assert len(paths) == len(set(paths))
        assert len(paths) > 0

        for path in paths:
            seg = path.read_segmentation(annotation_name="all")
            if seg is None:
                assert not (path() / "annotations").exists()
                assert len(sorted(path().glob("*Marker.txt"))) == 0
            else:
                for seg_mask in seg.values():
                    assert seg_mask.dtype == np.uint8
                    assert len(np.unique(seg_mask)) >= 2

    def test_read_segmentation(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        # 1 layer
        path = DataPath.from_image_name("healthy#bo997#None#recording_session_hand#2021_02_11_10_01_46")
        seg = path.read_segmentation()

        assert seg.dtype == np.uint8 and seg.shape == (480, 640)
        assert np.all(np.unique(seg) == [4, 255])

        # 2 layer
        path = DataPath.from_image_name("sepsis#JG955#t_7-post_admission_day_2_visit_1#leg_upper#2019_09_18_09_21_28")
        seg = path.read_segmentation()

        assert seg.dtype == np.uint8 and seg.shape == (2, 480, 640)
        assert np.all(np.unique(seg) == [6, 7, 9, 10, 255])
        assert np.all(np.unique(seg[0]) == [10, 255])
        assert np.all(np.unique(seg[1]) == [6, 7, 9, 255])

        # No annotations
        path = DataPath.from_image_name("sepsis#HM942#t_5-post_admission_day_1_visit_3#hand_inner#2019_09_17_20_59_37")
        seg = path.read_segmentation()
        assert seg is None
