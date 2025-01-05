# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from htc.dataset_preparation.run_dataset_sepsis_icu import DatasetGeneratorSepsisICU
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc_projects.camera.calibration.CalibrationSwap import CalibrationSwap, recalibrated_path


class TestDatasetGeneratorSepsisICU:
    output_path = settings.datasets.sepsis_ICU["path_dataset"]

    def test_recalibration_match(self, caplog: pytest.LogCaptureFixture) -> None:
        generator = DatasetGeneratorSepsisICU(output_path=self.output_path)
        t = CalibrationSwap()

        # example case - more comprehensive test cases are covered in test_RecalibrationMatcherSepsis.py
        img_path = DataPath.from_image_name("calibration_colorchecker#2023_02_07_10_26_50")
        white_path = DataPath.from_image_name("calibration_white#2023_02_07_10_25_26")
        cube1 = t.transform_image(
            img_path, image=torch.from_numpy(img_path.read_cube()), calibration_target=white_path
        ).numpy()
        cube2 = generator.recalibration_matcher.match(img_path).read_cube()
        assert np.allclose(cube1, cube2)

    def test_recalibration(self) -> None:
        img_path = DataPath.from_image_name("calibration_colorchecker#2023_02_06_18_30_48")
        white_path = DataPath.from_image_name("calibration_white#2023_02_06_18_28_34")

        # manual recalibration
        manual_recal = img_path.read_cube() / white_path.read_cube()
        manual_recal = manual_recal / np.linalg.norm(manual_recal, ord=1, axis=-1, keepdims=True)
        manual_recal = np.nan_to_num(manual_recal, copy=False)

        # automated recalibration
        auto_recal = recalibrated_path(img_path, white_path).read_cube(normalization=1)
        assert np.allclose(manual_recal, auto_recal)
