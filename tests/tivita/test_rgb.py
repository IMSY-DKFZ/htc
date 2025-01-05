# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np
import pytest
import torch
from pytest import MonkeyPatch

from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.rgb import align_rgb_sensor, hsi_to_rgb, hsi_to_rgb_tensor, read_tivita_rgb
from htc.utils.Config import Config


def test_read_tivita_rgb() -> None:
    path = DataPath.from_image_name("P068#2020_07_20_18_17_26")
    rgb = path.read_rgb_reconstructed()

    assert rgb.shape == (480, 640, 3)
    assert np.all(0 <= rgb) and np.all(rgb <= 255)
    assert np.all(rgb == read_tivita_rgb(path() / "2020_07_20_18_17_26_RGB-Image.png"))

    tivita_surgery_path = DataPath(
        settings.data_dirs.studies / "2022_07_30_Silvia_swollen_foot/swollen/2022_07_30_18_54_59"
    )
    rgb_tivita_surgery = tivita_surgery_path.read_rgb_reconstructed()

    assert rgb_tivita_surgery.shape == (480, 640, 3)
    assert np.all(0 <= rgb_tivita_surgery) and np.all(rgb_tivita_surgery <= 255)
    assert np.all(
        rgb_tivita_surgery == read_tivita_rgb(tivita_surgery_path() / "2022_07_30_18_54_59_HSI-RGB-Image.png")
    )

    rgb_sensor = tivita_surgery_path.read_rgb_sensor()
    assert rgb_sensor.shape == (480, 640, 3)
    assert np.all(0 <= rgb_sensor) and np.all(rgb_sensor <= 255)
    assert np.all(rgb_sensor == read_tivita_rgb(tivita_surgery_path() / "2022_07_30_18_54_59_RGB-Capture.png"))

    assert np.all(rgb_sensor == read_tivita_rgb(tivita_surgery_path() / "2022_07_30_18_54_59_RGB-Image.png"))
    with pytest.raises(AssertionError):
        path.read_rgb_sensor()


def test_hsi_to_rgb_tensor() -> None:
    path = DataPath.from_image_name("P068#2020_07_20_18_17_26")
    cube = path.read_cube()
    rgb_file = path.read_rgb_reconstructed()

    rgb_computed = hsi_to_rgb(cube)
    rgb_tensor = hsi_to_rgb_tensor(torch.from_numpy(cube)).numpy()

    diff_computed_file = np.abs(rgb_computed.astype(np.float32) - rgb_file.astype(np.float32))
    diff_computed_tensor = np.abs(rgb_computed.astype(np.float32) - rgb_tensor.astype(np.float32))
    assert diff_computed_tensor.mean() < 0.001 < diff_computed_file.mean()
    assert diff_computed_tensor.std() < 0.01 < diff_computed_file.std()

    rgb_tensor2 = hsi_to_rgb_tensor(torch.from_numpy(cube).unsqueeze(dim=0)).squeeze(dim=0).numpy()
    assert np.all(rgb_tensor == rgb_tensor2)


def test_align_rgb_sensor(monkeypatch: MonkeyPatch, check_sepsis_data_accessible: Callable) -> None:
    check_sepsis_data_accessible()

    n_calls = 0

    def call_count(func):
        def _call_count(*args, **kwargs):
            nonlocal n_calls
            n_calls += 1
            return func(*args, **kwargs)

        return _call_count

    monkeypatch.setattr("htc.tivita.rgb.align_rgb_sensor", call_count(align_rgb_sensor))

    path = DataPath.from_image_name("S001#2022_10_24_13_49_45")
    img_sensor = path.read_rgb_sensor()
    img_aligned = path.align_rgb_sensor(recompute=True)
    img_aligned2 = path.align_rgb_sensor(recompute=False)
    assert n_calls == 1

    assert img_sensor.shape == img_aligned.shape
    assert img_sensor.dtype == img_aligned.dtype
    assert np.all(img_aligned.mask[:, 0])
    assert np.all(img_aligned == img_aligned2)

    sample = DatasetImage([path], train=False, config=Config({"input/preprocessing": "rgb_sensor_aligned"}))[0]
    assert torch.allclose(sample["features"], torch.from_numpy(img_aligned.data) / 255)
