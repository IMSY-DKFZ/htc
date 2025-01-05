# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np

from htc.models.image.DatasetImage import DatasetImage
from htc.tivita.DataPath import DataPath
from htc.tivita.hsi import read_tivita_hsi
from htc.tivita.rgb import hsi_to_rgb
from htc.utils.Config import Config


def test_read_tivita_hsi() -> None:
    path = DataPath.from_image_name("P068#2020_07_20_18_17_26")
    cube = path.read_cube()
    rgb = path.read_rgb_reconstructed()

    assert cube.shape == (480, 640, 100)
    assert rgb.shape == (480, 640, 3)

    assert np.all(cube == read_tivita_hsi(path() / "2020_07_20_18_17_26_SpecCube.dat"))
    assert np.all(cube == read_tivita_hsi(path.cube_path()))

    # The shape information is stored in the first three integer values of the file
    shape = np.fromfile(path.cube_path(), dtype=">i", count=3)
    assert np.all(shape == (640, 480, 100))

    config = Config({"input/no_labels": True})
    sample = DatasetImage([path], train=False, config=config)[0]
    assert np.all(cube == sample["features"].numpy())

    abs_diff = np.abs(hsi_to_rgb(cube).astype(np.float32) - rgb.astype(np.float32))
    assert abs_diff.mean() < 2 and abs_diff.std() < 2

    assert not np.allclose(np.sum(np.abs(cube), axis=-1), 1)
    cube_l1 = read_tivita_hsi(path.cube_path(), normalization=1)
    assert np.allclose(np.sum(np.abs(cube_l1), axis=-1), 1)
    assert np.all(cube_l1 == path.read_cube(normalization=1))
