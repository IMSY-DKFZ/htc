# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from pytest_console_scripts import ScriptRunner

import htc.data_processing.run_l1_normalization as run_l1_normalization
import htc.data_processing.run_parameter_images as run_parameter_images
import htc.data_processing.run_standardization as run_standardization
from htc.data_processing.run_standardization import RunningStats
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestRunningStats:
    def test_same(self) -> None:
        np.random.seed(1)
        X1 = np.random.rand(100, 200, 5)
        X2 = np.random.rand(100, 200, 5)
        X = np.concatenate([X1, X2])

        rs = RunningStats(channels=5)
        rs.add_data(X1)
        rs.add_data(X2)

        mean, std = rs.channel_params()
        assert pytest.approx(mean) == np.mean(X, axis=(0, 1)) and pytest.approx(std) == np.std(X, axis=(0, 1))
        mean, std = rs.pixel_params()
        assert pytest.approx(mean) == np.mean(X) and pytest.approx(std) == np.std(X)


def test_standardization(make_tmp_example_data: Callable, tmp_path: Path, script_runner: ScriptRunner) -> None:
    specs_json = """
    [
        {
            "fold_name": "fold_1",
            "train1": {
                "image_names": ["P044#2020_02_01_09_51_15"]
            },
            "train2": {
                "image_names": ["P044#2020_02_01_09_51_31"]
            },
            "val": {
                "image_names": ["P045#2020_02_05_16_51_41"]
            }
        },
        {
            "fold_name": "fold_2",
            "train1": {
                "image_names": ["P049#2020_02_11_19_09_49"]
            },
            "train2": {
                "image_names": ["P058#2020_05_13_17_33_48"]
            },
            "val": {
                "image_names": ["P058#2020_05_13_20_42_53"]
            }
        }
    ]
    """
    tmp_specs_path = tmp_path / "specs.json"
    with tmp_specs_path.open("w") as f:
        f.write(specs_json)

    specs = DataSpecification(tmp_specs_path)
    tmp_example_data = make_tmp_example_data(paths=specs.paths())

    # We need preprocessed files for the example images
    res = script_runner.run(run_l1_normalization.__file__, "--dataset-name", "2021_02_05_Tivita_multiorgan_semantic")
    assert res.success
    res = script_runner.run(run_parameter_images.__file__, "--dataset-name", "2021_02_05_Tivita_multiorgan_semantic")
    assert res.success

    # Precalculate the standardization
    res = script_runner.run(
        run_standardization.__file__,
        "--spec",
        str(tmp_specs_path),
        "--dataset-name",
        "2021_02_05_Tivita_multiorgan_semantic",
    )
    assert res.success

    params_path = tmp_example_data / "intermediates" / "data_stats" / f"{specs.name()}#standardization.pkl"
    results = pickle.load(params_path.open("rb"))
    assert list(results.keys()) == ["fold_1", "fold_2"]

    paths_fold_1 = [
        DataPath.from_image_name("P044#2020_02_01_09_51_15"),
        DataPath.from_image_name("P044#2020_02_01_09_51_31"),
    ]
    paths_fold_2 = [
        DataPath.from_image_name("P049#2020_02_11_19_09_49"),
        DataPath.from_image_name("P058#2020_05_13_17_33_48"),
    ]

    for paths, res in zip([paths_fold_1, paths_fold_2], [results["fold_1"], results["fold_2"]], strict=True):
        dataset_hsi = DatasetImage(
            paths,
            train=False,
            config=Config({"input/n_channels": 100, "input/preprocessing": "L1", "input/no_labels": True}),
        )
        dataset_tpi = DatasetImage(
            paths,
            train=False,
            config=Config({"input/n_channels": 4, "input/preprocessing": "parameter_images", "input/no_labels": True}),
        )
        dataset_rgb = DatasetImage(paths, train=False, config=Config({"input/n_channels": 3, "input/no_labels": True}))

        for dataset, name in [(dataset_hsi, "hsi"), (dataset_tpi, "tpi"), (dataset_rgb, "rgb")]:
            X = np.concatenate([sample["features"].numpy().astype(np.float64) for sample in dataset])
            assert pytest.approx(res[f"{name}_pixel_mean"]) == np.mean(X) and pytest.approx(
                res[f"{name}_pixel_std"]
            ) == np.std(X)
            assert pytest.approx(res[f"{name}_channel_mean"]) == np.mean(X, axis=(0, 1)) and pytest.approx(
                res[f"{name}_channel_std"]
            ) == np.std(X, axis=(0, 1))

    # Test the transformation
    config = Config({
        "input/no_labels": True,
        "input/normalization": "L1",
        "input/data_spec": str(tmp_specs_path),
        "input/n_channels": 100,
        "input/transforms_cpu": [{"class": "StandardizeHSI", "stype": "pixel"}],
    })
    sample_default = DatasetImage(specs.paths(), train=False, config=config)[0]
    sample_pixel = DatasetImage(specs.paths(), train=True, config=config, fold_name="fold_1")[0]

    assert abs(sample_pixel["features"].std() - 1) < abs(sample_default["features"].std() - 1)
    assert abs(sample_pixel["features"].mean()) < abs(sample_default["features"].mean())

    config["input/transforms_cpu"] = [{"class": "StandardizeHSI", "stype": "channel"}]
    sample_channel = DatasetImage(specs.paths(), train=True, config=config, fold_name="fold_1")[0]
    assert torch.all(
        torch.abs(sample_channel["features"].std(dim=(0, 1)) - 1)
        < torch.abs(sample_default["features"].std(dim=(0, 1)) - 1)
    )
    # Mean is already very small and does not get significantly closer to 0 with standardization
