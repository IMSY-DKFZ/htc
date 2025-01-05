# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_console_scripts import ScriptRunner

import htc.model_processing.run_image_figures as run_image_figures
from htc.tivita.DataPath import DataPath


@pytest.mark.serial
def test_image_figures(script_runner: ScriptRunner, tmp_path: Path, make_tmp_example_data: Callable) -> None:
    paths_selected = [
        DataPath.from_image_name("P043#2019_12_20_10_05_27"),
        DataPath.from_image_name("P043#2019_12_20_10_05_48"),
    ]
    tmp_example_dataset = make_tmp_example_data(n_images=2, paths=paths_selected, include_intermediates=True)
    input_dir = tmp_example_dataset / "data"

    run_folder = "2023-02-08_14-48-02_organ_transplantation_0.8"
    res = script_runner.run(
        run_image_figures.__file__,
        "--model",
        "image",
        "--run-folder",
        run_folder,
        "--test",
        "--input-dir",
        input_dir,
        "--output-dir",
        tmp_path,
        "--num-consumers",
        "1",
    )
    assert res.success

    predictions_dir = tmp_path / "image" / run_folder / "prediction_figures"
    assert predictions_dir.is_dir()

    # We only check that for every path in the specs the html exists
    paths_input = list(DataPath.iterate(input_dir))
    paths_output = sorted(predictions_dir.iterdir())
    assert len(paths_input) == len(paths_output) == len(paths_selected)

    for p in paths_output:
        assert p.stat().st_size > 0

    for path in paths_input:
        assert sum(p.name.startswith(path.image_name()) for p in paths_output) == 1
