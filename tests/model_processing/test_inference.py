# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from pytest_console_scripts import ScriptRunner

import htc.model_processing.run_inference as run_inference
from htc.settings import settings
from htc.tivita.DataPath import DataPath


@pytest.mark.serial
def test_inference(script_runner: ScriptRunner, tmp_path: Path) -> None:
    input_dir = settings.data_dirs.semantic / "subjects" / "P041"
    run_folder = "2023-02-08_14-48-02_organ_transplantation_0.8"
    res = script_runner.run(
        run_inference.__file__,
        "--model",
        "image",
        "--run-folder",
        run_folder,
        "--input-dir",
        input_dir,
        "--output-dir",
        tmp_path,
        "--num-consumers",
        "1",
    )
    assert res.success

    predictions_dir = tmp_path / "image" / run_folder / "predictions"
    assert predictions_dir.is_dir()

    file_config = predictions_dir / "config.json"
    assert file_config.exists() and file_config.stat().st_size > 0

    paths = list(DataPath.iterate(input_dir))
    assert len(list(predictions_dir.iterdir())) == 2 * len(paths) + 1
    for p in paths:
        file_predictions = predictions_dir / f"{p.image_name()}.blosc"
        file_html = predictions_dir / f"{p.image_name()}.html"
        assert file_predictions.exists() and file_predictions.stat().st_size > 0
        assert file_html.exists() and file_html.stat().st_size > 0
