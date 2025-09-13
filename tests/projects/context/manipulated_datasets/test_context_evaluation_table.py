# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

import pandas as pd
import pytest
from pytest_console_scripts import ScriptRunner

import htc_projects.context.manipulated_datasets.run_context_evaluation_table as run_context_evaluation_table
from htc.tivita.DataPath import DataPath


@pytest.mark.serial
def test_example_image(script_runner: ScriptRunner, tmp_path: Path) -> None:
    data_dir = tmp_path / "input" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # We need the data and the segmentations for this path for the test
    path = DataPath.from_image_name("P072#2020_08_08_18_05_23")
    image_dir = data_dir / "subjects" / path.subject_name
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / path.timestamp).symlink_to(path())
    shutil.copy2(path.dataset_settings.settings_path, data_dir / "dataset_settings.json")

    seg_dir = tmp_path / "input/intermediates/segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path.segmentation_path(), seg_dir / path.segmentation_path().name)

    res = script_runner.run(
        run_context_evaluation_table.__file__,
        "--model",
        "image",
        "--run-folder",
        "2022-02-03_22-58-44_generated_default_model_comparison",
        "--input-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--test",
        "--transformation-name",
        "isolation_cloth",
    )
    assert res.success

    df = pd.read_pickle(
        output_dir
        / "image"
        / "2022-02-03_22-58-44_generated_default_model_comparison"
        / "test_table_isolation_cloth.pkl.xz"
    )
    assert len(df) == 1
    assert df.iloc[0]["image_name"] == path.image_name()
    assert df.iloc[0]["used_labels"][2] == 2
    assert pytest.approx(df.iloc[0]["dice_metric"][2], abs=0.01) == 0.7599487
