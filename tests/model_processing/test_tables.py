# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from pytest_console_scripts import ScriptRunner

import htc.model_processing.run_tables as run_tables
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.Config import Config


@pytest.mark.serial
@pytest.mark.slow
class TestTables:
    expected_columns = (
        "ece",
        "dice_metric",
        "used_labels",
        "dice_metric_image",
        "confusion_matrix",
        "surface_distance_metric",
        "surface_distance_metric_image",
        "image_name",
    )

    @pytest.mark.parametrize("model_name", settings_seg.model_names)
    def test_validation_table(
        self,
        model_name: str,
        tmp_path: Path,
        script_runner: ScriptRunner,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Only one image per validation set to speed things up
        specs_data = [
            {"fold_name": "fold_P041,P060,P069", "val": {"image_names": ["P041#2019_12_14_12_00_16"]}},
            {"fold_name": "fold_P044,P050,P059", "val": {"image_names": ["P044#2020_02_01_09_51_15"]}},
            {"fold_name": "fold_P045,P061,P071", "val": {"image_names": ["P045#2020_02_05_10_54_19"]}},
            {"fold_name": "fold_P047,P049,P070", "val": {"image_names": ["P047#2020_02_07_17_28_15"]}},
            {"fold_name": "fold_P048,P057,P058", "val": {"image_names": ["P048#2020_02_08_10_34_35"]}},
        ]
        specs_path = tmp_path / "data.json"
        with specs_path.open("w") as f:
            json.dump(specs_data, f, indent=4)
            f.write("\n")

        spec = DataSpecification(specs_path)
        image_names = [p.image_name() for p in spec.paths("^val")]

        # We need to create a temporary run directory because the script loads the existing validation table and modifies it in-place
        run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"
        tmp_run_dir = tmp_path / "training" / model_name / f"{run_folder}_tmp"
        tmp_run_dir.mkdir(parents=True, exist_ok=True)

        run_dir = settings.training_dir / model_name / run_folder
        for f in sorted(run_dir.iterdir()):
            if f.name not in ["validation_table.pkl.xz", "config.json"]:
                os.symlink(f, tmp_run_dir / f.name)

        config = Config(run_dir / "config.json")
        config["input/data_spec"] = spec
        config_path = tmp_run_dir / "config.json"
        config.save_config(config_path)

        df_val_old = pd.read_pickle(run_dir / "validation_table.pkl.xz")
        df_val_old = df_val_old[df_val_old["image_name"].isin(image_names)]
        df_val_old = df_val_old.reset_index(drop=True)
        df_val_old.to_pickle(tmp_run_dir / "validation_table.pkl.xz")

        monkeypatch.setenv("PATH_HTC_RESULTS", str(tmp_path))
        res = script_runner.run(
            run_tables.__file__, "--model", model_name, "--run-folder", tmp_run_dir.name, "--output-dir", str(tmp_path)
        )
        assert res.success
        df_val_new = pd.read_pickle(tmp_path / model_name / tmp_run_dir.name / "validation_table.pkl.xz")

        assert set(df_val_new["image_name"].unique()) == set(image_names)
        assert all(c in df_val_new for c in self.expected_columns)
        assert any("surface_dice_metric" in c for c in df_val_new.columns), "No NSD data found"

        assert "image_name_annotations" not in df_val_new.columns
        assert "annotation_name" in df_val_new.columns
        assert (
            pd.isna(df_val_new.query("epoch_index == best_epoch_index and dataset_index == 0")["annotation_name"]).sum()
            == 0
        )

        assert not df_val_old.equals(df_val_new)
        assert np.all(
            df_val_old["image_name"].values == df_val_new["image_name"].values
        ) and df_val_old.index.identical(df_val_new.index)

        # Check that non best epochs remained untouched
        common_columns = list(set(df_val_old.columns).intersection(set(df_val_new.columns)))
        df_unchanged_old = df_val_old.query("not (epoch_index == best_epoch_index and dataset_index == 0)")[
            common_columns
        ]
        df_unchanged_new = df_val_new.query("not (epoch_index == best_epoch_index and dataset_index == 0)")[
            common_columns
        ]
        assert_frame_equal(df_unchanged_old, df_unchanged_new, check_exact=True)

        # We manually change the value of one cell and re-run the script again, the changed value should get overwritten
        first_index_value = df_val_new.query("epoch_index == best_epoch_index and dataset_index == 0").index.values[0]
        old_value = df_val_new.loc[first_index_value, "ece"]
        fake_value = 1337
        assert old_value != fake_value
        df_val_new.loc[first_index_value, "ece"] = fake_value
        df_val_new.to_pickle(tmp_path / "validation_table.pkl.xz")

        res = script_runner.run(
            run_tables.__file__,
            "--model",
            model_name,
            "--run-folder",
            tmp_run_dir.name,
            "--output-dir",
            str(tmp_path),
            "--config",
            str(config_path),
        )
        assert res.success

        # The manual change should now be reverted
        df_val_new2 = pd.read_pickle(tmp_path / model_name / tmp_run_dir.name / "validation_table.pkl.xz")
        assert (
            np.all(df_val_new["fold_name"].values == df_val_new2["fold_name"].values)
            and np.all(df_val_new["image_name"].values == df_val_new2["image_name"].values)
            and df_val_new.index.identical(df_val_new2.index)
        )
        assert df_val_new2.loc[first_index_value, "ece"] == old_value

        assert not (tmp_path / "predictions").exists()

    @pytest.mark.parametrize("model_name", settings_seg.model_names)
    def test_test_table(self, model_name: str, script_runner: ScriptRunner, tmp_path: Path) -> None:
        # We use the config switch to pass a custom data specification with only two images
        specs_data = [
            {"fold_name": "fold", "test": {"image_names": ["P045#2020_02_05_16_51_41", "P059#2020_05_14_12_50_10"]}}
        ]
        specs_path = tmp_path / "data.json"
        with specs_path.open("w") as f:
            json.dump(specs_data, f, indent=4)
            f.write("\n")

        spec = DataSpecification(specs_path)
        spec.activate_test_set()
        image_names = [p.image_name() for p in spec.paths("^test")]

        run_dir = settings.training_dir / model_name / "2022-02-03_22-58-44_generated_default_model_comparison"
        config = Config(run_dir / "config.json")
        config["input/data_spec"] = spec
        config_path = tmp_path / "config.json"
        config.save_config(config_path)

        res = script_runner.run(
            run_tables.__file__,
            "--model",
            model_name,
            "--run-folder",
            run_dir.name,
            "--test",
            "--output-dir",
            str(tmp_path),
            "--config",
            str(config_path),
        )
        assert res.success

        df_test = pd.read_pickle(tmp_path / model_name / run_dir.name / "test_table.pkl.xz")

        assert set(df_test["image_name"].unique()) == set(image_names)
        assert all(c in df_test for c in self.expected_columns)
        assert any("surface_dice_metric" in c for c in df_test.columns), "No NSD data found"
        assert all(not any(df_test[c].isna()) for c in self.expected_columns), "test table contains nan values"

        assert "image_name_annotations" not in df_test.columns
        assert "annotation_name" in df_test.columns
        assert pd.isna(df_test["annotation_name"]).sum() == 0

        assert not (tmp_path / "predictions").exists()
