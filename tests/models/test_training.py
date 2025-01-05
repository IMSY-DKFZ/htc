# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from pytest_console_scripts import ScriptRunner

import htc.evaluation.run_table_generation as run_table_generation
import htc.models.run_training as run_training
import htc_projects.atlas.model_processing.run_test_table_generation as run_test_table_generation
from htc.evaluation.analyze_tfevents import read_tfevent_losses
from htc.models.data.DataSpecification import DataSpecification
from htc.settings_seg import settings_seg
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


@pytest.mark.serial
class TestTraining:
    run_folder = "small_training"
    epochs = 2
    models = ("median_pixel", "image", "patch", "pixel", "superpixel_classification")

    @pytest.fixture()
    def config(self, model_name: str) -> Iterator[Config]:
        # Reduce the training time of the default config
        config_name = "default.json"
        config = Config.from_model_name(config_name, model_name)
        config["trainer_kwargs/max_epochs"] = self.epochs
        config["dataloader_kwargs/batch_size"] = max(config["dataloader_kwargs/batch_size"] // 2, 1)
        config["input/epoch_size"] = 2 * config["dataloader_kwargs/batch_size"]

        if "label_mapping" not in config:
            config["label_mapping"] = settings_seg.label_mapping

        # Create a new temporary config object
        config["config_name"] = "default_test"
        yield config

    def _simplify_specs(self, config: Config, training_dir: Path) -> None:
        specs = DataSpecification.from_config(config)
        specs.activate_test_set()

        new_folds = []
        for fold_name in specs.fold_names()[:2]:  # Only the first two folds
            datasets = specs.folds[fold_name]
            current_fold = {"fold_name": fold_name}
            for name, paths in datasets.items():
                if not name.startswith("train"):
                    # We reduce val or test paths to make this test faster
                    current_fold[name] = {"image_names": [p.image_name() for p in paths[:2]]}
                else:
                    current_fold[name] = {"image_names": [p.image_name() for p in paths]}

            new_folds.append(current_fold)

        specs_file = training_dir / "data.json"
        with specs_file.open("w") as f:
            json.dump(new_folds, f, indent=4)

        config["input/data_spec"] = str(specs_file)

    @pytest.mark.parametrize("model_name, config", [[m] * 2 for m in models], indirect=["config"])
    @pytest.mark.filterwarnings(r"ignore:Checkpoint directory.*exists and is not empty\.:UserWarning")
    @pytest.mark.filterwarnings(
        r"ignore:Your `IterableDataset` has `__len__` defined.*:UserWarning"
    )  # This is not a problem since the length is never used (if so, an error would occur)
    @pytest.mark.filterwarnings(
        r"ignore:This DataLoader will create \d+ worker processes in total\.:UserWarning"
    )  # This is intentional and no problem for the pixel model
    def test_training(
        self, tmp_path: Path, model_name: str, config: Config, script_runner: ScriptRunner, monkeypatch: MonkeyPatch
    ) -> None:
        # Setup the training directory
        monkeypatch.setenv("PATH_HTC_RESULTS", str(tmp_path))

        # In case other result directories are defined, redefine them as well
        for env_name in os.environ.keys():
            if env_name.startswith("PATH_HTC_RESULTS_"):
                monkeypatch.setenv(env_name, str(tmp_path))

        training_dir = tmp_path / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        self._simplify_specs(config, training_dir)
        specs = DataSpecification.from_config(config)
        assert len(specs) == 2

        # For simplicity, we just store the config in the training directory
        config.save_config(training_dir / "default_test.json")

        # If a model is outside the models dir, we need to specify the lightning module absolutely
        module_args = ["--test"] if model_name == "median_pixel" else []

        # We train two folds to check whether the fold combination also works
        train_stats_lengths = []
        for fold_name in specs.fold_names():
            res = script_runner.run(
                run_training.__file__,
                "--model",
                model_name,
                "--config",
                str(config.path_config),
                "--fold-name",
                fold_name,
                "--run-folder",
                self.run_folder,
                *module_args,
            )
            assert res.success

            run_dir = training_dir / model_name / self.run_folder
            assert run_dir.exists()
            fold_dir = run_dir / fold_name
            assert fold_dir.exists()

            config_run = Config(fold_dir / "config.json")
            assert "unused_keys" not in config_run.keys(), (
                f"The keys {config_run['unused_keys']} are defined in the {config['config_name']} config but have never"
                " been used"
            )
            filter_keys = lambda keys: [k for k in keys if not k.startswith("label_mapping")]
            assert set(filter_keys(config.keys())).issubset(filter_keys(config_run.keys()))

            # Check that config values are the same
            assert LabelMapping.from_config(config) == LabelMapping.from_config(config_run)
            assert DataSpecification.from_config(config) == DataSpecification.from_config(config_run)
            for k in config.keys():
                if k.startswith(("label_mapping", "config_name", "input/data_spec")):
                    continue

                assert config[k] == config_run[k]

            assert (fold_dir / "data.json").exists()
            assert len(list(fold_dir.glob("*ckpt"))) == 1

            log_path = fold_dir / "log.txt"
            assert log_path.exists()
            with log_path.open() as f:
                log_text = f.read()
            assert len(log_text) > 0
            assert all(level not in log_text for level in ["ERROR", "CRITICAL"])

            if "pixel" not in model_name:
                assert (fold_dir / "trainings_stats.npz").exists()
                stats = np.load(fold_dir / "trainings_stats.npz", allow_pickle=True)["data"]
                train_stats_lengths.append(len(stats))
                for epoch_stats in stats:
                    assert epoch_stats["img_indices"].shape == (config["input/epoch_size"],)

        # For the table generation, we don't want network results anymore
        monkeypatch.setenv("HTC_ADD_NETWORK_ALTERNATIVES", "false")
        if model_name == "median_pixel":
            res = script_runner.run(run_test_table_generation.__file__)
            assert res.success

        # Generate the validation results for the run (merges results of all folds)
        notebook = (
            "atlas/ExperimentAnalysis.ipynb" if model_name == "median_pixel" else "evaluation/ExperimentAnalysis.ipynb"
        )
        res = script_runner.run(run_table_generation.__file__, "--notebook", notebook)
        assert res.success

        for f in [
            "config.json",
            "data.json",
            "ExperimentAnalysis.html",
            "validation_table.pkl.xz",
        ]:
            assert (run_dir / f).exists()

        df_train = read_tfevent_losses(run_dir)
        df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz")

        assert df_train is not None
        assert sorted(df_val.query("dataset_index == 0")["fold_name"].unique().tolist()) == specs.fold_names()
        assert "image_name" in df_val.columns
        assert "image_name_annotations" not in df_val.columns
        assert pd.isna(df_val["image_name"]).sum() == 0

        if model_name != "median_pixel":
            assert "annotation_name" in df_val.columns
            assert pd.isna(df_val["annotation_name"]).sum() == 0

        if config["swa_kwargs"]:
            # With SWA enabled, we have one additional epoch
            assert df_train["epoch_index"].max() == self.epochs
            assert df_val["epoch_index"].max() == self.epochs
            assert all(l == self.epochs + 1 for l in train_stats_lengths)
        else:
            assert df_train["epoch_index"].max() == self.epochs - 1
            assert df_val["epoch_index"].max() == self.epochs - 1
            assert all(l == self.epochs for l in train_stats_lengths)

    @pytest.mark.parametrize("model_name, config", [["image", "image"]], indirect=["config"])
    def test_last_epoch_validated(
        self, tmp_path: Path, model_name: str, config: Config, script_runner: ScriptRunner, monkeypatch: MonkeyPatch
    ) -> None:
        config["trainer_kwargs/check_val_every_n_epoch"] = 3
        config["trainer_kwargs/max_epochs"] = 4

        # Setup the training directory
        monkeypatch.setenv("PATH_HTC_RESULTS", str(tmp_path))
        training_dir = tmp_path / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        self._simplify_specs(config, training_dir)
        specs = DataSpecification.from_config(config)
        assert len(specs) == 2

        # For simplicity, we just store the config in the training directory
        config.save_config(training_dir / "default_test.json")

        assert "swa_kwargs" in config
        fold_name = specs.fold_names()[0]
        res = script_runner.run(
            run_training.__file__,
            "--model",
            model_name,
            "--config",
            str(config.path_config),
            "--fold-name",
            fold_name,
            "--run-folder",
            self.run_folder,
        )
        assert res.success

        # Check that the SWA epoch is validated
        run_dir = training_dir / model_name / self.run_folder / fold_name
        val_data = pd.read_pickle(run_dir / "validation_results.pkl.xz")
        assert val_data["epoch_index"].max() == config["trainer_kwargs/max_epochs"], (
            "Last epoch should always be validated"
        )
