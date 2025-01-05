# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch

from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


class TestMetricAggregation:
    @pytest.fixture(scope="function")
    def df_example(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                [0, 10, 100, [0, 1], [0.5, 0.3], [1, 1]],
                [0, 10, 101, [0, 1], [0.1, 0.2], [1, 1]],
                [0, 10, 102, [0], [0.3], [1]],
                [0, 10, 103, [1], [0.2], [0]],
                [0, 20, 201, [0], [0.3], [1]],
                [1, 30, 301, [1], [0.5], [0]],
                [1, 30, 302, [0], [0.8], [1]],
            ],
            columns=["camera_index", "subject_name", "image_name", "used_labels", "dice_metric", "other_metric"],
        )

        return df

    def test_checkpoint_metric(self, df_example: pd.DataFrame) -> None:
        agg = MetricAggregation(
            df_example,
            config=Config({
                "input/target_domain": ["camera_index"],
                "validation/checkpoint_metric_mode": "class_level",
            }),
        )
        # We average over images, pigs, cams and labels to retrieve a final label score
        pig_10_liver = (0.5 + 0.1 + 0.3) / 3
        cam_0_liver = (pig_10_liver + 0.3) / 2
        liver = (cam_0_liver + 0.8) / 2

        assert agg.checkpoint_metric() == (liver + (0.5 + ((0.3 + 0.2 + 0.2) / 3)) / 2) / 2

        agg = MetricAggregation(
            df_example,
            config=Config({
                "input/target_domain": ["camera_index"],
                "validation/checkpoint_metric_mode": "image_level",
            }),
        )
        pig_10 = (0.8 / 2 + 0.3 / 2 + 0.3 + 0.2) / 4
        pig_20 = 0.3
        pig_30 = (0.5 + 0.8) / 2

        assert agg.checkpoint_metric() == ((pig_10 + pig_20 + pig_30) / 3)

    def test_cam_metrics(self, df_example: pd.DataFrame) -> None:
        agg = MetricAggregation(
            df_example,
            metrics=["dice_metric", "other_metric"],
            config=Config({
                "input/target_domain": ["camera_index"],
                "label_mapping": LabelMapping({"a": 0, "b": 1}, last_valid_label_index=0),
                "input/data_spec": "pigs_masks_loocv_4cam.json",
            }),
        )
        df_cam = agg.grouped_metrics(domains="camera_index")

        assert len(df_cam) == 4
        assert df_cam.query("label_index == 0")["other_metric"].mean() == 1
        assert pytest.approx(df_cam.query("label_index == 1 and camera_index == 0")["other_metric"].item()) == 2 / 3
        assert df_cam.query("label_index == 1 and camera_index == 1")["other_metric"].item() == 0
        assert round(df_cam.query("label_index == 1")["other_metric"].mean(), 2) == 0.33

    def test_grouped_metrics(self, df_example: pd.DataFrame) -> None:
        agg = MetricAggregation(
            df_example, config=Config({"label_mapping": LabelMapping({"a": 0, "b": 1}, last_valid_label_index=0)})
        )
        df_metrics = agg.grouped_metrics(keep_subjects=True)

        df_target = pd.DataFrame(
            [
                [10, 0, (0.3 + 0.1 + 0.5) / 3, "a"],
                [20, 0, 0.3, "a"],
                [30, 0, 0.8, "a"],
                [10, 1, 0.7 / 3, "b"],
                [30, 1, 0.5, "b"],
            ],
            columns=["subject_name", "label_index", "dice_metric", "label_name"],
        )
        assert_frame_equal(df_metrics, df_target)

        df_metrics = agg.grouped_metrics(mode="image_level")
        df_target = pd.DataFrame([[10, 0.2625], [20, 0.3], [30, 0.65]], columns=["subject_name", "dice_metric"])
        assert_frame_equal(df_metrics, df_target)

    def test_validation(self, df_example: pd.DataFrame) -> None:
        df_example["dataset_index"] = [0, 0, 0, 0, 0, 1, 1]
        df_example["epoch_index"] = [0, 0, 0, 1, 0, 1, 1]
        df_example["best_epoch_index"] = [0, 0, 0, 0, 0, 1, 1]

        agg = MetricAggregation(df_example)
        df_metrics = agg.grouped_metrics()
        df_target = pd.DataFrame([[0, 0.3], [1, 0.25]], columns=["used_labels", "dice_metric"])
        assert_frame_equal(df_metrics, df_target, check_dtype=False)

        df_metrics = agg.grouped_metrics(domains=["epoch_index", "best_epoch_index"], best_epoch_only=True)
        df_target = pd.DataFrame(
            [[0, 0, 0, 0.3], [0, 0, 1, 0.25]], columns=["epoch_index", "best_epoch_index", "used_labels", "dice_metric"]
        )
        assert_frame_equal(df_metrics, df_target, check_dtype=False)

        df_metrics = agg.grouped_metrics(domains=["best_epoch_index"], best_epoch_only=False)
        df_target = pd.DataFrame(
            [
                [0, 0, 0, 0.3],
                [0, 0, 1, 0.25],
                [0, 1, 1, 0.2],
            ],
            columns=["best_epoch_index", "epoch_index", "used_labels", "dice_metric"],
        )
        assert_frame_equal(df_metrics, df_target, check_dtype=False)

    def test_cam_accuracy(self) -> None:
        df = pd.DataFrame(
            [
                [0, 10, 100, 0, [0, 1]],
                [0, 10, 101, 0, [0]],
                [0, 10, 102, 0, [0]],
                [0, 10, 103, 1, [0]],
                [0, 20, 201, 0, [0]],
                [1, 30, 301, 1, [0]],
                [1, 30, 302, 0, [0]],
            ],
            columns=["camera_index", "subject_name", "image_name", "camera_index_predicted", "used_labels"],
        )

        df_result = pd.DataFrame(
            [
                [0, 10, 3 / 4],
                [0, 20, 1],
                [1, 30, 1 / 2],
            ],
            columns=["camera_index", "subject_name", "accuracy"],
        )

        agg = MetricAggregation(
            df,
            config=Config({
                "input/target_domain": ["camera_index"],
                "label_mapping": LabelMapping({"a": 0, "b": 1}, last_valid_label_index=0),
                "input/data_spec": "pigs_masks_loocv_4cam.json",
            }),
            metrics=[],
        )
        assert all(
            agg.domain_accuracy(domain="camera_index")[["camera_index", "subject_name", "accuracy"]] == df_result
        )

    def test_domain_defaults(self) -> None:
        df = pd.DataFrame(
            [
                ["P044#2020_02_01_09_51_15", "P044", "2020_02_01_09_51_15"],
                ["P045#2020_02_05_16_51_41", "P045", "2020_02_05_16_51_41"],
            ],
            columns=["image_name", "subject_name", "timestamp"],
        )
        agg = MetricAggregation(
            df.copy(),
            config=Config({
                "input/target_domain": ["camera_index"],
                "input/data_spec": "pigs_masks_loocv_4cam.json",
            }),
            metrics=[],
        )
        assert agg._domain_defaults(None) == ["camera_index"]
        assert "camera_index" in agg.df

        agg = MetricAggregation(
            df.copy(),
            config=Config({
                "input/target_domain": ["camera_index"],
                "input/data_spec": "pigs_masks_loocv_4cam.json",
            }),
            metrics=[],
        )
        assert agg._domain_defaults(False) == []
        assert "camera_index" not in agg.df

        agg = MetricAggregation(
            df.copy(),
            config=Config({
                "input/data_spec": "pigs_masks_loocv_4cam.json",
            }),
            metrics=[],
        )
        assert agg._domain_defaults("camera_index") == ["camera_index"]
        assert "camera_index" in agg.df

    def test_grouped_cm(self) -> None:
        df_example = pd.DataFrame(
            [
                [0, 10, np.array([[1, 2], [3, 4]])],
                [0, 11, np.array([[0, 1], [0, 1]])],
                [1, 20, np.array([[0, 1], [0, 1]])],
            ],
            columns=["subject_name", "timestamp", "confusion_matrix"],
        )

        agg = MetricAggregation(df_example, metrics=[])
        df_metrics = agg.grouped_cm()
        df_target = pd.DataFrame(
            [
                [0, np.array([[1, 3], [3, 5]])],
                [1, np.array([[0, 1], [0, 1]])],
            ],
            columns=["subject_name", "confusion_matrix"],
        )

        assert_frame_equal(df_metrics, df_target)

    def test_real_data(self) -> None:
        run_dir = settings.training_dir / "image" / "2022-02-03_22-58-44_generated_default_model_comparison"
        config = Config(run_dir / "config.json")

        df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz")
        df_metrics = MetricAggregation(df_val).grouped_metrics()
        df_metrics2 = MetricAggregation(
            df_val.query("epoch_index == best_epoch_index and dataset_index == 0")
        ).grouped_metrics()

        assert_frame_equal(df_metrics, df_metrics2)

        agg = MetricAggregation(df_val)
        df_metrics = agg.grouped_metrics(mode="image_level")
        df_metrics2 = (
            MetricAggregation(df_val, metrics=["dice_metric_image"])
            .grouped_metrics(mode="image_level")
            .rename(columns={"dice_metric_image": "dice_metric"})
        )
        assert_frame_equal(df_metrics, df_metrics2)

        df_cm = agg.grouped_cm()
        assert (df_metrics["subject_name"] == df_cm["subject_name"]).all()
        df_metrics["confusion_matrix"] = df_cm["confusion_matrix"]
        assert (df_metrics["subject_name"] == df_cm["subject_name"]).all()

        df_metric3 = agg.grouped_cm(additional_metrics=["dice_metric"])
        assert_frame_equal(df_metrics[sorted(df_metrics)], df_metrics[sorted(df_metric3)])

        df_metric4 = MetricAggregation(df_val, metrics=["dice_metric", "confusion_matrix"]).grouped_metrics(
            mode="image_level"
        )
        assert_frame_equal(df_metric3, df_metric4)

        df_metrics = MetricAggregation(df_val, config).grouped_metrics()
        assert df_metrics["label_name"][:2].tolist() == ["stomach", "small_bowel"]

    def test_missing_dataset(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        run_dir = settings.training_dir / "image/2022-02-03_22-58-44_generated_default_model_comparison"
        example_name = "P041#2019_12_14_12_00_16"
        assert DataPath.image_name_exists(example_name)

        df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz").query("dataset_index == 0")
        df_metrics1 = MetricAggregation(df_val).grouped_metrics()

        # Make the semantic dataset unavailable (aggregation should still work)
        monkeypatch.setenv("PATH_Tivita_multiorgan_semantic", f"{tmp_path}")
        monkeypatch.setattr(settings, "_datasets", None)
        monkeypatch.setattr(DataPath, "_local_meta_cache", None)
        monkeypatch.setattr(DataPath, "_network_meta_cache", None)
        monkeypatch.setattr(DataPath, "_data_paths_cache", {})
        assert not DataPath.image_name_exists(example_name)

        df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz").query("dataset_index == 0")
        df_metrics2 = MetricAggregation(df_val).grouped_metrics()

        assert_frame_equal(df_metrics1, df_metrics2)
