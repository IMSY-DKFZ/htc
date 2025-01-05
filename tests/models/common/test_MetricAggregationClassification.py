# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from htc.models.common.MetricAggregationClassification import MetricAggregationClassification
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc_projects.rat.settings_rat import settings_rat


class TestMetricAggregationClassification:
    def test_basics(self) -> None:
        df_example = pd.DataFrame(
            [
                ["P001", 0, [0.5, 0.1, 0.1]],
                ["P001", 1, [0.1, 0.5, 0.1]],
                ["P001", 2, [0.1, 0.1, 0.5]],
                ["P001", 2, [0.1, 0.5, 0.1]],  # Wrong class
                # Everything correct
                ["P002", 0, [0.5, 0.1, 0.1]],
                ["P002", 1, [0.1, 0.5, 0.1]],
                ["P002", 2, [0.1, 0.1, 0.5]],
            ],
            columns=["subject_name", "label", "ensemble_logits"],
        )
        config = Config({"label_mapping": LabelMapping({"a": 0, "b": 1, "c": 2})})

        agg = MetricAggregationClassification(
            df_example, config, metrics=["accuracy", "specificity", "recall", "f1_score", "confusion_matrix"]
        )
        df_metrics = agg.subject_metrics()
        assert df_metrics.query('subject_name == "P001"')["accuracy"].item() == 0.75
        assert df_metrics.query('subject_name == "P001"')["specificity"].item() == (3 + 2 + 2) / (3 + 2 + 2 + 1)
        assert df_metrics.query('subject_name == "P001"')["recall"].item() == (1 + 1 + 1) / (1 + 1 + 2)
        assert df_metrics.query('subject_name == "P001"')["f1_score"].item() == 2 * (1 + 1 + 1) / (
            2 * (1 + 1 + 1) + 1 + 1
        )
        cm_P001 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ])
        assert np.all(df_metrics.query('subject_name == "P001"')["confusion_matrix"].item() == cm_P001)

        cm_P002 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        assert df_metrics.query('subject_name == "P002"')["accuracy"].item() == 1
        assert df_metrics.query('subject_name == "P002"')["specificity"].item() == 1
        assert df_metrics.query('subject_name == "P002"')["recall"].item() == 1
        assert df_metrics.query('subject_name == "P002"')["f1_score"].item() == 1
        assert np.all(df_metrics.query('subject_name == "P002"')["confusion_matrix"].item() == cm_P002)

    def test_validation(self) -> None:
        run_dir = settings.training_dir / "median_pixel" / settings_rat.best_run_standardized
        config = Config(run_dir / "config.json")

        # Metrics computed during training
        df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz").query(
            "dataset_index == 0 and epoch_index == best_epoch_index"
        )
        df_val.sort_values(["accuracy"])

        # Metrics re-computed based on the existing confusion matrices
        agg = MetricAggregationClassification(
            run_dir / "validation_table.pkl.xz", config, metrics=["accuracy", "confusion_matrix"]
        )
        df_metrics = agg.subject_metrics()

        assert df_val["subject_name"].tolist() == df_metrics["subject_name"].tolist()
        assert (np.stack(df_val["confusion_matrix"].values) == np.stack(df_metrics["confusion_matrix"].values)).all()
        assert np.allclose(df_val["accuracy"], df_metrics["accuracy"])
