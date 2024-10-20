# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as metrics

from htc.evaluation.metrics.scores import confusion_matrix_to_predictions
from htc.models.common.utils import get_n_classes
from htc.utils.Config import Config


class MetricAggregationClassification:
    def __init__(self, path_or_df: Path | pd.DataFrame, config: Config, metrics: list[str] = None):
        """
        Class for calculating classification metrics for the tissue atlas.

        Args:
            path_or_df: The path or dataframe of the test or validation table.
            config: The configuration of the training run.
            metrics: Name of the metrics to compute. Must be part of the torchmetrics.functional module.
        """
        if metrics is None:
            metrics = ["accuracy"]
        self.metrics = metrics
        self.config = config

        if isinstance(path_or_df, pd.DataFrame):
            self.df = path_or_df
        elif isinstance(path_or_df, Path):
            self.df = pd.read_pickle(path_or_df)
        else:
            raise ValueError("Neither a dataframe nor path given")

    def subject_metrics(
        self,
        ensemble: str = "logits",
        dataset_index: int | None = 0,
        best_epoch_only: bool = True,
        **metric_kwargs,
    ) -> pd.DataFrame:
        """
        Calculates the metrics for each subject.

        Args:
            ensemble: The ensemble approach to use. Can be "mode", "logits" or "softmax".
            dataset_index: The index of the dataset which is selected in the table (if available). If None, no selection is carried out.
            best_epoch_only: If True, only results from the best epoch are considered (if available). If False, no selection is carried out and you will get aggregated results per epoch_index (which will also be a column in the resulting table).
            **metric_kwargs: Additional keyword arguments for the metrics (passed on to the torchmetrics function).

            Returns: Table with the metrics for each subject.
        """
        assert ensemble in ["mode", "logits", "softmax"], f"Invalid ensemble approach {ensemble}"

        df = self.df
        if dataset_index is not None and "dataset_index" in df:
            df = df[df["dataset_index"] == dataset_index]
        if best_epoch_only and "epoch_index" in df and "best_epoch_index" in df:
            df = df[df["epoch_index"] == df["best_epoch_index"]]

        rows = []
        for subject_name in df["subject_name"].unique():
            if f"ensemble_{ensemble}" not in df and "confusion_matrix" in df:
                # The predictions are not available so we infer them from the confusion matrix
                cm = df["confusion_matrix"].values[df["subject_name"].values == subject_name].item()
                predictions, labels = confusion_matrix_to_predictions(cm)
                predictions = torch.from_numpy(predictions)
                labels = torch.from_numpy(labels)
            else:
                predictions = np.stack(df[f"ensemble_{ensemble}"].values[df["subject_name"].values == subject_name])
                predictions = torch.from_numpy(predictions)
                labels = df["label"].values[df["subject_name"].values == subject_name]
                labels = torch.from_numpy(labels)

            current_row = {"subject_name": subject_name}

            for metric_name in self.metrics:
                assert hasattr(metrics, metric_name), f"There is no metric {metric_name} in torchmetrics"

                metric = getattr(metrics, metric_name)(
                    predictions, labels, task="multiclass", num_classes=get_n_classes(self.config), **metric_kwargs
                )
                if metric.ndimension() == 0:
                    metric = metric.item()
                else:
                    metric = metric.numpy()

                current_row[metric_name] = metric

            rows.append(current_row)

        return pd.DataFrame(rows)
