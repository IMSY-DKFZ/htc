# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as metrics

from htc.models.common.utils import get_n_classes
from htc.utils.Config import Config


class MetricAggregationClassification:
    def __init__(self, path_or_df: Union[Path, pd.DataFrame], config: Config, metrics: list[str] = None):
        """
        Class for calculating classification metrics for the tissue atlas.

        Args:
            path_or_df: The path or dataframe of the test table.
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

    def subject_metrics(self, ensemble: str = "logits", **metric_kwargs) -> pd.DataFrame:
        assert ensemble in ["mode", "logits", "softmax"], f"Invalid ensemble approach {ensemble}"

        rows = []
        for subject_name in self.df["subject_name"].unique():
            predictions = np.stack(
                self.df[f"ensemble_{ensemble}"].values[self.df["subject_name"].values == subject_name]
            )
            predictions = torch.from_numpy(predictions)
            labels = self.df["label"].values[self.df["subject_name"].values == subject_name]
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
