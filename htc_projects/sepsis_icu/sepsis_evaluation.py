# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc, average_precision, precision_recall_curve, recall, roc

from htc.evaluation.bootstrapped_metric import bootstrapped_metric
from htc.evaluation.metrics.BrierScore import BrierScore
from htc.models.common.utils import get_n_classes
from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu


def compute_metrics(
    subgroups: list[str],
    target_dim: int,
    run_dir: Path | list[Path] = None,
    df: pd.DataFrame | list[pd.DataFrame] = None,
    config: Config = None,
    table_name: str = "validation_table",
    filters: list[Callable] | None = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor]]:
    """
    Compute evaluation metrics for a sepsis study run (or the given table) with bootstrapping.

    Args:
        subgroups: A list of subgroups to compute metrics for, in addition to computing them on the entire dataset, e.g. ["septic_shock"].
        target_dim: The class index to compute the metrics for.
        run_dir: The path to the training run or a list of path runs for which predictions will be averaged.
        df: Table with the evaluation data or a list of tables for which predictions will be averaged.
        config: The configuration of the run (if run_dir is not given).
        table_name: Name of the table to read (e.g. `validation_table`).
        filters: A list of functions that act on the dataframe before computing the metrics, e.g. first_timepoint_filter restricts the metric computation to first timepoints only.

    Returns: A dictionary containing the computed metrics with keys for the different subgroups.
    """
    if not isinstance(run_dir, list):
        run_dir = [run_dir]

    if config is None:
        assert run_dir[0] is not None, "Either run_dir or config must be provided"
        config = Config(run_dir[0] / "config.json")

    if df is None:
        assert run_dir[0] is not None, "Either run_dir or df must be provided"
        dfs = [pd.read_pickle(r / f"{table_name}.pkl.xz") for r in run_dir]
        if table_name.startswith("validation"):
            dfs = [df.query("best_epoch_index == epoch_index and dataset_index == 0") for df in dfs]
    else:
        if not isinstance(df, list):
            dfs = [df]
        else:
            dfs = df

    if filters is None:
        filters = []

    for f in filters:
        dfs = [f(df, config=config) for df in dfs]

    # stack and aggregate the dataframes
    df = pd.concat(dfs)
    df = (
        df.groupby(["image_name", "image_labels"], as_index=False)
        .agg(predictions=pd.NamedAgg(column="predictions", aggfunc=lambda x: np.mean(np.stack(x), axis=0)))
        .reset_index()
    )
    assert len(df) == df.image_name.nunique(), "Image names are not unique"
    specs = DataSpecification.from_config(config)
    specs.activate_test_set()
    if "nested" in config["input/data_spec"] and table_name == "test_table_new":
        assert len(df) == len(specs.paths("^test")) + len(specs.paths("^val")), (
            "Length of DataFrame does not match the number of paths in the data specification"
        )

    predictions = torch.from_numpy(np.stack(df["predictions"]))
    predicted_prob = predictions.softmax(dim=1)
    if table_name == "test_table_unclear_sepsis_status":
        df_image_labels = median_table("2022_10_24_Tivita_sepsis_ICU#subjects", table_name="recalibrated")[
            ["image_name", "enforced_sepsis_status"]
        ]
        df_image_labels["enforced_sepsis_status"] = [
            settings_sepsis_icu.sepsis_label_mapping.name_to_index(l) for l in df_image_labels["enforced_sepsis_status"]
        ]
        df_image_labels.rename({"enforced_sepsis_status": "image_labels"}, axis=1, inplace=True)
        df.set_index("image_name", inplace=True)
        df_image_labels.set_index("image_name", inplace=True)
        df.update(df_image_labels, overwrite=True)
        df.reset_index(inplace=True)
        assert sorted(df.image_labels.unique()) == [0, 1], f"Unclear sepsis status labels in {table_name} for {run_dir}"

    image_labels = torch.from_numpy(df["image_labels"].values)

    paths = [DataPath.from_image_name(name) for name in df["image_name"]]
    analysis_tuple = [(predicted_prob, image_labels, "all")]
    if "septic_shock" in subgroups:
        shock_subset = torch.tensor([p.meta("sepsis_status") == "no_sepsis" or p.meta("septic_shock") for p in paths])
        if shock_subset.any():
            analysis_tuple.append((predicted_prob[shock_subset], image_labels[shock_subset], "septic_shock"))
    if "septic_no_shock" in subgroups:
        no_shock_subset = torch.tensor([not p.meta("septic_shock") for p in paths])
        if no_shock_subset.any():
            analysis_tuple.append((predicted_prob[no_shock_subset], image_labels[no_shock_subset], "septic_no_shock"))
    if "low_SOFA" in subgroups:
        low_sofa_subset = torch.tensor([
            p.meta("sepsis_status") == "no_sepsis" or (p.meta("SOFA") is not None and p.meta("SOFA") <= 11)
            for p in paths
        ])
        if low_sofa_subset.any():
            analysis_tuple.append((predicted_prob[low_sofa_subset], image_labels[low_sofa_subset], "low_SOFA"))
    if "high_SOFA" in subgroups:
        high_sofa_subset = torch.tensor([
            p.meta("sepsis_status") == "no_sepsis" or (p.meta("SOFA") is not None and p.meta("SOFA") > 11)
            for p in paths
        ])
        if high_sofa_subset.any():
            analysis_tuple.append((predicted_prob[high_sofa_subset], image_labels[high_sofa_subset], "high_SOFA"))
    if "sepsis" in subgroups:
        sepsis_subset = torch.tensor([p.meta("sepsis_status") == "sepsis" for p in paths])
        analysis_tuple.append((predicted_prob[sepsis_subset], image_labels[sepsis_subset], "sepsis"))
    if "no_sepsis" in subgroups:
        no_sepsis_subset = torch.tensor([p.meta("sepsis_status") == "no_sepsis" for p in paths])
        analysis_tuple.append((predicted_prob[no_sepsis_subset], image_labels[no_sepsis_subset], "no_sepsis"))

    torch.manual_seed(42)
    metric_data = {}
    for preds, target, name in analysis_tuple:

        def calc_metrics(indices: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor]]:
            n_thresholds = 500
            res = {}
            res["balanced_accuracy"] = recall(
                preds[indices, :],
                target[indices],
                task="multiclass",
                average="macro",
                num_classes=get_n_classes(config),
            )
            res["auroc"] = auroc(
                preds=preds[indices, :],
                target=target[indices],
                thresholds=n_thresholds,
                average="none",
                num_classes=get_n_classes(config),
                task="multiclass",
            )
            res["roc"] = roc(
                preds=preds[indices, :],
                target=target[indices],
                thresholds=n_thresholds,
                num_classes=get_n_classes(config),
                task="multiclass",
                validate_args=True,
                average=None,
            )
            res["prc"] = precision_recall_curve(
                preds=preds[indices, :],
                target=target[indices],
                thresholds=n_thresholds,
                num_classes=get_n_classes(config),
                task="multiclass",
                validate_args=True,
                average=None,
            )
            res["auprc"] = average_precision(
                preds=preds[indices, :],
                target=target[indices],
                thresholds=n_thresholds,
                num_classes=get_n_classes(config),
                task="multiclass",
                validate_args=True,
                average=None,
            )
            res["minimum_ppv"] = (target[indices] == target_dim).sum() / len(target[indices])

            index20 = np.argmin(np.abs(res["roc"][0] - 0.2), axis=1)
            res["tpr@fpr=0.2"] = torch.stack([res["roc"][1][i, j] for i, j in enumerate(index20)])

            res["brier_binary"] = BrierScore(n_classes=get_n_classes(config), variant="binary")(
                preds[indices, :], target[indices]
            )

            res["tpr@fpr=0.2"] = res["tpr@fpr=0.2"][target_dim]

            res["roc"] = list(res["roc"])
            res["roc"][0] = res["roc"][0][target_dim, :]  # fpr
            res["roc"][1] = res["roc"][1][target_dim, :]  # tpr
            res["roc"] = tuple(res["roc"])

            res["prc"] = list(res["prc"])
            res_prc_copy = res["prc"].copy()
            res["prc"][0] = res_prc_copy[1][target_dim, :]  # recall
            res["prc"][1] = res_prc_copy[0][target_dim, :]  # precision
            res["prc"] = tuple(res["prc"])

            res["auroc"] = res["auroc"][target_dim]
            res["auprc"] = res["auprc"][target_dim]

            return res

        metric_data[name] = bootstrapped_metric(n_samples=len(target), metric_calculator=calc_metrics)

    return metric_data
