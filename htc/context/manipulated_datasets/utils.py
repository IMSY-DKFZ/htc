# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

from htc.models.common.MetricAggregation import MetricAggregation
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


def compare_performance(
    experiment_name: str,
    experiment_dir: Path,
    reference_experiment: Path = None,
    return_baseline_cm: bool = False,
) -> Union[Union[torch.FloatTensor, list[str]], Union[torch.FloatTensor, torch.FloatTensor, list[str]]]:
    """
    Creates a matrix to plot the performance difference between an experiment and a reference model.

    Args:
        experiment_name: ["organ_removal_0", "organ_isolation_0", "organ_removal_cloth", "organ_isolation_cloth"], which decides for which experiment the performance comparison is being made.
        experiment_dir: The experiment subdirectories where the inference has been saved.
        reference_experiment: An experiment path with which we want to compare the DSC.
        return_baseline_cm: If True, also return a confusion matrix for the baseline model. This is useful if percentages should be computed.

    Returns: One or two matrices and its column names.
    """
    # Check that the experiment_name is permitted
    assert experiment_name in [
        "organ_removal_0",
        "organ_isolation_0",
        "organ_removal_cloth",
        "organ_isolation_cloth",
    ], (
        "The experiment can only be run for the transformations: organ_removal_0, organ_isolation_0,"
        " organ_removal_cloth, organ_isolation_cloth"
    )

    strings = transform_string(experiment_name)
    exp_string = strings["experiment_subdir_name"]
    mapping = LabelMapping.from_config(Config(experiment_dir / "config_reference.json"))

    # Select all dirs in the experiment dir that start with exp
    all_subdir = list(experiment_dir.iterdir())
    exp_subdirs = []
    for dir in all_subdir:
        if dir.name.startswith("exp"):
            exp_subdirs.append(dir)
    exp_subdirs = sorted(exp_subdirs, key=lambda i: int(i.name.removeprefix(exp_string)))

    # Get all the reference dirs
    if reference_experiment:
        assert reference_experiment.exists()
        all_reference_subdir = list(reference_experiment.iterdir())
        ref_subdirs = []
        for dir in all_reference_subdir:
            if dir.name.startswith("exp"):
                ref_subdirs.append(dir)
        ref_subdirs = sorted(ref_subdirs, key=lambda i: int(i.name.removeprefix(exp_string)))
        assert len(ref_subdirs) == len(exp_subdirs)
    else:
        ref_subdirs = exp_subdirs

    n_labels = len(exp_subdirs)

    if experiment_name == "organ_removal_0" or experiment_name == "organ_removal_cloth":
        # Percentage of dice metric change
        # Create a df with all blacked out labels dice metric
        confusion_matrix = torch.zeros([n_labels, n_labels], dtype=torch.float64)
        confusion_matrix_baseline = torch.zeros([n_labels, n_labels], dtype=torch.float64)

        column_names = []
        for directory, ref_dir in zip(exp_subdirs, ref_subdirs):
            label = int(directory.name.removeprefix(strings["experiment_subdir_name"]))

            # If it has to be compared to another experiment, check its test table, else check the ref table
            if reference_experiment:
                df_test_reference = pd.read_pickle(ref_dir / "test_table_ttt.pkl.xz").sort_values("image_name")
                config_reference_dir = Config(ref_dir / "config.json")
            else:
                df_test_reference = pd.read_pickle(ref_dir / "test_table_reference.pkl.xz").sort_values("image_name")
                config_reference_dir = Config(experiment_dir / "config_reference.json")

            df_test_reference_grouped_mean = MetricAggregation(
                df_test_reference, config=config_reference_dir
            ).grouped_metrics()

            df_test_experiment = pd.read_pickle(directory / "test_table_ttt.pkl.xz").sort_values("image_name")
            test_experiment_explode = df_test_experiment.explode(["dice_metric", "used_labels"]).reset_index(drop=True)

            # Explode and drop the blacked out labels from the not blacked out dataframe
            test_reference_explode = df_test_reference.explode(["dice_metric", "used_labels"]).reset_index(drop=True)

            # assert the same shape that image_name & used_labels are at the same row height
            assert test_experiment_explode.shape[0] == test_reference_explode.shape[0]
            assert test_experiment_explode["image_name"].equals(test_reference_explode["image_name"])
            assert test_experiment_explode["used_labels"].equals(test_reference_explode["used_labels"])

            # Find the difference between dice metrics and aggregate again
            dm_difference = test_reference_explode
            dm_difference["dice_metric"] = test_experiment_explode["dice_metric"].subtract(
                test_reference_explode["dice_metric"]
            )
            dm_difference = dm_difference.groupby(["image_name"], as_index=False).agg(
                {
                    "dice_metric": lambda x: x.tolist(),
                    "used_labels": lambda x: x.tolist(),
                }
            )

            # Aggregate and group with subjects
            config_experiment = Config(directory / "config.json")
            dm_diff_grouped = MetricAggregation(dm_difference, config=config_experiment).grouped_metrics()

            assert np.all(dm_diff_grouped["label_index"] == df_test_reference_grouped_mean["label_index"])
            dice_diff = torch.from_numpy(dm_diff_grouped["dice_metric"].values)

            confusion_matrix[label, dm_diff_grouped["label_index"].values] = dice_diff
            confusion_matrix_baseline[label, df_test_reference_grouped_mean["label_index"].values] = torch.from_numpy(
                df_test_reference_grouped_mean["dice_metric"].values
            )

            column_names.append(mapping.index_to_name(label))

    if experiment_name == "organ_isolation_0" or experiment_name == "organ_isolation_cloth":
        assert not return_baseline_cm, "Not implemented"
        confusion_matrix = torch.zeros([n_labels], dtype=torch.float64)
        column_names = []

        # In the case that no reference_experiment is given, directory and ref_dir are the same
        for directory, ref_dir in zip(exp_subdirs, ref_subdirs):
            # Get label the configs and the test tables
            label = int(directory.name.removeprefix(strings["experiment_subdir_name"]))
            config_experiment = Config(directory / "config.json")
            config_reference_dir = Config(ref_dir / "config.json")

            if reference_experiment:
                df_test_reference = pd.read_pickle(ref_dir / "test_table_ttt.pkl.xz").sort_values("image_name")
                config_reference_dir = Config(ref_dir / "config.json")
            else:
                df_test_reference = pd.read_pickle(directory / "test_table_reference.pkl.xz").sort_values("image_name")
                config_reference_dir = Config(experiment_dir / "config_reference.json")

            # Load and group reference dataframe
            df_test_reference_grouped_mean = MetricAggregation(
                df_test_reference, config=config_reference_dir
            ).grouped_metrics()

            df_test_experiment = pd.read_pickle(directory / "test_table_ttt.pkl.xz").sort_values("image_name")

            # assert the same shape that image_name & used_labels are at the same row height
            assert df_test_experiment.shape[0] == df_test_reference.shape[0]
            assert df_test_experiment["image_name"].tolist() == df_test_reference["image_name"].tolist()
            assert df_test_experiment["used_labels"].tolist() == df_test_reference["used_labels"].tolist()

            dm_difference = df_test_reference
            dm_difference["dice_metric"] = df_test_reference["dice_metric"].subtract(df_test_experiment["dice_metric"])

            # Aggregate and group with subjects
            dm_diff_grouped = MetricAggregation(dm_difference, config=config_experiment).grouped_metrics()
            confusion_matrix[label] = dm_diff_grouped["dice_metric"].item()

            column_names.append(mapping.index_to_name(label))

    if return_baseline_cm:
        return confusion_matrix, confusion_matrix_baseline, column_names
    else:
        return confusion_matrix, column_names


def transform_string(transformation_config: str) -> dict[str, str]:
    """
    Function that returns the corresponding config string for each transformation.

    Args:
        transformation_config: Name of the transformation config.

    Returns: The strings that are used for giving names to directories and test time transformations.
    """
    strings = {}

    if transformation_config == "organ_removal_0":
        strings["experiment_subdir_name"] = "exp_0_organ_removal_"
        strings["ttt_class"] = "htc.context.context_transforms>OrganRemoval"
        strings["ttt_index"] = "target_label"
        strings["ttt_fill_value"] = "0"

    if transformation_config == "organ_isolation_0":
        strings["experiment_subdir_name"] = "exp_0_organ_isolation_"
        strings["ttt_class"] = "htc.context.context_transforms>OrganIsolation"
        strings["ttt_index"] = "target_label"
        strings["ttt_fill_value"] = "0"

    if transformation_config == "organ_removal_cloth":
        strings["experiment_subdir_name"] = "exp_cloth_organ_removal_"
        strings["ttt_class"] = "htc.context.context_transforms>OrganRemoval"
        strings["ttt_index"] = "target_label"
        strings["ttt_fill_value"] = "cloth"

    if transformation_config == "organ_isolation_cloth":
        strings["experiment_subdir_name"] = "exp_cloth_organ_isolation_"
        strings["ttt_class"] = "htc.context.context_transforms>OrganIsolation"
        strings["ttt_index"] = "target_label"
        strings["ttt_fill_value"] = "cloth"

    return strings
