# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from htc.models.common.MetricAggregation import MetricAggregation
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping
from htc_projects.species.settings_species import settings_species
from htc_projects.species.tables import icg_table, ischemic_table


def load_nested_table(run_folder: str, table_name: str, model: str = "image") -> pd.DataFrame:
    assert run_folder.count("*") == 1, f"There must be exactly one wildcard in the run folder name: {run_folder}"

    match = re.search(r"nested-\*-(\d+)", run_folder)
    assert match is not None, f"Could not infer the number of folds from the run folder name: {run_folder}"
    n_folds = int(match.group(1)) + 1

    dfs = []
    image_names = set()
    runs = sorted((settings.training_dir / model).glob(run_folder))
    assert len(runs) == n_folds, f"Expected {n_folds} runs, but found {len(runs)}"

    for run_dir in runs:
        table_path = run_dir / f"{table_name}.pkl.xz"
        if table_path.exists():
            df = pd.read_pickle(table_path)
            assert set(df["image_name"]).isdisjoint(image_names), (
                f"The same images must not be used across runs:\n{runs = }\n{run_dir = }"
            )
            image_names.update(df["image_name"])
            dfs.append(df)

    assert len(dfs) > 0, f"No tables found for the run folder {run_folder}"
    return pd.concat(dfs, ignore_index=True)


def parameter_evaluation(
    df: pd.DataFrame, param_name: str, metric_name: str = "dice_metric", n_bootstraps: int = 1000
) -> pd.DataFrame:
    """
    Evaluate across parameter values for a table of segmentation scores.

    Args:
        df: The table with the segmentation scores. It must contain the scores (e.g., dice_metric) per label and subject and a column with the parameter.
        param_name: The name of the parameter to be evaluated. Must exist as column in the table.
        metric_name: The name of the metric to be evaluated. Must exist as column in the table.
        n_bootstraps: The number of bootstraps to perform.

    Returns: A table with the evaluation results.
    """
    params = np.linspace(0, 1, 100)

    def _param_interpolate(x: pd.DataFrame) -> pd.Series:
        assert len(x) >= 1, "There must be at least one row"

        if len(x) == 1:
            # No interpolation possible with only one value.
            param_index = (np.abs(params - x[param_name].item())).argmin()
            param_score = np.full(len(params), np.nan)
            param_score[param_index] = x[metric_name].item()
            return pd.Series({f"{metric_name}_{param_name}": param_score})
        else:
            param2score = interp1d(x[param_name], x[metric_name], bounds_error=False)
            return pd.Series({f"{metric_name}_{param_name}": param2score(params)})

    # A line for the parameter per label_name and subject_name
    dfg = df.groupby(["label_name", "subject_name"], as_index=False).apply(_param_interpolate, include_groups=False)

    def _aggregator_bootstrapping(x: pd.DataFrame) -> pd.Series:
        bootstrap_indices = np.random.randint(0, len(x), (len(x), n_bootstraps))

        # We bootstrap the subject lines
        scores = np.stack(x[f"{metric_name}_{param_name}"])  # [n_subjects, n_params]
        scores_bootstrap = np.nanmean(scores[bootstrap_indices], axis=0)  # [n_bootstraps, n_params]

        return pd.Series({
            param_name: params,  # [n_params]
            f"{metric_name}_{param_name}": np.nanmean(scores, axis=0),  # [n_params]
            f"{metric_name}_{param_name}_bootstraps": scores_bootstrap.T,  # [n_params, n_bootstraps]
            f"{metric_name}_{param_name}_mean": np.nanmean(scores_bootstrap, axis=0),  # [n_params]
            f"{metric_name}_{param_name}_q025": np.nanquantile(scores_bootstrap, q=0.025, axis=0),  # [n_params]
            f"{metric_name}_{param_name}_q975": np.nanquantile(scores_bootstrap, q=0.975, axis=0),  # [n_params]
        })

    dfg = dfg.groupby("label_name", as_index=False).apply(_aggregator_bootstrapping, include_groups=False)

    # One row per parameter value
    dfg = dfg.explode(dfg.columns.difference(["label_name"]).tolist()).reset_index(drop=True)

    return dfg


def parameter_comparison(
    runs: dict[str, Path], table_name: str, param_name: str, target_labels: list[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare multiple runs across a parameter.

    Args:
        runs: A dictionary mapping run names to their corresponding directories. The run names will appear in the output table in the network column.
        table_name: The name of the table to load the scores from.
        param_name: The name of the parameter to be evaluated.
        target_labels: Restrict the analysis to these labels.

    Returns: A tuple containing the bootstrapped evaluation scores and raw, subject-level data as tables.
    """
    dfs_scores = []
    dfs_raw = []
    image_names_prev = None
    image_names_prev_baseline = None
    for name, run_dir in runs.items():
        config = Config(run_dir.with_name(run_dir.name.replace("nested-*-2", "nested-0-2")) / "config.json")
        label_mapping = LabelMapping.from_config(config)

        df = load_nested_table(run_folder=run_dir.name, table_name=table_name)
        if image_names_prev is None:
            image_names_prev = set(df["image_name"])
        else:
            assert image_names_prev == set(df["image_name"]), (
                f"The same images must be used across runs: {run_dir.name}"
            )

        df_baseline = load_nested_table(run_folder=run_dir.name, table_name=table_name.removesuffix("_perfusion"))
        if image_names_prev_baseline is None:
            image_names_prev_baseline = set(df_baseline["image_name"])
        else:
            assert image_names_prev_baseline == set(df_baseline["image_name"]), (
                f"The same images must be used across baseline runs: {run_dir.name}"
            )

        df = pd.concat([df, df_baseline], ignore_index=True)

        df = df.explode(["used_labels", "dice_metric"])
        df = df.rename(columns={"used_labels": "label_index"}).infer_objects()
        df["label_name"] = [label_mapping.index_to_name(i) for i in df["label_index"]]

        if target_labels is not None:
            df = df[df["label_name"].isin(target_labels)]

        image_name_annotations = df["image_name"] + "@" + df["annotation_name"]

        # Sanity check
        match = re.search(r"(pig|rat|human)_nested", run_dir.name)
        assert match is not None, f"Could not infer species from run folder name {run_dir.name}"
        species = match.group(1)
        df_ischemic = ischemic_table(label_mapping)
        df_ischemic = df_ischemic[df_ischemic["species_name"] == species]
        if target_labels is not None:
            df_ischemic = df_ischemic[df_ischemic["label_name"].isin(target_labels)]

        assert set(image_name_annotations) == set(df_ischemic["image_name"] + "@" + df_ischemic["annotation_name"]), (
            "Unexpected image names"
        )

        df_median = median_table(image_names=image_name_annotations.tolist())
        df = df.merge(
            df_median,
            on=["image_name", "label_name", "annotation_name"],
            how="left",
            suffixes=("", "_y"),
            validate="one_to_one",
        )
        assert not pd.isna(df["median_sto2"]).any(), (
            "Could not match every image to find the corresponding parameter value"
        )

        np.random.seed(42)  # Same bootstraps for each run

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="Mean of empty slice|All-NaN slice encountered", category=RuntimeWarning
            )
            df_scores = parameter_evaluation(df, param_name)

        df_scores["network"] = name
        dfs_scores.append(df_scores)

        df["network"] = name
        dfs_raw.append(df)

    assert image_names_prev is not None

    return pd.concat(dfs_scores, ignore_index=True), pd.concat(dfs_raw, ignore_index=True)


def baseline_performance() -> pd.DataFrame:
    np.random.seed(42)

    runs = sorted((settings.training_dir / "image").glob(f"{settings_species.model_timestamp}_baseline*_nested-0-2"))
    runs += sorted(
        (settings.training_dir / "image").glob(f"{settings_species.model_timestamp}_joint_pig-p+rat-p2human_nested-0-2")
    )

    dfs = []
    for run_dir in runs:
        match = re.search(r"(pig|rat|human|pig-p\+rat-p2human)(?:_nested-\d+-\d+)?$", run_dir.name)
        assert match is not None, f"Could not infer species from run folder name {run_dir.name}"
        species = match.group(1)

        config = Config(run_dir / "config.json")

        for target_species in settings_species.species_colors.keys():
            df = load_nested_table(
                run_dir.name.replace("nested-0-2", "nested-*-2"), table_name=f"test_table_{target_species}"
            )
            df = MetricAggregation(
                df, config=config, metrics=["dice_metric", settings_seg.nsd_aggregation_short]
            ).grouped_metrics(domains=[], n_bootstraps=1000)
            df["network"] = re.sub(r"_nested-\d+-\d+", "", run_dir.name[20:])
            df["source_species"] = species
            df["target_species"] = target_species

            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def icg_performance(runs: list[Path]) -> pd.DataFrame:
    np.random.seed(42)

    df_icg = icg_table()
    df_icg = df_icg[df_icg["perfusion_state"] == "icg"]

    species_images = {}
    dfs = []
    for run_dir in runs:
        match = re.search(r"(pig|rat|human|pig-p\+rat-p2human)(?:_nested-\d+-\d+)?$", run_dir.name)
        assert match is not None, f"Could not infer species from run folder name {run_dir.name}"
        species = match.group(1)

        config = Config(run_dir / "config.json")

        df = load_nested_table(run_dir.name.replace("nested-0-2", "nested-*-2"), table_name=f"test_table_{species}_icg")
        df = df[df.image_name.isin(df_icg.image_name)]

        if species in species_images:
            assert set(df["image_name"]) == species_images[species], "The same images per species should be used"
        else:
            species_images[species] = set(df["image_name"])

        df = MetricAggregation(df, config=config, metrics=["dice_metric"]).grouped_metrics(
            domains=[], keep_subjects=True
        )
        df = df[df.label_name.isin(settings_species.icg_labels)]
        df["network"] = re.sub(r"_nested-\d+-\d+", "", run_dir.name[20:])
        df["species"] = species

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


class TrainingDistanceComputation:
    def __init__(self, param_name: str):
        self.param_name = param_name
        self.df_all = ischemic_table()
        self.df_projections = pd.read_feather(
            settings_species.results_dir / "projections" / "projections_clear.feather"
        )

    def compute_distances(
        self, target_species: str, source_species: str, label_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_xeno = self.df_projections[
            (self.df_projections.source_species == source_species)
            & (self.df_projections.target_species == target_species)
            & (self.df_projections.label_name == label_name)
        ]
        df_real = self.df_all[(self.df_all.species_name == target_species) & (self.df_all.label_name == label_name)]

        rows_data = {
            "image_name": [],
            "subject_name": [],
            "label_name": [],
            self.param_name: [],
        }
        baseline = []
        extended = []

        # For the data used during training, we need to consider the nested folds
        # The test data per nested fold is compared to the training data
        for nested_index in range(settings_species.n_nested_folds):
            spec = DataSpecification(
                settings_species.spec_names[target_species].replace("nested-*", f"nested-{nested_index}")
            )
            spec.activate_test_set()
            images_train = {p.image_name() for p in spec.paths("train")}
            images_test = {p.image_name() for p in spec.paths("test")}
            assert len(images_train) + len(images_test) == len(spec.paths())

            df_real_train = df_real[df_real.image_name.isin(images_train)]
            df_real_test = df_real[df_real.image_name.isin(images_test)]

            X_train = torch.from_numpy(np.stack(df_real_train["median_normalized_spectrum"], axis=0))
            X_train_extended = torch.from_numpy(
                np.stack(df_xeno[df_xeno.image_name.isin(images_train)]["median_normalized_spectrum"], axis=0)
            )
            X_test = torch.from_numpy(np.stack(df_real_test["median_normalized_spectrum"], axis=0))

            rows_data["image_name"].extend(df_real_test["image_name"])
            rows_data["subject_name"].extend(df_real_test["subject_name"])
            rows_data["label_name"].extend(df_real_test["label_name"])
            rows_data[self.param_name].extend(df_real_test[self.param_name])
            baseline.extend(torch.cdist(X_test, X_train).min(dim=1).values.tolist())
            extended.extend(torch.cdist(X_test, X_train_extended).min(dim=1).values.tolist())

        # The remaining data (perfusion data) is only used as test set
        df_real_train = df_real[df_real.baseline_dataset]
        df_real_test = df_real[~df_real.baseline_dataset]

        X_train = torch.from_numpy(np.stack(df_real_train["median_normalized_spectrum"], axis=0))
        X_test = torch.from_numpy(np.stack(df_real_test["median_normalized_spectrum"], axis=0))
        X_train_extended = torch.from_numpy(np.stack(df_xeno["median_normalized_spectrum"], axis=0))

        assert set(df_real_train.image_name) == set(df_xeno.image_name), "Same training images should be used"
        assert set(rows_data["image_name"]) == set(df_real_train.image_name), (
            "All previously tested images should now comprise the training data"
        )
        assert set(rows_data["image_name"]).intersection(df_real_test.image_name) == set(), (
            "The perfusion data should not be part of any spec"
        )

        rows_data["image_name"].extend(df_real_test["image_name"])
        rows_data["subject_name"].extend(df_real_test["subject_name"])
        rows_data["label_name"].extend(df_real_test["label_name"])
        rows_data[self.param_name].extend(df_real_test[self.param_name])
        baseline.extend(torch.cdist(X_test, X_train).min(dim=1).values.tolist())
        extended.extend(torch.cdist(X_test, X_train_extended).min(dim=1).values.tolist())

        df_baseline_raw = pd.DataFrame(rows_data | {"euclidean_distance": baseline})
        df_baseline_raw["network"] = "in-species"
        df_extended_raw = pd.DataFrame(rows_data | {"euclidean_distance": extended})
        df_extended_raw["network"] = "xeno-learning"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="Mean of empty slice|All-NaN slice encountered", category=RuntimeWarning
            )

            np.random.seed(42)
            df_baseline = parameter_evaluation(df_baseline_raw, self.param_name, metric_name="euclidean_distance")
            df_baseline["network"] = "in-species"

            np.random.seed(42)
            df_extended = parameter_evaluation(df_extended_raw, self.param_name, metric_name="euclidean_distance")
            df_extended["network"] = "xeno-learning"

        return pd.concat([df_baseline, df_extended]), pd.concat([df_baseline_raw, df_extended_raw])
