# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pandas as pd

from htc.context.settings_context import settings_context
from htc.evaluation.utils import split_test_table
from htc.models.common.HTCModel import HTCModel
from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.Config import Config
from htc.utils.helper_functions import run_info


def aggregate_removal_table(path: Path) -> pd.DataFrame:
    """
    Read a removal results table and aggregate the scores per image to make the results comparable to the other tables.

    Originally, there are multiple scores per label depending on the number of neighbours (one for each removed neighbour). Aggregation is done by taking the minimal performance per label in an image which corresponds to the performance of an organ if the most important neighbour is missing.

    Args:
        path: Path to the table to read.

    Returns: Table with aggregated results.
    """
    df = pd.read_pickle(path)

    if "surface_distance_metric" in df.columns:
        additional_metrics = ["surface_distance_metric", settings_seg.nsd_aggregation_short]
        additional_metrics_image = ["surface_distance_metric_image", settings_seg.nsd_aggregation]
    else:
        additional_metrics = []
        additional_metrics_image = []

    df.drop(columns=["dice_metric_image", "confusion_matrix"] + additional_metrics_image, errors="ignore", inplace=True)
    df = df.explode(["used_labels", "dice_metric"] + additional_metrics)

    # Keep the same column order in the end
    column_order = [c for c in df.columns if c != "target_label"]

    # Take the minimum for each used label, i.e. keep the worst performance per label (this corresponds to the performance of an organ if the most important neighbour is missing)
    columns = [c for c in df.columns if c not in ["target_label", "dice_metric"] + additional_metrics]
    df = df.groupby(columns, as_index=False).agg(
        dice_metric=pd.NamedAgg(column="dice_metric", aggfunc=min),
        **{m: pd.NamedAgg(column=m, aggfunc=min) for m in additional_metrics},
    )

    # Implode the dataframe (to keep the same format as before)
    columns = [c for c in df.columns if c not in ["used_labels", "dice_metric"] + additional_metrics]
    df = df.groupby(columns, as_index=False).agg(
        {
            "used_labels": lambda x: x.tolist(),
            "dice_metric": lambda x: x.tolist(),
        }
        | {m: lambda x: x.tolist() for m in additional_metrics}
    )

    assert len(df) == df["image_name"].nunique(), "There should be one line per image"
    return df.reindex(columns=column_order)


def context_evaluation_table(run_dir: Path, test: bool = False, aggregate: bool = True) -> pd.DataFrame:
    """
    Collects all the context results for a training run.

    The resulting table include (class-wise aggregated) results for each dataset (original semantic, isolation_0, isolation_cloth, mask_isolation) and network (baseline MIA network, new context network).

    >>> run_dir = settings.training_dir / "image/2023-02-08_14-48-02_organ_transplantation_0.8"
    >>> print("some log messages"); df = context_evaluation_table(run_dir)  # doctest: +ELLIPSIS
    some log messages...
    >>> len(df)
    150
    >>> print(df.head().to_string())
        network   dataset  label_index  dice_metric  surface_distance_metric  surface_dice_metric_mean   label_name
    0  baseline  semantic            6     0.786303                27.260113                  0.704220      stomach
    1  baseline  semantic            5     0.931890                 9.759657                  0.711251  small_bowel
    2  baseline  semantic            4     0.894402                 9.178340                  0.854179        colon
    3  baseline  semantic            3     0.929220                 8.130060                  0.597044        liver
    4  baseline  semantic            8     0.830421                 7.742953                  0.479262  gallbladder

    Args:
        run_dir: Path to the training run to the context network.
        test: If True, read the test table instead of the validation table.
        aggregate: If True, organ-level aggregated results are returned. If False, a much larger table with metric values per image is returned.

    Returns: Table with (aggregated) results.
    """
    config = Config(run_dir / "config.json")

    def read_table(path: Path) -> pd.DataFrame:
        df = pd.read_pickle(path)
        if not test:
            df = df.query("epoch_index == best_epoch_index and dataset_index == 0")
            df = df.reset_index(drop=True)

        df.sort_values("image_name", inplace=True, ignore_index=True)
        return df

    def real_data_tables(names: list[str]) -> list[pd.DataFrame]:
        tables = []

        for name in names:
            # For the real data, we only have test predictions
            table_path_baseline = (
                settings.results_dir
                / "neighbour_analysis"
                / name
                / model
                / run_baseline.name
                / f"test_table_{name}.pkl.xz"
            )
            table_path_context = run_dir / f"test_table_{name}.pkl.xz"
            if not table_path_baseline.exists() or not table_path_context.exists():
                continue

            df_baseline = pd.read_pickle(table_path_baseline)
            df_baseline["network"] = "baseline"
            df_baseline["dataset"] = name
            df_baseline.sort_values("image_name", inplace=True, ignore_index=True)
            tables.append(df_baseline)

            # New network performance on the masks isolation dataset
            df_context = pd.read_pickle(table_path_context)
            df_context.sort_values("image_name", inplace=True, ignore_index=True)
            df_context["network"] = "context"
            df_context["dataset"] = name
            tables.append(df_context)

        return tables

    table_base_name = "test_table" if test else "validation_table"

    # We need access to the results from the corresponding MIA run
    info = run_info(run_dir)
    model = info["model_name"]
    model_type_suffix = "" if info["model_type"] == "hsi" else f"_{info['model_type']}"
    if "glove" in run_dir.name:
        assert info["model_type"] in ["hsi", "rgb"], "glove baseline models are only available for HSI and RGB"
        run_baseline = (
            settings_context.glove_runs["baseline"]
            if info["model_type"] == "hsi"
            else settings_context.glove_runs_rgb["baseline"]
        )
    else:
        run_baseline = HTCModel.find_pretrained_run(
            model, f"{settings_seg.model_comparison_timestamp}_generated_default{model_type_suffix}_model_comparison"
        )

    # Original MIA results
    df_baseline = read_table(run_baseline / f"{table_base_name}.pkl.xz")
    df_baseline["network"] = "baseline"
    df_baseline["dataset"] = "semantic"
    df_baseline.sort_values("image_name", inplace=True, ignore_index=True)

    # Reference data for the context problem
    df_baseline_isolation_0 = read_table(
        settings.results_dir
        / "neighbour_analysis/organ_isolation_0"
        / model
        / run_baseline.name
        / f"{table_base_name}_isolation_0.pkl.xz"
    )
    df_baseline_isolation_0["dataset"] = "isolation_0"
    df_baseline_isolation_0["network"] = "baseline"
    df_baseline_isolation_cloth = read_table(
        settings.results_dir
        / "neighbour_analysis/organ_isolation_cloth"
        / model
        / run_baseline.name
        / f"{table_base_name}_isolation_cloth.pkl.xz"
    )
    df_baseline_isolation_cloth["dataset"] = "isolation_cloth"
    df_baseline_isolation_cloth["network"] = "baseline"

    # New network on clean data
    df_context = read_table(run_dir / f"{table_base_name}.pkl.xz")
    df_context.sort_values("image_name", inplace=True, ignore_index=True)
    df_context["network"] = "context"
    df_context["dataset"] = "semantic"

    # New network on simulated transformed data
    df_context_isolation_0 = read_table(run_dir / f"{table_base_name}_isolation_0.pkl.xz")
    df_context_isolation_0.sort_values("image_name", inplace=True, ignore_index=True)
    df_context_isolation_0["network"] = "context"
    df_context_isolation_0["dataset"] = "isolation_0"

    df_context_isolation_cloth = read_table(run_dir / f"{table_base_name}_isolation_cloth.pkl.xz")
    df_context_isolation_cloth.sort_values("image_name", inplace=True, ignore_index=True)
    df_context_isolation_cloth["network"] = "context"
    df_context_isolation_cloth["dataset"] = "isolation_cloth"

    assert (df_baseline["image_name"].values == df_baseline_isolation_0["image_name"].values).all()
    assert (df_baseline["image_name"].values == df_baseline_isolation_cloth["image_name"].values).all()
    assert (df_baseline["image_name"].values == df_context["image_name"].values).all()
    assert (df_baseline["image_name"].values == df_context_isolation_0["image_name"].values).all()
    assert (df_baseline["image_name"].values == df_context_isolation_cloth["image_name"].values).all()

    # Aggregate results for all tables
    tables = [
        df_baseline,
        df_baseline_isolation_0,
        df_baseline_isolation_cloth,
        df_context,
        df_context_isolation_0,
        df_context_isolation_cloth,
    ]
    tables += real_data_tables(list(settings_context.real_datasets.keys()))

    # Add removal results if they exist
    for name in ["removal_0", "removal_cloth"]:
        path = run_dir / f"{table_base_name}_{name}.pkl.xz"
        if path.exists():
            df_context_removal = aggregate_removal_table(path)
            df_context_removal["network"] = "context"
            df_context_removal["dataset"] = name
            tables.append(df_context_removal)

            df_baseline_removal = aggregate_removal_table(
                settings.results_dir / "neighbour_analysis" / f"organ_{name}" / model / run_baseline.name / path.name
            )
            df_baseline_removal["network"] = "baseline"
            df_baseline_removal["dataset"] = name
            tables.append(df_baseline_removal)

    if aggregate:
        df_agg = []
        for df in tables:
            metrics = [
                m for m in ["dice_metric", "surface_distance_metric", settings_seg.nsd_aggregation_short] if m in df
            ]
            agg = MetricAggregation(
                df,
                config,
                metrics=metrics,
            )
            df_agg.append(agg.grouped_metrics(mode="class_level", domains=["network", "dataset"]))
        assert all(len(df) > 0 for df in df_agg), "All tables must have at least one row"

        return pd.concat(df_agg)
    else:
        return pd.concat(tables)


def compare_context_runs(run_dirs: list[Path], test: bool = False) -> pd.DataFrame:
    """
    Collect all scores for the given training runs and combine it into one table. The network column is adapted to distinguish the different runs.

    Args:
        run_dirs: List of training runs which should be combined.
        test: If True, read the test table instead of the validation table.

    Returns: Table with the combined results.
    """
    dfs = []
    for run_dir in run_dirs:
        # run folder name without the timestamp
        name = run_dir.name[20:]

        df = context_evaluation_table(run_dir, test)
        if "context" in name:
            df = df.replace(to_replace={"network": {"context": name}})
        else:
            df = df.replace(to_replace={"network": {"context": f"context_{name}"}})

        if len(dfs) == 0:
            dfs.append(df)
        else:
            # We need the baseline runs only once
            dfs.append(df.query("network != 'baseline'"))

    return pd.concat(dfs)


def find_best_transform_run(name: str) -> Path:
    """
    From a set of probability run (0.2, 0.4, 0.6, 0.8, 1) for a transformation, selects the run with the best mean dice score. The score is first aggregated for each task and then across tasks.

    >>> print("some_log_messages"); run_dir = find_best_transform_run("jigsaw")  # doctest: +ELLIPSIS
    some_log_messages...
    >>> run_dir.name
    '2023-02-16_21-17-59_jigsaw_0.8'

    Args:
        name: Name of the transformation.

    Returns: Path to the training run which has the best score.
    """
    runs = []
    for run_dir in (settings_context.results_dir / "training" / "image").iterdir():
        if "glove" in run_dir.name:
            continue

        match = re.search(name + "_(?:0.2|0.4|0.6|0.8|1)$", run_dir.name)
        if match is not None:
            runs.append(run_dir)

    assert len(runs) == 5, f"Could not exactly 5 runs for the name {name}"

    df = compare_context_runs(runs)
    df = df.query("network != 'baseline' and dataset in ['semantic', 'isolation_0', 'isolation_cloth']")
    df = df.groupby(["network", "dataset"], as_index=False)["dice_metric"].mean()
    df = df.groupby(["network"], as_index=False)["dice_metric"].mean()

    best_dice = df["dice_metric"].max()
    best_index = df["dice_metric"].argmax()
    best_row = df.iloc[best_index]
    assert best_row["dice_metric"] == best_dice
    best_network = best_row["network"].removeprefix("context_")

    best_run = [r for r in runs if best_network in str(r)]
    assert len(best_run) == 1
    return best_run[0]


def glove_runs(networks: dict[str, Path] = None, aggregate: bool = True) -> pd.DataFrame:
    """
    Collects the test results for all glove runs. There will be two test datasets (glove and no-glove) corresponding to the out-of-distribution and in-distribution, respectively.

    Note: The results are always from the test dataset since only here glove and no-glove is available.

    Args:
        networks: Dictionary of (name, run_dir) pairs of glove runs which should be included in the final table. If None, the default glove runs (as specified in settings_context.glove_runs) are used.
        aggregate: If True, organ-level aggregated results are returned. If False, a much larger table with metric values per image is returned.

    Returns: Table with all aggregated results.
    """

    def aggregate_run(tables: dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
        df_agg = []
        for df in tables.values():
            metrics = [
                m for m in ["dice_metric", "surface_distance_metric", settings_seg.nsd_aggregation_short] if m in df
            ]
            agg = MetricAggregation(
                df,
                config,
                metrics=metrics,
            )
            df_agg.append(agg.grouped_metrics(mode="class_level", domains=["network", "dataset"]))

        df_agg = pd.concat(df_agg)
        return df_agg

    if networks is None:
        networks = settings_context.glove_runs

    runs = []
    for name, run_dir in networks.items():
        config = Config(run_dir / "config.json")
        tables = split_test_table(run_dir)
        for df in tables.values():
            df["network"] = name
            df.replace({"dataset": {"test": "no-glove", "test_ood": "glove"}}, inplace=True)

        if aggregate:
            runs.append(aggregate_run(tables, config))
        else:
            runs += list(tables.values())

    return pd.concat(runs)


def best_run_data(test: bool = False) -> pd.DataFrame:
    """
    Creates a table with the aggregated scores for the best organ transplantation run with both HSI and RGB data.

    Args:
        test: If True, read the test table instead of the validation table.

    Returns: Table with aggregated scores.
    """
    df1 = context_evaluation_table(settings_context.best_transform_runs["organ_transplantation"], test=test)
    df1["modality"] = "HSI"
    df2 = context_evaluation_table(settings_context.best_transform_runs_rgb["organ_transplantation"], test=test)
    df2["modality"] = "RGB"

    networks = ["baseline", "organ_transplantation"]
    df3 = glove_runs({k: v for k, v in settings_context.glove_runs.items() if k in networks})
    df3["modality"] = "HSI"
    df4 = glove_runs({k: v for k, v in settings_context.glove_runs_rgb.items() if k in networks})
    df4.replace({"network": {"baseline_rgb": "baseline"}}, inplace=True)
    df4["modality"] = "RGB"

    df = pd.concat([df1, df2, df3, df4]).query("dataset in @settings_context.task_name_mapping.keys()")
    df.replace({"network": {"context": "organ_transplantation"}}, inplace=True)

    return df
