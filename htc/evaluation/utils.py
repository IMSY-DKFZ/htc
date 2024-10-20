# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pandas as pd

from htc import sort_labels
from htc.models.common.MetricAggregation import MetricAggregation
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.Config import Config


def split_test_table(
    run_dir: Path, table_name: str = "test_table.pkl.xz", split_tables: bool = True
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Read a test table which contains multiple test sets. This is the default if your data specification has more than one test set but the TestPredictor only creates one resulting test table.

    The default is to get a test table per split:
    >>> from htc.settings import settings
    >>> tables = split_test_table(settings.training_dir / "image" / "2023-02-21_23-14-44_glove_baseline")
    >>> tables.keys()
    dict_keys(['test', 'test_ood'])
    >>> tables["test"]["dataset"].unique().tolist()
    ['test']

    But it is also possible to retrieve only one final test table:
    >>> df = split_test_table(settings.training_dir / "image" / "2023-02-21_23-14-44_glove_baseline", split_tables=False)
    >>> df["dataset"].unique().tolist()
    ['test', 'test_ood']

    Args:
        run_dir: Path to the training directory.
        table_name: Name of the test table to read.
        split_tables: If True, a dictionary with the test table (value) per split (key) is returned. This makes it easier if you want to aggregate the results per test table. If False, one test table is returned (which still contains the "dataset" column).

    Returns: Test table(s) with easy access to the different splits.
    """
    specs = DataSpecification(run_dir / "data.json")
    specs.activate_test_set()
    df = pd.read_pickle(run_dir / table_name).sort_values(by="image_name")

    # All test splits
    split_names = [n for n in specs.split_names() if n.startswith("test")]
    assert len(split_names) > 0, "Cannot find any test splits"

    # We need to map every path to its corresponding split
    name_to_split = {}
    for split in split_names:
        for p in specs.paths(f"^{split}$"):
            assert p.image_name() not in name_to_split, f"The path {p} is part of more than one split"
            name_to_split[p.image_name()] = split

    # Add a column to the dataframe denoting the corresponding split
    assert len(name_to_split) > 0, "Could not find any test paths"
    assert len(name_to_split) == len(specs.paths("^test")), "Could not map every path to a split"
    df["dataset"] = [name_to_split[name] for name in df["image_name"]]
    assert df["dataset"].unique().tolist() == split_names

    if split_tables:
        # Return a dataframe per split (easier for aggregating)
        tables = {split: df[df["dataset"] == split].reset_index(drop=True) for split in split_names}
        return tables
    else:
        return df


def aggregated_table(run_dir: Path, table_name: str, **kwargs) -> pd.DataFrame | None:
    """
    Read a validation or test table and aggregate the results organ-wise.

    Args:
        run_dir: Path to the training directory.
        table_name: Name of the table to read (e.g. `validation_table`).
        **kwargs: Arguments passed to the `MetricAggregation` class, either to the `__init__()` or to the `grouped_metrics()` method.

    Returns: Table with aggregated metrics per organ or None if no table could be found.
    """
    config = Config(run_dir / "config.json")

    table_path = run_dir / f"{table_name}.pkl.xz"
    if not table_path.exists():
        return None

    df = pd.read_pickle(table_path)
    if table_name.startswith("validation"):
        df = df.query("best_epoch_index == epoch_index and dataset_index == 0")
    df = MetricAggregation(df, config=config, metrics=kwargs.pop("metrics", None)).grouped_metrics(**kwargs)
    df["network"] = run_dir.name[20:]

    df = sort_labels(df)
    return df


def aggregated_confidences_table(run_dir: Path, table_name: str) -> pd.DataFrame:
    """
    Read a validation or test table and aggregate the confidence values per threshold (`DSC_confidences` column). The column can be created via the `run_tables.py` script, e.g.:
    ```bash
    htc tables --model image --run-folder "2023-05-26_21-09-30_humans_extreme" --metrics DSC_confidences --gpu-only
    ```

    Args:
        run_dir: Training run directory.
        table_name: Name of the table to read (e.g. validation_table).

    Returns: Aggregated results per confidence threshold. The `areas` and `dice_metric` columns contain the aggregated values per threshold.
    """
    config = Config(run_dir / "config.json")

    df_results = pd.read_pickle(run_dir / f"{table_name}.pkl.xz")
    if "validation" in table_name:
        df_results = df_results.query("best_epoch_index == epoch_index and dataset_index == 0")

    rows = []
    for _, row in df_results.iterrows():
        conf = row["DSC_confidences"]
        for t, v in conf.items():
            rows.append({
                "image_name": row["image_name"],
                "subject_name": row["subject_name"],
                "timestamp": row["timestamp"],
                "used_labels": row["used_labels"],
                "threshold": t,
                "areas": v["areas"],
                "dice_metric": v["dice_metric"],
            })

    df_thresh = pd.DataFrame(rows)
    df_agg = MetricAggregation(df_thresh, config=config, metrics=["dice_metric", "areas"]).grouped_metrics(
        domains=["threshold"]
    )
    df_agg = df_agg.sort_values(by=["threshold"]).reset_index(drop=True)

    return df_agg


def aggregator_bootstrapping(x: pd.DataFrame, columns: list[str], n_bootstraps: int = 1000) -> pd.Series:
    bootstrap_indices = np.random.randint(0, len(x), (len(x), n_bootstraps))

    res = {}
    for col in columns:
        values = np.nanmean(np.stack(x[col])[bootstrap_indices], axis=0)
        res[col + "_mean"] = np.nanmean(values, axis=0)
        res[col + "_q025"] = np.nanquantile(values, q=0.025, axis=0)
        res[col + "_q975"] = np.nanquantile(values, q=0.975, axis=0)

    return pd.Series(res)
