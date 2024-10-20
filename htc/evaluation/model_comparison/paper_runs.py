# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd

from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.helper_functions import run_info


def collect_comparison_runs(timestamp: str) -> pd.DataFrame:
    run_dirs = []
    for model_name in settings_seg.model_names:
        model_dir = settings.training_dir / model_name
        if len(model_run_dirs := sorted(model_dir.glob(f"{timestamp}*model_comparison"))) > 0:
            run_dirs += model_run_dirs
        elif len(model_run_dirs := sorted(model_dir.glob(timestamp))) > 0:
            run_dirs += model_run_dirs
        else:
            raise ValueError(f"Could not find any run directories with the timestamp {timestamp}")

    assert (
        len(run_dirs) == settings_seg.n_algorithms
    ), f"Could not find runs for all algorithms (only {len(run_dirs)} runs found)"
    model_type_rows = {"rgb": [], "param": [], "hsi": []}
    model_image_size = {
        "image": 1,
        "patch_64": 75,  # 480*640 / 64**2
        "patch_32": 300,
        "superpixel_classification": 1000,
        "pixel": 480 * 640,
    }

    prev_data_spec = None
    for run_dir in run_dirs:
        if "default_128" in run_dir.name:
            # For now we don't use 128x128 patches
            continue

        info = run_info(run_dir)
        model_name = info["model_name"]
        model_type = info["model_type"]
        config = info["config"]
        model = run_dir.parent.name

        if prev_data_spec is None:
            prev_data_spec = config["input/data_spec"]
        else:
            assert prev_data_spec == config["input/data_spec"], (
                f"All runs must use the same data specification file (The {run_dir} has the specification"
                f" {config['input/data_spec']} instead of {prev_data_spec})"
            )

        main_loss = "train/dice_loss_epoch"
        if model == "pixel":
            main_loss = "train/ce_loss_epoch"
        elif model == "superpixel_classification":
            main_loss = "train/kl_loss_epoch"

        model_type_rows[model_type].append([model, model_name, main_loss, run_dir.name, model_image_size[model_name]])

    dfs = []
    for model_type, rows in model_type_rows.items():
        dfs.append(pd.DataFrame(rows, columns=["model", "name", "main_loss", f"run_{model_type}", "model_image_size"]))

    df = pd.concat(dfs, join="inner", axis=1).T.drop_duplicates().T
    df = df.sort_values("model_image_size", ascending=False).reset_index(drop=True)

    return df


def model_comparison_table(df_runs: pd.DataFrame, test: bool = False, metrics: list[str] = None) -> pd.DataFrame:
    df_all = []

    for i, row in df_runs.iterrows():
        df_types = []

        for model_type in ["rgb", "param", "hsi"]:
            run_dir = settings.training_dir / row["model"] / row[f"run_{model_type}"]
            table_name = "test_table.pkl.xz" if test else "validation_table.pkl.xz"
            agg = MetricAggregation(run_dir / table_name, metrics=metrics)
            df_metrics = agg.grouped_metrics(mode="image_level")

            df_types.append(df_metrics)

        df_types = pd.concat(df_types, keys=["rgb", "param", "hsi"], names=["model_type", "row_id"])
        df_types = df_types.reset_index().drop(columns=["row_id"])
        df_all.append(df_types)

    df_all = pd.concat(df_all, keys=df_runs["name"], names=["model_name", "row_id"])
    df_all = df_all.reset_index().drop(columns=["row_id"])

    return df_all
