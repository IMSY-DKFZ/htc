# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pandas as pd

from htc.models.common.MetricAggregation import MetricAggregation
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.parallel import p_map


def dataset_size_table(runs: list[Path], metric_name: str = "dice_metric") -> pd.DataFrame:
    global run_results  # Make function accessible to p_map

    def run_results(run_dir: Path) -> pd.DataFrame:
        rows = []

        model = run_dir.parent.name
        if model == "patch":
            config = Config(run_dir / "config.json")
            model = f"{model}_{config['input/patch_size'][0]}"

        df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz")
        for fold_name in df_val["fold_name"].unique():
            df_fold = df_val.query("fold_name == @fold_name")

            agg = MetricAggregation(df_fold, metrics=[metric_name])
            metric_images = agg.grouped_metrics(mode="image_level")[metric_name].mean()
            metric_classes = agg.grouped_metrics(mode="class_level")[metric_name].tolist()

            match = re.search(r"fold_pigs=(\d+)_seed=(\d+)", fold_name)
            assert match is not None

            rows.append([model, int(match.group(1)), int(match.group(2)), metric_images, *metric_classes])

        return pd.DataFrame(rows, columns=["model", "n_pigs", "seed", "metric_images", *mapping.label_names()])

    config = Config(runs[0] / "config.json")
    mapping = LabelMapping.from_config(config)
    results = pd.concat(p_map(run_results, runs)).sort_values(by=["n_pigs", "seed"])

    del run_results
    return results
