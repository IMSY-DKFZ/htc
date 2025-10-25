# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import functools
import json
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from htc.utils.Config import Config
from htc.utils.parallel import p_map
from htc_projects.sepsis_icu.baseline_methods import classifier_results
from htc_projects.sepsis_icu.sepsis_evaluation import compute_metrics
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu
from htc_projects.sepsis_icu.utils import config_meta_selection, target_to_subgroup
from htc_projects.sepsis_icu.visualization_helpers import generate_run_data


def generate_df_feature_adding(target, timedelta, restricted: bool = False):
    if restricted:
        name_str = "_restricted"
    else:
        name_str = ""

    metadata_ranking_path = settings_sepsis_icu.results_dir / f"feature_importance_rankings{name_str}.json"
    with metadata_ranking_path.open("r") as f:
        metadata_ranking_dict = json.load(f)
    metadata_ranking = metadata_ranking_dict[target][str(timedelta)]
    j_max = len(metadata_ranking) + 1

    # extract run data
    target_runs = [f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_palm_image_nested-*-4_seed-*-2"]
    basename = f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_palm_image-meta_top-*-features-{timedelta}hrs_nested-*-4_seed-*-2"

    for j in range(1, j_max):
        target_runs += [basename.replace("top-*", f"top-{j}")]
    run_data = generate_run_data(target, target_runs)

    # generate dataframe
    rows = []
    for i, (run, metric_data) in enumerate(run_data.items()):
        metric_values = metric_data["all"]
        if "image-meta" in run:
            model = "HSI + clinical data"
            n_features = int(re.search(r"top-(\d+)-features", run).group(1))
        elif "image" in run:
            model = "HSI"
            n_features = 0
        rows.append([
            n_features,
            model,
            np.array(metric_values["auroc"]),
            np.array(metric_values["auprc"]),
            np.array(metric_values["minimum_ppv"]),
            np.array(metric_values["brier_binary"]),
        ])

        if n_features >= 1:
            base_config = Config(f"htc_projects/sepsis_icu/configs/{target}-inclusion_palm_meta.json")
            config_meta_selection(base_config, attribute_names=metadata_ranking[:n_features])

            configs = []
            for i in np.arange(5):
                nested_config = copy.deepcopy(base_config)
                nested_config["input/data_spec"] = nested_config["input/data_spec"].replace("test-0.25", f"nested-{i}")
                configs.append(nested_config)

            dfs = []
            for config in configs:
                df = classifier_results(RandomForestClassifier, config, test_results=True, class_weight="balanced")[
                    "df_test"
                ]
                dfs.append(df)

            subgroups, target_dim = target_to_subgroup(target)
            metric_values = compute_metrics(df=dfs, config=config, subgroups=subgroups, target_dim=target_dim)["all"]
            rows.append([
                n_features,
                "clinical data",
                np.array(metric_values["auroc"]),
                np.array(metric_values["auprc"]),
                np.array(metric_values["minimum_ppv"]),
                np.array(metric_values["brier_binary"]),
            ])

    df = pd.DataFrame(rows, columns=["n_features", "model", "AUROC", "AUPRC", "minimum_ppv", "brier_binary"])
    df_exploded = df.copy().explode(["AUROC", "AUPRC", "minimum_ppv", "brier_binary"])
    df_agg = df_exploded.groupby(["model", "n_features"], as_index=False).agg(
        median_AUROC=pd.NamedAgg(column="AUROC", aggfunc=np.median),
        std_AUROC=pd.NamedAgg(column="AUROC", aggfunc=np.std),
        percentile_025_AUROC=pd.NamedAgg(column="AUROC", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_AUROC=pd.NamedAgg(column="AUROC", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=np.median),
        std_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=np.std),
        percentile_025_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_ppv=pd.NamedAgg(column="minimum_ppv", aggfunc=np.median),
        std_ppv=pd.NamedAgg(column="minimum_ppv", aggfunc=np.std),
        percentile_025_ppv=pd.NamedAgg(column="minimum_ppv", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_ppv=pd.NamedAgg(column="minimum_ppv", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=np.median),
        std_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=np.std),
        percentile_025_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_brier_binary=pd.NamedAgg(
            column="brier_binary", aggfunc=functools.partial(np.percentile, q=97.5)
        ),
    )

    df_name = f"{target}_metadata_adding_rf_importances_{timedelta}hrs{name_str}.pkl"

    df["target"] = target
    df["timedelta"] = timedelta
    df_agg["target"] = target
    df_agg["timedelta"] = timedelta

    df.to_pickle(settings_sepsis_icu.results_dir / df_name)
    df_agg.to_pickle(settings_sepsis_icu.results_dir / f"{df_name.replace('.pkl', '_agg.pkl')}")


if __name__ == "__main__":
    params = {
        "target": [],
        "timedelta": [],
        "restricted": [],
    }
    for target in ["sepsis", "survival"]:
        for timedelta in [1, 10]:
            for restricted in [True, False]:
                params["target"].append(target)
                params["timedelta"].append(timedelta)
                params["restricted"].append(restricted)

    p_map(generate_df_feature_adding, *params.values(), num_cpus=2)
