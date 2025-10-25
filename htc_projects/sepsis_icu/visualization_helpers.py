# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import functools
import json
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from htc.settings import settings
from htc.utils.Config import Config
from htc_projects.sepsis_icu.baseline_methods import classifier_results
from htc_projects.sepsis_icu.sepsis_evaluation import compute_metrics
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu
from htc_projects.sepsis_icu.utils import config_meta_selection, target_to_subgroup


def generate_run_data(
    target: str, target_runs: list, table_name: str = "test_table_new", subgroup_selection: list = None
) -> dict:
    run_data = {}
    for run in target_runs:
        results_dir = settings.results_dir / "training"
        subgroups, target_dim = target_to_subgroup(target)
        if subgroup_selection is not None:
            subgroups = subgroup_selection
        seed_runs = list(results_dir.glob(run))
        if len(seed_runs) == 0:
            continue
        if "nested" in run:
            assert len(seed_runs) == 15, f"Instead of 15 runs, {len(seed_runs)} runs were found for {run}"
        elif "seed" in run:
            assert len(seed_runs) == 3, f"Instead of 3 runs, {len(seed_runs)} runs were found for {run}"
        if len(seed_runs) == 0:
            raise ValueError(f"No matching seed runs found for {run}")
        run_data[run.split("/")[1]] = compute_metrics(
            run_dir=seed_runs, subgroups=subgroups, target_dim=target_dim, table_name=table_name
        )

    return run_data


def generate_df(run_data, target, add_random_forest=True, subgroup: str = "all"):
    rows = []
    for i, (run, metric_data) in enumerate(run_data.items()):
        if "image-meta" in run:
            model = "HSI + clinical data"
            match = re.search(r"meta_([\w,+]+)_nested", run)
            assert match is not None, f"Could not extract metadata from {run}"
            metadata = match.group(1)
        elif "image" in run:
            if "rgb" in run:
                model = "RGB"
                metadata = "RGB alone"
            elif "tpi" in run:
                model = "TPI"
                metadata = "TPI alone"
            else:
                model = "HSI"
                metadata = "HSI alone"
        elif "median" in run:
            if "rgb" in run:
                model = "RGB, median"
                metadata = "RGB alone"
            if "tpi" in run:
                model = "TPI, median"
                metadata = "TPI alone"
            else:
                model = "HSI, median"
                metadata = "HSI alone"
        else:
            raise ValueError(f"Unknown model type: {run}")

        if "palm" in run:
            site = "palm"
        elif "finger" in run:
            site = "finger"

        metric_values = metric_data[subgroup]
        rows.append([
            model,
            metadata,
            site,
            np.array(metric_values["auroc"]),
            np.array(metric_values["auprc"]),
            np.array(metric_values["roc"][0]),
            np.array(metric_values["roc"][1]),
            np.array(metric_values["prc"][0]),
            np.array(metric_values["prc"][1]),
            np.array(metric_values["minimum_ppv"]),
            np.array(metric_values["brier_binary"]),
        ])

        if "image-meta" in run and add_random_forest:
            base_config = Config(f"htc_projects/sepsis_icu/configs/{target}-inclusion_{site}_meta.json")
            config_meta_selection(base_config, attribute_names=metadata)

            if "nested" in run:
                configs = []
                for i in np.arange(5):
                    nested_config = copy.deepcopy(base_config)
                    nested_config["input/data_spec"] = nested_config["input/data_spec"].replace(
                        "test-0.25", f"nested-{i}"
                    )
                    configs.append(nested_config)
            else:
                configs = [base_config]

            dfs = []
            for config in configs:
                df = classifier_results(RandomForestClassifier, config, test_results=True, class_weight="balanced")[
                    "df_test"
                ]
                dfs.append(df)

            subgroups, target_dim = target_to_subgroup(target)
            metric_values = compute_metrics(df=dfs, config=configs[0], subgroups=subgroups, target_dim=target_dim)[
                "all"
            ]
            rows.append([
                "clinical data",
                metadata,
                site,
                np.array(metric_values["auroc"]),
                np.array(metric_values["auprc"]),
                np.array(metric_values["roc"][0]),
                np.array(metric_values["roc"][1]),
                np.array(metric_values["prc"][0]),
                np.array(metric_values["prc"][1]),
                np.array(metric_values["minimum_ppv"]),
                np.array(metric_values["brier_binary"]),
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "metadata",
            "site",
            "AUROC",
            "AUPRC",
            "ROC_x",
            "ROC_y",
            "PRC_x",
            "PRC_y",
            "minimum_ppv",
            "brier_binary",
        ],
    )
    df_exploded = (
        df[["model", "metadata", "site", "AUROC", "AUPRC", "brier_binary"]]
        .copy()
        .explode(["AUROC", "AUPRC", "brier_binary"])
    )
    df_agg = df_exploded.groupby(["model", "metadata", "site"], as_index=False).agg(
        median_AUROC=pd.NamedAgg(column="AUROC", aggfunc=np.median),
        std_AUROC=pd.NamedAgg(column="AUROC", aggfunc=np.std),
        percentile_025_AUROC=pd.NamedAgg(column="AUROC", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_AUROC=pd.NamedAgg(column="AUROC", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=np.median),
        std_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=np.std),
        percentile_025_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=np.median),
        std_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=np.std),
        percentile_025_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_brier_binary=pd.NamedAgg(
            column="brier_binary", aggfunc=functools.partial(np.percentile, q=97.5)
        ),
    )

    return df_exploded, df_agg, df


def generate_df_feature_adding(target, timedelta, add_random_forest=True, restricted: bool = False):
    if restricted:
        name_str = "_restricted"
    else:
        name_str = ""

    # extract run data
    target_runs = settings_sepsis_icu.final_model_runs[f"{target}_palm_hsi"]
    basename = settings_sepsis_icu.final_model_runs[
        f"{target}_metadata_adding_rf_importances_{timedelta}hrs{name_str}"
    ][0]
    metadata_ranking_path = settings_sepsis_icu.results_dir / f"feature_importance_rankings{name_str}.json"
    with metadata_ranking_path.open("r") as f:
        metadata_ranking_dict = json.load(f)
    metadata_ranking = metadata_ranking_dict[target][str(timedelta)]
    j_max = len(metadata_ranking) + 1
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
            ])

    df = pd.DataFrame(rows, columns=["n_features", "model", "AUROC", "AUPRC", "minimum_ppv"])
    df_exploded = df.copy().explode(["AUROC", "AUPRC", "minimum_ppv"])
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
    )

    return df_exploded, df_agg


def generate_df_score_comparison(target: str, site: str = "palm"):
    target_runs = [
        f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_{site}_image_nested-*-4_seed-*-2",
        f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_{site}_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-*-4_seed-*-2",
        f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_{site}_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-*-4_seed-*-2",
    ]
    run_data = generate_run_data(target, target_runs)
    df_baseline, _, _ = generate_df(run_data, target)

    timemapping = {
        "HSI alone": "bedside",
        "demographic+vital+BGA+diagnosis+ventilation+catecholamines": "1hr",
        "demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab": "10hrs",
    }
    df_baseline["metadata"] = df_baseline["metadata"].map(timemapping)

    # add score results
    rows = []

    score_dict = settings_sepsis_icu.clinical_scores[target]
    for key, scores in score_dict.items():
        for score in scores:
            base_config = Config(f"htc_projects/sepsis_icu/configs/{target}-inclusion_palm_meta.json")
            config_meta_selection(base_config, attribute_names=[score])

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
            if score == "mottling_score":
                score = "SMS"
            if score == "recap_time":
                score = "CRT"
            rows.append([
                key,
                score,
                np.array(metric_values["auroc"]),
                np.array(metric_values["auprc"]),
                np.array(metric_values["minimum_ppv"]),
                np.array(metric_values["brier_binary"]),
            ])

    df_score = pd.DataFrame(rows, columns=["metadata", "model", "AUROC", "AUPRC", "minimum_ppv", "brier_binary"])
    df_score = df_score.explode(["AUROC", "AUPRC", "minimum_ppv", "brier_binary"])

    df = pd.concat([df_baseline, df_score])
    df_agg = df.groupby(["model", "metadata"], as_index=False).agg(
        median_AUROC=pd.NamedAgg(column="AUROC", aggfunc=np.median),
        std_AUROC=pd.NamedAgg(column="AUROC", aggfunc=np.std),
        percentile_025_AUROC=pd.NamedAgg(column="AUROC", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_AUROC=pd.NamedAgg(column="AUROC", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=np.median),
        std_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=np.std),
        percentile_025_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_AUPRC=pd.NamedAgg(column="AUPRC", aggfunc=functools.partial(np.percentile, q=97.5)),
        median_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=np.median),
        std_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=np.std),
        percentile_025_brier_binary=pd.NamedAgg(column="brier_binary", aggfunc=functools.partial(np.percentile, q=2.5)),
        percentile_975_brier_binary=pd.NamedAgg(
            column="brier_binary", aggfunc=functools.partial(np.percentile, q=97.5)
        ),
    )

    return df, df_agg
