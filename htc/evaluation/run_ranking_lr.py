# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from htc.evaluation.model_comparison.paper_runs import collect_comparison_runs
from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.helper_functions import run_info
from htc.utils.import_extra import requires_extra
from htc.utils.parallel import p_map

try:
    from challenger_pydocker import ChallengeR

    _missing_library = ""
except ImportError:
    _missing_library = "challenger_pydocker"


def run_table(run_dir: Path, table_name: str, metrics: list[str]) -> pd.DataFrame:
    info = run_info(run_dir)
    domains = None if table_name == "test_table.pkl.xz" else "fold_name"

    df = MetricAggregation(run_dir / table_name, metrics=metrics).grouped_metrics(domains=domains, mode="image_level")
    df["model_name"] = info["model_name"]
    df["model_type"] = info["model_type"]
    df["model_id"] = f"{info['model_name']}#{info['model_type']}"
    df["lr"] = info["config"]["optimization/optimizer/lr"]
    df["run_dir"] = run_dir

    return df


def challengeR_table_runs(runs: list[Path]) -> pd.DataFrame:
    assert len(runs) == settings_seg.n_algorithms

    # Final ranking across all models
    # We use all metrics here even though we make the ranking only based on the dice metric. However, the additional metrics also have a small impact on the bootstrap results
    metrics = ["dice_metric_image", "surface_distance_metric_image", settings_seg.nsd_aggregation]
    df = pd.concat(p_map(partial(run_table, table_name="test_table.pkl.xz", metrics=metrics), runs))

    # We use smallBetter=True in challengeR
    for c in df.columns:
        if "dice" in c:
            df[c] = 1 - df[c]

    # Make names more similar to the paper names
    for name_old, name_new in settings_seg.modality_names.items():
        df["model_id"] = df["model_id"].str.replace(name_old, name_new)
    df["model_id"] = df["model_id"].str.replace("superpixel_classification", "superpixel")

    # Transform table from wide to long format (to pass the metrics as tasks to challengeR)
    columns_keep = ["subject_name", "model_name", "model_type", "model_id", "lr", "run_dir"]
    df = df.melt(id_vars=columns_keep, var_name="metric", value_name="metric_value")

    return df.sort_values(by=["model_id", "subject_name", "metric"], ignore_index=True)


def run_challengeR_lr(row: list[dict[str, str]], df_all: pd.DataFrame) -> Path:
    with ChallengeR() as c:
        model_name = row["model_name"]
        model_type = row["model_type"]
        model_id = row["model_id"]
        df_sel = df_all.query("model_name == @model_name and model_type == @model_type")

        # Ranking across lrs
        c.add_data_matrix(df_sel)
        c.create_challenge(algorithm="lr", case="subject_name", value="dice_metric_image", smallBetter=False)

        # Get ranking between lrs
        df_ranking = c.get_df("ranking$matlist").query("`dummyTask.rank` == 1")
        assert len(df_ranking) == 1
        best_lr = df_ranking["rowname"].item()
        settings.log.info(f"{model_id} = {best_lr}")

        c.generate_report(
            target_file=target_dir / f"{model_id}_best_lr={best_lr}.html", title=f"Comparison of lrs for {model_id}"
        )

        # Get values for the best ranking
        df_values = df_sel.query("lr == @best_lr").copy()
        assert len(df_values) == df_all["subject_name"].nunique()

        return df_values["run_dir"].unique().item()


def run_challengeR_final(df: pd.DataFrame, name: str) -> None:
    with ChallengeR() as c:
        c.add_data_matrix(df)
        c.create_challenge(
            algorithm="model_id", case="subject_name", value="metric_value", task="metric", smallBetter=True
        )

        # Also store the bootstrap results for the visualization in the paper
        df_bootstraps = c.get_df('rankingBootstrapped$bootsrappedRanks["dice_metric_image"]')
        df_bootstraps["final_rank"] = c.get_df('rankingBootstrapped$matlist[["dice_metric_image"]]')["rank"]
        df_bootstraps.sort_values(by="final_rank", inplace=True, ignore_index=True)
        df_bootstraps.to_pickle(target_dir / f"bootstrapped_{name}.pkl.xz")

        c.generate_report(
            target_file=target_dir / f"challengeR_lrs_{name}.html", title=f"Comparison of models with {name} lr each"
        )


@requires_extra(_missing_library)
def main() -> None:
    runs_comparison = []
    for i, row_runs in collect_comparison_runs(settings_seg.model_comparison_timestamp).iterrows():
        for model_type in ["rgb", "param", "hsi"]:
            run_dir = settings.training_dir / row_runs["model"] / row_runs[f"run_{model_type}"]
            runs_comparison.append(run_dir)

    runs_lrs = runs_comparison.copy()
    for model_name in settings_seg.model_names:
        model_dir = settings.training_dir / model_name
        runs_lrs += sorted(model_dir.glob(f"{settings_seg.lr_experiment_timestamp}*"))

    assert len(runs_lrs) == settings_seg.n_algorithms * 3, len(runs_lrs)

    df_all = pd.concat(
        p_map(partial(run_table, table_name="validation_table.pkl.xz", metrics=["dice_metric_image"]), runs_lrs)
    )
    n_runs = settings_seg.n_algorithms * df_all["fold_name"].nunique() * 3
    assert np.all(df_all.groupby("lr").count()["subject_name"].values == np.array([n_runs] * 3))

    df_models = df_all.groupby(["model_name", "model_type", "model_id"], as_index=False).size()
    assert set(df_models["size"].values) == {len(df_all) // df_all["subject_name"].nunique()}
    df_models = df_models[["model_name", "model_type", "model_id"]]
    assert len(df_models) == settings_seg.n_algorithms

    # Make a ranking per model to find the best lr
    runs_best = p_map(partial(run_challengeR_lr, df_all=df_all), df_models.to_dict(orient="records"))

    df_best = challengeR_table_runs(runs_best)  # Ranking based on the best lr
    df_best.to_pickle(target_dir / "runs_best.pkl.xz")
    df_fixed = challengeR_table_runs(runs_comparison)  # Ranking all with the same lr (= ranking of the paper)
    df_fixed.to_pickle(target_dir / "runs_fixed.pkl.xz")

    p_map(run_challengeR_final, [df_best, df_fixed], ["best", "fixed"])


if __name__ == "__main__":
    target_dir = settings_seg.results_dir / "challengeR/lrs"
    target_dir.mkdir(exist_ok=True, parents=True)

    main()
