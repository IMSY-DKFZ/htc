# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import sys
from pathlib import Path

import htc_projects.context.extra_datasets.run_dataset_tables as run_dataset_tables
import htc_projects.context.manipulated_datasets.run_context_evaluation_table as run_context_evaluation_table
from htc.evaluation.model_comparison.paper_runs import collect_comparison_runs
from htc.models.common.HTCModel import HTCModel
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.general import subprocess_run
from htc_projects.context.models.run_context_test_tables import compute_glove_test_tables
from htc_projects.context.settings_context import settings_context


def compute_context_tables(runs: list[Path], table_name: str, recalculate: bool = False) -> None:
    """
    Compute missing tables for all context datasets except glove runs for the baseline network.

    Note: Results for the masks_isolation table are only computed when test data is requested.

    Args:
        runs: List of training run directories.
        table_name: Name of the table to compute (`validation_table` or `test_table`).
        recalculate: If True, compute the table even if it already exists (overwrites existing tables). If False, existing tables are skipped.
    """
    for run_dir in runs:
        assert run_dir.exists(), f"The run directory {run_dir} does not exist. Cannot compute the context tables"

    # Test table on the real dataset
    neighbour_dir = settings.results_dir / "neighbour_analysis"

    for run_dir in runs:
        if table_name == "test_table":
            # We don't need to compute the default test table again because it already contains all the metrics

            test_table_real = (
                neighbour_dir
                / "masks_isolation"
                / run_dir.parent.name
                / run_dir.name
                / "test_table_masks_isolation.pkl.xz"
            )
            if not test_table_real.exists() or recalculate:
                settings.log.info(f"Computing masks isolation test table for {run_dir}")
                res = subprocess_run([
                    sys.executable,
                    run_dataset_tables.__file__,
                    "--model",
                    run_dir.parent.name,
                    "--run-folder",
                    run_dir.name,
                    "--metrics",
                    "DSC",
                    "ASD",
                    "NSD",
                    "--test",
                    "--dataset-name",
                    "masks_isolation",
                    "--output-dir",
                    str(test_table_real.parent),
                ])
                assert res.returncode == 0, (
                    f"Computation of the masks isolation test table for the run folder {run_dir} was not successful"
                )
                assert test_table_real.exists()

        # Tables on the simulated datasets
        simulated_datasets = ["isolation_0", "isolation_cloth", "removal_0", "removal_cloth"]
        missing_datasets = [
            d
            for d in simulated_datasets
            if not (
                neighbour_dir / f"organ_{d}" / run_dir.parent.name / run_dir.name / f"{table_name}_{d}.pkl.xz"
            ).exists()
            or recalculate
        ]
        if len(missing_datasets) > 0:
            settings.log.info(f"Computing results on the simulated datasets {missing_datasets} for {run_dir}")
            args = [
                sys.executable,
                run_context_evaluation_table.__file__,
                "--model",
                run_dir.parent.name,
                "--run-folder",
                run_dir.name,
                "--transformation-name",
                *missing_datasets,
                "--output-dir",
                neighbour_dir,
            ]
            if table_name == "test_table":
                args += ["--test"]

            res = subprocess_run(args)
            assert res.returncode == 0, (
                f"Computation of the context table for the run folder {run_dir} was not successful"
            )

        # The tables are temporarily stored in the neighbour folder and then moved to the corresponding dataset folder
        for dataset in missing_datasets:
            table_path = neighbour_dir / f"{table_name}_{dataset}.pkl.xz"
            assert table_path.exists(), table_path

            target_dir = neighbour_dir / f"organ_{dataset}" / run_dir.parent.name / run_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            table_path.rename(target_dir / f"{table_name}_{dataset}.pkl.xz")

        assert len(list(neighbour_dir.glob("*pkl.xz"))) == 0
        assert all(
            (neighbour_dir / f"organ_{d}" / run_dir.parent.name / run_dir.name / f"{table_name}_{d}.pkl.xz").exists()
            for d in simulated_datasets
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Computes all tables for the context problem with the baseline network. For masks_isolation and the glove"
            " runs, only test tables will be computed. For the simulated datasets, also validation datasets will be"
            " computed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--recalculate",
        default=False,
        action="store_true",
        help="Always compute the tables for every dataset, even if they already exist (overwrites existing files).",
    )
    args = parser.parse_args()

    # The main difference between the baseline and the context networks is that we store the context tables for the baseline network at a different location (settings.results_dir / "neighbour_analysis") since we do not want to change the existing models
    # Additionally, we also compute the validation tables for the baseline network, but not for the context networks (as this is done automatically during training)
    # This is why we cannot use the same script for both
    runs_main = [
        HTCModel.find_pretrained_run(
            "image", f"{settings_seg.model_comparison_timestamp}_generated_default_model_comparison"
        ),
        HTCModel.find_pretrained_run(
            "image", f"{settings_seg.model_comparison_timestamp}_generated_default_rgb_model_comparison"
        ),
    ]

    runs_other_granularities = []
    df_runs = collect_comparison_runs(settings_seg.model_comparison_timestamp)
    for _, row in df_runs.iterrows():
        if row["model"] == "image":
            continue
        elif row["model"] == "superpixel_classification":
            runs_other_granularities.append(
                HTCModel.find_pretrained_run(
                    row["model"], settings_context.superpixel_classification_timestamp + "_default"
                )
            )
            runs_other_granularities.append(
                HTCModel.find_pretrained_run(
                    row["model"], settings_context.superpixel_classification_timestamp + "_default_rgb"
                )
            )
        else:
            runs_other_granularities.append(HTCModel.find_pretrained_run(row["model"], row["run_hsi"]))
            runs_other_granularities.append(HTCModel.find_pretrained_run(row["model"], row["run_rgb"]))

    runs_glove = list(settings_context.glove_runs_granularities.values()) + list(
        settings_context.glove_runs_granularities_rgb.values()
    )

    compute_context_tables(runs_main, "validation_table", args.recalculate)
    compute_context_tables(runs_main, "test_table", args.recalculate)
    compute_context_tables(runs_other_granularities, "test_table", args.recalculate)
    compute_glove_test_tables(runs_glove, args.recalculate)
