# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re
import sys
from pathlib import Path

import htc.context.extra_datasets.run_dataset_tables as run_dataset_tables
import htc.context.manipulated_datasets.run_context_evaluation_table as run_context_evaluation_table
import htc.model_processing.run_tables as run_tables
from htc.context.settings_context import settings_context
from htc.settings import settings
from htc.utils.general import subprocess_run


def compute_test_tables(runs: list[Path], recalculate: bool = False) -> None:
    """
    Compute missing test tables for all context datasets except glove runs for every network.

    Args:
        runs: List of training run directories.
        recalculate: If True, compute the test table even if it already exists (overwrites existing tables). If False, existing test tables are skipped.
    """
    for run_dir in runs:
        assert run_dir.exists(), f"The run directory {run_dir} does not exist. Cannot compute the test tables"

    for run_dir in runs:
        # Test table on the default semantic dataset
        test_table = run_dir / "test_table.pkl.xz"
        if not test_table.exists() or recalculate:
            settings.log.info(f"Computing test table for {run_dir}")
            res = subprocess_run(
                [
                    sys.executable,
                    run_tables.__file__,
                    "--model",
                    run_dir.parent.name,
                    "--run-folder",
                    run_dir.name,
                    "--metrics",
                    "DSC",
                    "ASD",
                    "NSD",
                    "--test",
                ]
            )
            assert res.returncode == 0, f"Computation of the test table for the run folder {run_dir} was not successful"
            assert test_table.exists()

        test_table_real = run_dir / "test_table_masks_isolation.pkl.xz"
        if not test_table_real.exists() or recalculate:
            settings.log.info(f"Computing masks isolation test table for {run_dir}")
            res = subprocess_run(
                [
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
                ]
            )
            assert (
                res.returncode == 0
            ), f"Computation of the masks isolation test table for the run folder {run_dir} was not successful"
            assert test_table_real.exists()

        # Test tables on the simulated datasets
        simulated_datasets = ["isolation_0", "isolation_cloth", "removal_0", "removal_cloth"]
        missing = [d for d in simulated_datasets if not (run_dir / f"test_table_{d}.pkl.xz").exists() or recalculate]
        if len(missing) > 0:
            settings.log.info(f"Computing test results on the simulated datasets {missing} for {run_dir}")
            res = subprocess_run(
                [
                    sys.executable,
                    run_context_evaluation_table.__file__,
                    "--model",
                    run_dir.parent.name,
                    "--run-folder",
                    run_dir.name,
                    "--test",
                    "--transformation-name",
                    *missing,
                ]
            )
            assert (
                res.returncode == 0
            ), f"Computation of the context test table for the run folder {run_dir} was not successful"

        assert all((run_dir / f"test_table_{d}.pkl.xz").exists() for d in simulated_datasets)


def compute_glove_test_tables(runs: list[Path], recalculate: bool = False) -> None:
    """
    Compute missing test tables for all glove runs (this basically runs the run_tables script).

    This function works for both, the baseline and the context networks.

    Args:
        runs: List of glove training run directories.
        recalculate: If True, compute the test table even if it already exists (overwrites existing tables). If False, existing test tables are skipped.
    """
    for run_dir in runs:
        assert run_dir.exists(), f"The run directory {run_dir} does not exist. Cannot compute the glove test tables"

    for run_dir in runs:
        # Test table on the default semantic dataset (contains both glove and no-glove)
        test_table = run_dir / "test_table.pkl.xz"
        if not test_table.exists() or recalculate:
            settings.log.info(f"Computing glove test results for {run_dir}")
            res = subprocess_run(
                [
                    sys.executable,
                    run_tables.__file__,
                    "--model",
                    run_dir.parent.name,
                    "--run-folder",
                    run_dir.name,
                    "--metrics",
                    "DSC",
                    "ASD",
                    "NSD",
                    "--test",
                ]
            )
            assert res.returncode == 0, f"Computation of the test table for the run folder {run_dir} was not successful"
            assert test_table.exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Computes all test tables for the context problem. If no argument is given, computes test tables for the"
            " best transformation runs"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--filter",
        default=None,
        required=False,
        type=str,
        help="Regex to filter specific run dirs.",
    )
    parser.add_argument(
        "--recalculate",
        default=False,
        action="store_true",
        help=(
            "Always compute the test tables for every dataset, even if they already exist (overwrites existing files)."
        ),
    )
    args = parser.parse_args()

    runs = []
    runs_glove = []
    if args.filter is None:
        for name, run_dir in settings_context.best_transform_runs.items():
            runs.append(run_dir)
            runs_glove.append(settings_context.glove_runs[name])

        # We also need the main RGB runs
        runs.append(settings_context.best_transform_runs_rgb["organ_transplantation"])
        runs_glove.append(settings_context.glove_runs_rgb["organ_transplantation"])
    else:
        for r in sorted(settings.training_dir.glob("*/*")):
            if re.search(args.filter, str(r)) is not None:
                if "glove" in str(r):
                    runs_glove.append(r)
                else:
                    runs.append(r)

    settings.log.info(f"runs: {[r.name for r in runs]}")
    settings.log.info(f"glove runs: {[r.name for r in runs_glove]}")
    compute_test_tables(runs, args.recalculate)
    compute_glove_test_tables(runs_glove, args.recalculate)
