# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re
import sys
from pathlib import Path

import htc.model_processing.run_tables as run_tables
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.general import subprocess_run


def compute_physiological_scores(run_dir: Path, recalculate: bool) -> None:
    match = re.search(r"(pig|rat|human)_nested-(\d+)-(\d+)$", run_dir.name)
    assert match is not None, f"Could not infer species from run folder name {run_dir.name}"
    species = match.group(1)
    nested_index = int(match.group(2))
    max_nested_index = int(match.group(3))

    tables_args = []

    # For the main species, we compute the test results of the current nested fold as usual
    config_main = Config(run_dir / "config.json")
    tables_args.append((run_dir.name, config_main["input/data_spec"], "test", f"test_table_{species}"))

    if nested_index == 0:
        # For the other species, we need to compute the test results by ensembling over all nested folds (we need to load all networks)
        # Since we use the other nested fold networks here as well, we store the table only for the first nested fold (similar to the physiological scores)
        run_dir_generic = run_dir.name.replace(
            f"_nested-{nested_index}-{max_nested_index}", f"_nested-*-{max_nested_index}"
        )
        other_species = sorted({"pig", "rat", "human"} - {species})
        for s in other_species:
            config = Config(f"species/configs/baseline_{s}.json")
            tables_args.append((run_dir_generic, config["input/data_spec"], "train|test", f"test_table_{s}"))

    returncodes = []
    for run_folder, spec, spec_split, test_table_name in tables_args:
        if not recalculate and (run_dir / f"{test_table_name}.pkl.xz").exists():
            continue

        settings.log.info(f"Computing the {test_table_name} table for the run {run_dir.name}")
        res = subprocess_run([
            sys.executable,
            run_tables.__file__,
            "--model",
            run_dir.parent.name,
            "--run-folder",
            run_folder,
            "--test",
            "--metrics",
            "DSC",
            "NSD",
            "CM",
            "--spec",
            spec,
            "--spec-split",
            spec_split,
            "--test-table-name",
            test_table_name,
        ])
        returncodes.append(res.returncode)

    assert all(returncode == 0 for returncode in returncodes), (
        f"Failed to compute test tables for the run {run_dir.name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute test scores for the physiological images of all species datasets (test sets of pig, rat and human"
            " datasets)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="The timestamp to select runs for which test scores should be calculated.",
    )
    parser.add_argument(
        "--recalculate",
        default=False,
        action="store_true",
        help="Always compute the tables, even if they already exist (overwrites existing files).",
    )
    args = parser.parse_args()

    run_dirs = []
    for model_dir in sorted(settings.training_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for run_dir in sorted(model_dir.glob(f"{args.timestamp}*")):
            compute_physiological_scores(run_dir, args.recalculate)
