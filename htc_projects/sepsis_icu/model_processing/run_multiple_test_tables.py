# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import sys

import htc_projects.sepsis_icu.model_processing.run_sepsis_test_table as run_sepsis_test_table
from htc.settings import settings
from htc.utils.general import subprocess_run
from htc.utils.parallel import p_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all sepsis and survival test tables", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="The timestamp to select runs for which test tables will be computed.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        required=False,
        help=(
            "The name of the test table to be computed. For example, for the sepsis bias models, pass"
            " 'test_table_sepsis_icu' to test on the sepsis ICU dataset, pass 'test_table_first' to test on the first"
            " sepsis timepoints of the sepsis bias dataset, and pass 'test_table_all' to test on the full sepsis bias"
            " dataset."
        ),
    )
    args = parser.parse_args()

    commands = []

    if args.table_name is None:
        table_name = "test_table_new"
    else:
        table_name = args.table_name

    for model_dir in sorted(settings.training_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for run_dir in sorted(model_dir.glob(f"{args.timestamp}*")):
            test_table_path = run_dir / f"{table_name}.pkl.xz"
            if not test_table_path.exists():
                if table_name == "test_table_unclear_sepsis_status" and "survival" in run_dir.name:
                    continue

                current_command = [
                    sys.executable,
                    run_sepsis_test_table.__file__,
                    "--model",
                    model_dir.name,
                    "--run-folder",
                    run_dir.name,
                    "--table-name",
                    table_name,
                ]

                if table_name == "test_table_unclear_sepsis_status":
                    if "finger" in run_dir.name:
                        paths_variable = "htc_projects.sepsis_icu.settings_sepsis_icu>test_unclear_paths_finger"
                    elif "palm" in run_dir.name:
                        paths_variable = "htc_projects.sepsis_icu.settings_sepsis_icu>test_unclear_paths_palm"

                    current_command.append("--paths-variable")
                    current_command.append(paths_variable)
                elif table_name == "test_table_recalibrated":
                    current_command.append("--config")
                    current_command.append("config_recalibrated.json")
                elif table_name == "test_table_sepsis_icu":
                    current_command.append("--config")
                    current_command.append("config_sepsis_icu.json")
                    current_command.append("--spec-split")
                    current_command.append("train|test")
                elif table_name == "test_table_first":
                    current_command.append("--spec-split")
                    current_command.append("test_first")
                elif table_name == "test_table_all":
                    current_command.append("--spec-split")
                    current_command.append("test_all")

                commands.append(current_command)

    res = p_map(subprocess_run, commands, num_cpus=6, use_threads=True)
    if not all(r.returncode == 0 for r in res):
        for r, c in zip(res, commands, strict=True):
            if r.returncode != 0:
                print("The following run failed:")
                print(f"Command: {c}")

        raise ValueError("Some runs failed")
