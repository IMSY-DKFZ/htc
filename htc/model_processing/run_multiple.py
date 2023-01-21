# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re
import subprocess
import sys
from pathlib import Path

from htc.settings import settings
from htc.utils.helper_functions import get_valid_run_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This is a helper script which can be used to run one of the producer-consumer models in this folder"
            " automatically for multiple runs (e.g. all model comparison runs)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--filter", required=True, type=str, help="Regex to filter specific run dirs.")
    parser.add_argument(
        "--scripts",
        required=True,
        type=str,
        metavar="N",
        nargs="+",
        help="Paths to the scripts to execute (either absolute or relative to the model_processing folder).",
    )
    parser.add_argument(
        "--store-predictions",
        default=False,
        action="store_true",
        help=(
            "Store predictions (<image_name>.npy file with softmax predictions). If a script (or another script) is run"
            " on the same run directory again and the --use-predictions switch is set, then the precalculated"
            " predictions are used per default."
        ),
    )
    parser.add_argument(
        "--use-predictions", default=False, action="store_true", help="Use existing predictions if they already exist."
    )
    parser.add_argument(
        "--set-type",
        default=["test", "validation"],
        metavar="N",
        nargs="+",
        help="Set type to create the predictions from.",
    )
    parser.add_argument(
        "--num-consumers",
        type=int,
        default=None,
        help=(
            "Number of consumers/processes to spawn which work on the predicted images. Defaults to cpu_count() - 1 to"
            " have at least one free CPU for the inference. Note that sometimes less is more and using fewer consumers"
            " may speed up the program time."
        ),
    )

    args = parser.parse_args()

    run_dirs = get_valid_run_dirs()
    if args.filter is not None:
        run_dirs = [r for r in run_dirs if re.search(args.filter, str(r)) is not None]

    if len(run_dirs) == 0:
        settings.log.warning("No run dirs left after filtering")

    for run_dir in run_dirs:
        for script in args.scripts:
            script = Path(script)
            if not script.exists():
                script = settings.htc_package_dir / "model_processing" / script
            assert script.exists(), f"Cannot find the script file {script}"

            for set_type in args.set_type:  # Per default, we run the script on both, the validation and the test set
                command = f'{sys.executable} {script} --model {run_dir.parent.name} --run-folder "{run_dir.name}"'
                if args.store_predictions:
                    command += " --store-predictions"
                if args.use_predictions:
                    command += " --use-predictions"
                if set_type == "test":
                    command += " --test"
                if args.num_consumers is not None:
                    command += f" --num-consumers {args.num_consumers}"

                # It is necessary to run a new Python process for each run since otherwise this would lead to issues with multiprocessing
                settings.log.info(f"Running the command {command}")
                res = subprocess.run(command, shell=True)
                if res.returncode != 0:
                    settings.log.error(
                        f"The process which worked on the run {run_dir} did not succeed successfully. Please check the"
                        " error messages"
                    )
