# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from htc.context.settings_context import settings_context
from htc.models.common.utils import cluster_command, run_jobs
from htc.settings import settings
from htc.utils.Config import Config


def transform_runs(
    name: str, transform: dict[str, Any], config: Config, pvalues: list[float] = None, rgb: bool = False
) -> list[str]:
    """
    Generate cluster training runs for the given transformation.

    Args:
        name: Name of the transformation.
        transform: Dictionary with the transformation parameters which will be appended to `input/transforms_gpu`.
        config: The default configuration object to use.
        pvalues: Probability (`p`) values which should be generated. If None, a grid search [0.2, 0.4, 0.6, 0.8, 1.0] will be performed.
        rgb: If True, RGB data will be used instead of HSI data.

    Returns: List of job commands for the cluster (can be passed to `run_jobs()`).
    """
    if pvalues is None:
        pvalues = [0.2, 0.4, 0.6, 0.8, 1]

    config_dir = settings.htc_package_dir / "context/models/configs"
    timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Prepend the config name to the run folder name
    base_name = f"{config['config_name']}_{name}"
    jobs = []

    for p in pvalues:
        new_config = copy.copy(config)

        transform["p"] = p
        new_config["input/transforms_gpu"].append(transform)

        # Disable logging for cluster runs
        new_config["trainer_kwargs/enable_progress_bar"] = False

        if rgb:
            # Load RGB instead of HSI data
            new_config["input/preprocessing"] = None
            new_config["input/n_channels"] = 3
            base_name += "_rgb"

        # Store config
        config_name = f"generated_{base_name}_{p}"
        new_config["config_name"] = config_name
        filename = config_name + ".json"
        new_config.save_config(config_dir / filename)
        run_name = f"{timestring}_{base_name}_{p}"

        # Single submit because the jobs don't take so long
        jobs.append(
            cluster_command(f'--model image --config "context/models/configs/{filename}" --run-folder "{run_name}"'),
        )

    assert len(jobs) == len(pvalues), "Incorrect number of jobs"
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Start runs for the different transformations. Can either be used for a grid search ([0.2, 0.4, 0.6, 0.8,"
            " 1]) or to start new training runs with the best transformation runs"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="context/models/configs/context.json",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Name of the configuration files to use as baseline (either absolute, relative to the current working"
            " directory or relative to the models config folder)."
        ),
    )
    parser.add_argument(
        "--best",
        default=False,
        action="store_true",
        help=(
            "If set, do not perform a grid search but use only the known best p value for the respective transformation"
            " (via settings_context.best_transform_runs)."
        ),
    )
    parser.add_argument(
        "--include-rgb",
        default=False,
        action="store_true",
        help=(
            "If set, will include RGB runs (only for the organ_transplantation transformation and only together with"
            " the --best flag)"
        ),
    )
    args = parser.parse_args()

    jobs = []
    for config_path in args.config:
        config = Config(config_path)

        for name, trans in settings_context.transforms.items():
            if args.best:
                # Submit only the job with the best known p value
                run_folder = settings_context.best_transform_runs[name].name
                match = re.search(r"\d+(?:\.\d+)?$", run_folder)
                assert match is not None, f"Could not extract probability from run folder {run_folder}"
                probability = float(match.group(0))

                jobs += transform_runs(name, trans, config, pvalues=[probability])

                if name == "organ_transplantation" and args.include_rgb:
                    jobs += transform_runs(name, trans, config, pvalues=[probability], rgb=True)
            else:
                # Submit jobs with different p values for the different transforms
                jobs += transform_runs(name, trans, config)

    run_jobs(jobs)
