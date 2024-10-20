# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import re
from datetime import datetime
from pathlib import Path

from htc.cluster.utils import cluster_command, run_jobs
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.Config import Config


def find_relevant_configs(config_dir: Path) -> list[Path]:
    config_files = []

    for config_file in sorted(config_dir.glob("default*")):
        if re.search(r"^default(?:_64)?(?:_parameters|_rgb)?\.json$", config_file.name) is not None:
            config_files.append(config_file)

    return config_files


def generate_model_comparison_runs() -> list[str]:
    timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    jobs = []
    n_folds = None

    for model in settings_seg.model_names:
        config_dir = settings.models_dir / model / "configs"
        for config_file in find_relevant_configs(config_dir):
            config = Config(config_file)
            new_config = copy.copy(config)

            # Disable logging for cluster runs
            new_config["trainer_kwargs/enable_progress_bar"] = False

            # Store the config in the configs folder
            config_name = f'generated_{new_config["config_name"]}_model_comparison'
            new_config["config_name"] = config_name

            filename = config_name + ".json"
            new_config.save_config(config_dir / filename)
            run_name = f'{timestring}_{new_config["config_name"]}'

            data_specs = DataSpecification.from_config(new_config)
            if n_folds is None:
                n_folds = len(data_specs)
            else:
                assert len(data_specs) == n_folds, "All models must use the same number of folds"

            for fold_name in data_specs.fold_names():
                jobs.append(
                    cluster_command(f"--model {model} --config {filename} --run-folder {run_name} --fold {fold_name}")
                )

    assert len(jobs) == settings_seg.n_algorithms * n_folds, "Incorrect number of jobs"
    settings.log.info(f"The following {len(jobs)} jobs are going to be submitted to the cluster:")
    for j in jobs:
        settings.log.info(j)

    return jobs


if __name__ == "__main__":
    jobs = generate_model_comparison_runs()
    run_jobs(jobs)
