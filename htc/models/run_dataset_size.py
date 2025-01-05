# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from datetime import datetime

from htc.cluster.utils import cluster_command, run_jobs
from htc.models.data.DataSpecification import DataSpecification
from htc.models.data.run_size_dataset import label_mapping_dataset_size
from htc.models.run_model_comparison import find_relevant_configs
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.Config import Config


def generate_dataset_size_runs() -> list[str]:
    label_mapping = label_mapping_dataset_size()
    timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    jobs = []

    for model in settings_seg.model_names:
        config_dir = settings.models_dir / model / "configs"
        for config_file in find_relevant_configs(config_dir):
            if "parameters" in config_file.name or "rgb" in config_file.name:
                continue

            config = Config(config_file)
            new_config = copy.copy(config)

            # Changes compared to the default config
            new_config["input/data_spec"] = "data/pigs_semantic-only_dataset-size_repetitions=5V2.json"
            new_config["label_mapping"] = label_mapping

            # Disable logging for cluster runs
            new_config["trainer_kwargs/enable_progress_bar"] = False

            # Store the config in the configs folder
            config_name = f"generated_{new_config['config_name']}_dataset_size"
            new_config["config_name"] = config_name

            filename = config_name + ".json"
            new_config.save_config(config_dir / filename)
            run_name = f"{timestring}_{new_config['config_name']}"

            data_specs = DataSpecification.from_config(new_config)
            for fold_name in data_specs.fold_names():
                jobs.append(
                    cluster_command(f"--model {model} --config {filename} --run-folder {run_name} --fold {fold_name}")
                )

    assert len(jobs) == len(settings_seg.model_colors) * len(data_specs.fold_names()), "Incorrect number of jobs"
    settings.log.info(f"The following {len(jobs)} jobs are going to be submitted to the cluster:")
    for j in jobs:
        settings.log.info(j)

    return jobs


if __name__ == "__main__":
    jobs = generate_dataset_size_runs()
    run_jobs(jobs)
