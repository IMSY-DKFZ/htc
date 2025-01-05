# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from datetime import datetime

from htc.cluster.utils import cluster_command, run_jobs
from htc.models.data.DataSpecification import DataSpecification
from htc.models.run_generate_configs import generate_configs
from htc.settings import settings
from htc.utils.Config import Config


def generate_seed_variations() -> list[str]:
    timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    jobs = []
    params = {"seed": list(range(5))}
    n_folds = None

    model = "image"
    config_file = settings.models_dir / model / "configs" / "default.json"
    for new_config in generate_configs(model, config_file, params=params, return_config_only=True):
        run_name = f"{timestring}_{new_config['config_name']}"
        rel_config_path = new_config.path_config.relative_to(settings.htc_package_dir)
        Config.from_model_name(rel_config_path, model)  # Make sure we can load the config

        data_specs = DataSpecification.from_config(new_config)
        if n_folds is None:
            n_folds = len(data_specs)
        else:
            assert len(data_specs) == n_folds, "All models must use the same number of folds"

        for fold_name in data_specs.fold_names():
            jobs.append(
                cluster_command(
                    f"--model {model} --config {rel_config_path} --run-folder {run_name} --fold {fold_name}"
                )
            )

    assert len(jobs) == n_folds * len(params["seed"]), "Incorrect number of jobs"
    settings.log.info(f"The following {len(jobs)} jobs are going to be submitted to the cluster:")
    for j in jobs:
        settings.log.info(j)

    return jobs


if __name__ == "__main__":
    jobs = generate_seed_variations()
    run_jobs(jobs)
