# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from datetime import datetime

from htc.models.common.utils import cluster_command, run_jobs
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.utils.Config import Config


class RunGenerator:
    def __init__(self) -> None:
        self.timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.jobs = []

    def generate_run(
        self,
        config: Config,
        config_adjustments: list[Callable[[Config], str]],
        model_name: str = "image",
        memory: str = "10.7G",
        single_submit: bool = False,
        **kwargs,
    ) -> None:
        """
        Generate a training run based on config adjustments (callbacks) from an existing config.

        Args:
            config: Configuration file to use as basis.
            config_adjustments: List of callables to alter the config. Each callable should take a config as input, modify the config accordingly and return a string that is appended to the run name.
            model_name: Name of the model.
            memory: Memory to request for the cluster job.
            single_submit: If True, only one job is submitted to the cluster. Otherwise, one job per fold is submitted.
            kwargs: Additional keyword arguments that are passed to the config adjustment callbacks.
        """
        assert "config_name" in config, (
            "Only configuration objects which were loaded from a file can be used for run generation (because only then"
            " the config_name is set)"
        )
        base_name = config["config_name"]

        for adjuster in config_adjustments:
            base_name += adjuster(config, **kwargs)

        # Disable logging for cluster runs
        config["trainer_kwargs/enable_progress_bar"] = False

        # Store config
        config.save_config(config.path_config.parent / f"generated_{base_name}.json")

        # Run properties
        run_name = f"{self.timestring}_{base_name}"
        config_relative_path = config.path_config.relative_to(settings.htc_package_dir)

        # Generate submit command
        if single_submit:
            jobs = [
                cluster_command(
                    f'--model {model_name} --config "{config_relative_path}" --run-folder "{run_name}"',
                    memory=memory,
                ),
            ]
        else:
            specs = DataSpecification.from_config(config)
            jobs = []
            for fold_name in specs.fold_names():
                jobs.append(
                    cluster_command(
                        f'--model {model_name} --config "{config_relative_path}" --run-folder "{run_name}" --fold'
                        f" {fold_name}",
                        memory=memory,
                    )
                )

            assert len(jobs) == len(specs)

        assert len(jobs) > 0, "No jobs generated"
        self.jobs += jobs

    def submit_jobs(self) -> None:
        run_jobs(self.jobs)
