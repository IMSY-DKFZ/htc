# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import copy
import itertools
from datetime import datetime
from pathlib import Path

from htc.models.common.utils import cluster_command
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.utils.Config import Config


def generate_configs(
    model_name: str,
    config_name_or_path: str,
    single_submit: bool = False,
    params: dict = None,
    test: bool = False,
    return_config_only: bool = False,
    tuning: bool = False,
    memory: str = "10.7G",
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = Config.load_config(config_name_or_path, model_name)

    if params is None:
        params = {
            "seed": [settings.default_seed],
            # 'model/architecture_name': ["Model3D2DSeg", "DynUNet"],
            # 'dataloader_kwargs/batch_size': [8],
            # 'dataloader_kwargs/num_workers': [8],
        }

    bashstring = (
        '#!/bin/bash\n\nscript_path=$(realpath "$BASH_SOURCE")\nscript_path=$(dirname "$script_path")\n\n# Execute'
        " everything in the root of the repository\ncd $script_path/../../../.. || exit\n\n# Make environment variables"
        ' available\nsource .env\n\n# Run jobs\nssh $DKFZ_USERID@bsub01.lsf.dkfz.de <<"BASH"\n'
    )

    configs = []
    for parameters in itertools.product(*params.values()):
        new_config = copy.copy(config)

        # Disable logging for cluster runs
        new_config["trainer_kwargs/enable_progress_bar"] = False

        # Adjust template config file
        filename_parts = []
        for name, parameter in zip(params.keys(), parameters):
            new_config[f"{name}"] = parameter

            name_short = name.split("/")[-1]

            # Avoid arrays in the filename
            if type(parameter) == list or type(parameter) == tuple:
                parameter = "+".join([str(p) for p in parameter])
            elif type(parameter) == dict:
                parameter = "+".join([str(v) for v in parameter.values()])

            filename_parts.append(f"{name_short}={parameter}")

        n_gpus = new_config.get("trainer_kwargs/devices", 1)

        # Use all available CPUs
        new_config["dataloader_kwargs/num_workers"] = new_config["dataloader_kwargs/num_workers"] * n_gpus

        # Store the config in the configs folder
        new_config["config_name"] = "generated_" + config["config_name"] + "_" + ",".join(filename_parts)
        new_config["config_name"] = new_config["config_name"].replace(
            "<", "LT"
        )  # Avoid special symbols which may break the cluster
        filename = new_config["config_name"] + ".json"
        config_dir = config.path_config.parent
        new_config.save_config(config_dir / filename)

        run_name = datetime.now().strftime(f'{timestamp}_{new_config["config_name"]}')
        config_rel_path = Path(config_name_or_path).parent / filename

        if single_submit:
            bashstring += cluster_command(
                f'--model {model_name} --config "{config_rel_path}"', n_gpus=n_gpus, memory=memory
            )
            if test:
                bashstring += " --test"
            if tuning:
                bashstring += " --tuning"
            bashstring += "\n"
        else:
            data_specs = DataSpecification.from_config(config)
            for fold_name in data_specs.fold_names():
                bashstring += cluster_command(
                    f'--model {model_name} --config "{config_rel_path}" --run-folder "{run_name}" --fold {fold_name}',
                    n_gpus=n_gpus,
                    memory=memory,
                )
                if test:
                    bashstring += " --test"
                if tuning:
                    bashstring += " --tuning"
                bashstring += "\n"
        bashstring += "\n"

        configs.append(new_config)

    if return_config_only:
        return configs
    else:
        bashstring += "BASH\n"
        bashsavepath = config_dir / f'submit_jobs_{config["config_name"]}.sh'
        with bashsavepath.open("wb") as f:
            f.write(bashstring.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate configs for a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train (e.g. image or pixel).")
    parser.add_argument(
        "--config",
        default="default.json",
        help=(
            "Name of the configuration file to use (either absolute, relative to the current working directory or"
            " relative to the models config folder)."
        ),
    )
    parser.add_argument(
        "--single-submit", default=False, action="store_true", help="Use only one cluster submission for all folds."
    )
    parser.add_argument(
        "--test", default=False, action="store_true", help="To add the testing parameter to run commands."
    )
    parser.add_argument(
        "--tuning",
        default=False,
        action="store_true",
        help="To add the hyper parameter tuning parameter to run commands.",
    )
    parser.add_argument("--memory", type=str, default="10.7G", help="GPU memory requirements for the cluster.")

    args = parser.parse_args()

    generate_configs(
        args.model, args.config, args.single_submit, test=args.test, tuning=args.tuning, memory=args.memory
    )
