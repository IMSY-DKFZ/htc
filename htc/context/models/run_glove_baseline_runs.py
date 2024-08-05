# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse

from htc.models.common.RunGenerator import RunGenerator
from htc.settings_seg import settings_seg
from htc.utils.Config import Config


def glove_adjustment(config: Config, **kwargs) -> str:
    config["input/data_spec"] = "data/pigs_semantic-only_5foldsV2_glove.json"
    return "_glove"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Start training runs on the cluster for the glove baseline models (MIA runs with the glove data"
            " specification)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=settings_seg.model_names,
        choices=settings_seg.model_names,
        nargs="+",
        type=str,
        help="One or more model names to generate runs for (each time with RGB and HSI).",
    )
    args = parser.parse_args()

    rg = RunGenerator()

    for model in args.model:
        for name in ["default", "default_rgb"]:
            config = Config.from_model_name(name, model)
            rg.generate_run(config, [glove_adjustment], model_name=model)

            if model == "patch":
                config = Config.from_model_name(name.replace("default", "default_64"), model)
                rg.generate_run(config, [glove_adjustment], model_name=model)

    rg.submit_jobs()
