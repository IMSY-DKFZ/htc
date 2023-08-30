# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.common.RunGenerator import RunGenerator
from htc.utils.Config import Config


def glove_adjustment(config: Config, **kwargs) -> str:
    config["input/data_spec"] = "data/pigs_semantic-only_5foldsV2_glove.json"
    return "_glove"


if __name__ == "__main__":
    rg = RunGenerator()

    for name in ["default", "default_rgb"]:
        config = Config.from_model_name(name, "image")
        rg.generate_run(config, [glove_adjustment])

    rg.submit_jobs()
