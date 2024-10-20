# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
import sys

import htc.model_processing.run_tables as run_tables
from htc.model_processing.Runner import Runner
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.general import subprocess_run

if __name__ == "__main__":
    runner = Runner(description="Compute test scores for the kidney dataset.")
    config = Config(runner.run_dir / "config.json")

    # Retrieve the name of the spec from the kidney pigs which are listed in the base name of the projection transform
    spec = None
    for t in config["input/transforms_gpu"]:
        if t["class"].endswith("ProjectionTransform"):
            match = re.search(r"kidney=(P\d+(?:,P\d+)*)", t["base_name"])
            if match is not None:
                kidney_spec = f"kidney_projection_train={match.group(1)}"
                for spec_file in sorted((settings.htc_projects_dir / "data").iterdir()):
                    if kidney_spec == spec_file.stem:
                        spec = spec_file
                break

    if spec is not None:
        # Training case (e.g. pig2pig)
        settings.log.info(f"Using train/test configuration from spec {spec}")
        for test_table_name, spec_split in [("test_table_kidney", "test"), ("test_table_kidney-train", "train")]:
            subprocess_run([
                sys.executable,
                run_tables.__file__,
                "--model",
                runner.run_dir.parent.name,
                "--run-folder",
                runner.run_dir.name,
                "--test",
                "--gpu-only",
                "--metrics",
                "DSC",
                "CM",
                "--spec",
                spec,
                "--spec-fold",
                "fold_all",
                "--spec-split",
                spec_split,
                "--test-table-name",
                test_table_name,
            ])
    else:
        # Testing case (e.g. rat2pig)
        spec = "kidney_projection_train=P091,P095,P097,P098.json"
        settings.log.info(f"No spec found, using all train and test images found in the default spec {spec}")
        subprocess_run([
            sys.executable,
            run_tables.__file__,
            "--model",
            runner.run_dir.parent.name,
            "--run-folder",
            runner.run_dir.name,
            "--test",
            "--gpu-only",
            "--metrics",
            "DSC",
            "CM",
            "--spec",
            spec,
            "--spec-fold",
            "fold_all",
            "--spec-split",
            "train|test",
            "--test-table-name",
            "test_table_kidney",
        ])
