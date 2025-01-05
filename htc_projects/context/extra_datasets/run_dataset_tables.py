# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import shutil
from functools import partial

from htc.model_processing.run_tables import ImageTableConsumer
from htc.model_processing.Runner import Runner
from htc.model_processing.TestPredictor import TestPredictor
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc_projects.context.settings_context import settings_context

if __name__ == "__main__":
    # For the context runs:
    # htc dataset_tables --model image --run-folder 2023-02-08_09-40-59_elastic_0.2 --metrics DSC --test --dataset-name masks_isolation
    # For the MIA runs:
    # htc dataset_tables --model image --run-folder 2022-02-03_22-58-44_generated_default_model_comparison --metrics DSC --test --dataset-name masks_isolation --output-dir ~/htc/results_context/neighbour_analysis/masks_isolation/image/2022-02-03_22-58-44_generated_default_model_comparison
    runner = Runner(
        description=(
            "Create a test table for one of the datasets used in the context problem (e.g. masks isolation dataset)."
        )
    )
    runner.add_argument("--test")
    runner.add_argument("--metrics")
    runner.add_argument("--NSD-thresholds")
    runner.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["masks_isolation"],
        help=(
            "The name of the dataset which will also be reflected in the name of the test table (e.g."
            " test_table_masks_isolation)."
        ),
    )
    assert runner.args.test, "The --test argument must be set"

    dataset = settings_context.real_datasets[runner.args.dataset_name]
    paths = []
    for names in dataset.values():
        for name in names:
            paths.append(DataPath.from_image_name(name))

    if not (runner.run_dir / runner.args.config).exists():
        # We need a config in the main folder for this script
        fold_dirs = sorted(runner.run_dir.glob("fold*"))
        shutil.copy2(fold_dirs[0] / "config.json", runner.run_dir / runner.args.config)

    # The label list has changed compared to the original models (e.g. new instruments) so we need to make sure we are using the up-to-date list in order to validate properly
    config = runner.config
    config["label_mapping"] = settings_seg.label_mapping

    runner.start(
        partial(TestPredictor, paths=paths),
        partial(
            ImageTableConsumer,
            test_table_name=f"test_table_{runner.args.dataset_name}",
            config=config,
        ),
    )
