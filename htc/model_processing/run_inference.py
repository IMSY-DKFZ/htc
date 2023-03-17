# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from functools import partial
from pathlib import Path

import numpy as np

from htc.model_processing.ImageConsumer import ImageConsumer
from htc.model_processing.Runner import Runner
from htc.model_processing.TestPredictor import TestPredictor
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import compress_file
from htc.utils.Config import Config
from htc.utils.visualization import compress_html, prediction_figure_html


class InferenceConsumer(ImageConsumer):
    def __init__(self, *args, paths: list[DataPath], **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.target_folder = self.output_dir / self.run_dir.name
        else:
            self.target_folder = settings.results_dir / "predictions" / self.run_dir.name
        self.target_folder.mkdir(parents=True, exist_ok=True)

        self.config.save_config(self.target_folder / "config.json")

        # We need a custom mapping from image_name to paths since this script should work with arbitrary images even without unique image names
        self.path_mapping = {p.image_name(): p for p in paths}

    def handle_image_data(self, image_data: dict) -> None:
        confidence = np.max(image_data["predictions"], axis=0)
        predictions = np.argmax(image_data["predictions"], axis=0)
        predictions_save = predictions if self.predictions_type == "labels" else image_data["predictions"]
        compress_file(self.target_folder / f'{image_data["image_name"]}.blosc', predictions_save)

        path = self.path_mapping[image_data["image_name"]]
        html_prediction = prediction_figure_html(predictions, confidence, path, self.config)
        html = f"""
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Results for image {image_data["image_name"]}</title>
    </head>
    <body>
        {html_prediction}
    </body>
</html>"""
        compress_html(self.target_folder / f'{image_data["image_name"]}.html', html)


if __name__ == "__main__":
    runner = Runner(
        description=(
            "General inference based on the TestPredictor. For each image, the softmax predictions and an accompanying"
            " HTML file which visualizes the result will be stored either in your results directory or in the specified"
            " output directory."
        )
    )
    runner.add_argument("--input-dir")
    runner.add_argument("--output-dir")
    runner.add_argument("--predictions-type", type=str, choices=["softmax", "labels"], default="labels")
    runner.add_argument(
        "--annotation-name",
        type=str,
        default=None,
        help="Filter the paths by this annotation name (default is no filtering).",
    )
    runner.add_argument(
        "--specs",
        type=Path,
        default=None,
        help=(
            "Path or name of a data specification file from which paths are collected. This parameter can be used as"
            " alternative to the --input-dir argument."
        ),
    )
    runner.add_argument(
        "--split-name",
        type=str,
        default=None,
        help=(
            "Optional regex selector for one or more splits in the specification file. Please note that the test set is"
            " only used if --split-name is given and contains the word `test`"
        ),
    )

    if runner.args.input_dir is not None:
        assert (
            runner.args.specs is None and runner.args.split_name is None
        ), "--specs and --split-name can only be used if --input-dir is not used"
        input_dir = runner.args.input_dir
        assert input_dir.exists(), "Directory for which inference should be computed does not exist."
        paths = list(DataPath.iterate(input_dir, annotation_name=runner.args.annotation_name))
    else:
        assert runner.args.specs is not None, "--specs must be provided if --input-dir is not provided"
        specs = DataSpecification(runner.args.specs)
        if runner.args.split_name is not None and "test" in runner.args.split_name:
            specs.activate_test_set()
            settings.log.info("Activating the test set of the data specification file")
        paths = specs.paths(runner.args.split_name)

    settings.log.info(f"Compute the prediction for {len(paths)} images")

    config = Config(runner.run_dir / "config.json")

    # This is a general script which should work for arbitrary images so we might not have access to the intermediate files
    # Hence, we compute the L1 normalization on the fly
    if config["input/preprocessing"] == "L1":
        config["input/preprocessing"] = None
        config["input/normalization"] = "L1"

    runner.start(
        partial(TestPredictor, paths=paths, config=config),
        partial(InferenceConsumer, paths=paths),
    )
