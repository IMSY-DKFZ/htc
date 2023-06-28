# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np

from htc.model_processing.ImageConsumer import ImageConsumer
from htc.model_processing.Runner import Runner
from htc.model_processing.TestPredictor import TestPredictor
from htc.settings import settings
from htc.utils.blosc_compression import compress_file
from htc.utils.Config import Config
from htc.utils.visualization import compress_html, prediction_figure_html


class InferenceConsumer(ImageConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_dir == self.run_dir:
            self.target_dir = self.target_dir / "predictions"
        else:
            self.target_dir = self.target_dir / self.run_dir.parent.name / self.run_dir.name
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.config.save_config(self.target_dir / "config.json")

    def handle_image_data(self, image_data: dict) -> None:
        path = image_data["path"]

        confidence = np.max(image_data["predictions"], axis=0)
        predictions = np.argmax(image_data["predictions"], axis=0)
        predictions_save = predictions if self.predictions_type == "labels" else image_data["predictions"]
        compress_file(self.target_dir / f"{path.image_name()}.blosc", predictions_save)

        html_prediction = prediction_figure_html(predictions, confidence, path, self.config)
        html = f"""
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Results for image {path.image_name()}</title>
    </head>
    <body>
        {html_prediction}
    </body>
</html>"""
        compress_html(self.target_dir / f"{path.image_name()}.html", html)


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
    runner.add_argument("--annotation-name")
    runner.add_argument("--spec")
    runner.add_argument("--spec-fold")
    runner.add_argument("--spec-split")
    runner.add_argument("--predictions-type", type=str, choices=["softmax", "labels"], default="labels")

    paths = runner.paths
    assert paths is not None, "Either --input-dir or --spec must be provided"
    settings.log.info(f"Compute the prediction for {len(paths)} images")

    config = Config(runner.run_dir / "config.json")

    # This is a general script which should work for arbitrary images so we might not have access to the intermediate files
    # Hence, we compute the L1 normalization on the fly
    if config["input/preprocessing"] == "L1":
        config["input/preprocessing"] = None
        config["input/normalization"] = "L1"

    runner.start(
        partial(TestPredictor, paths=paths, config=config),
        partial(InferenceConsumer),
    )
