# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from htc.model_processing.ImageConsumer import ImageConsumer
from htc.model_processing.Runner import Runner
from htc.model_processing.TestLeaveOneOutPredictor import TestLeaveOneOutPredictor
from htc.model_processing.TestPredictor import TestPredictor
from htc.model_processing.ValidationPredictor import ValidationPredictor
from htc.tivita.DataPath import DataPath
from htc.utils.LabelMapping import LabelMapping
from htc.utils.visualization import (
    compress_html,
    create_confusion_figure,
    create_image_scores_figure,
    prediction_figure_html,
)


class ImageFigureConsumer(ImageConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dir = self.run_dir / "prediction_figures"
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def handle_image_data(self, image_data: dict) -> None:
        predictions = image_data["predictions"]
        confidence = np.max(predictions, axis=0)
        predictions = np.argmax(predictions, axis=0)

        image_name = image_data["image_name"]
        path = DataPath.from_image_name(image_name)

        # Find the appropriate table where the evaluation results are stored for this image
        df_val = pd.read_pickle(self.run_dir / "validation_table.pkl.xz").query("dataset_index == 0")
        if self.test:
            df = pd.read_pickle(self.run_dir / "test_table.pkl.xz")
            df = df.query("image_name == @image_name")
            assert len(df) == 1, "There is more than one result for the image"
            df = df.iloc[0]
        else:
            df = df_val.query(f'epoch_index == best_epoch_index and image_name == "{image_name}"')
            assert len(df) == 1, "There is more than one result for the image"
            df = df.iloc[0]

        # Some metric data to display for the image
        dice = df["dice_metric_image"].item()
        title = f": dice={dice:0.2f}"
        if "surface_distance_metric_image" in df:
            surface = df["surface_distance_metric_image"].item()
            title += f", surface={surface:0.2f}"

        # Create figures
        html_prediction = prediction_figure_html(predictions, confidence, path, self.config, title_suffix=title)

        mapping = LabelMapping.from_config(self.config)
        label_names = [mapping.index_to_name(i) for i in df["used_labels"]]
        fig_image_scores = create_image_scores_figure(label_names, df["dice_metric"])

        if "confusion_matrix" in df:
            mapping = LabelMapping.from_config(self.config)
            fig_confusion = create_confusion_figure(df["confusion_matrix"], labels=mapping.label_names())
        else:
            fig_confusion = None

        # Combine all figures in one html file
        html = f"""
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Results for image {image_data["image_name"]}</title>
    </head>
    <body>
        {html_prediction}
        {fig_image_scores.to_html(full_html=False, include_plotlyjs='cdn', div_id='dice_scores')}
        {fig_confusion.to_html(full_html=False, include_plotlyjs='cdn', div_id='confusion_matrix') if fig_confusion is not None else ''}
    </body>
</html>"""

        target_file = self.target_dir / f'{image_data["image_name"]}_dice={dice:.02f}.html'
        compress_html(target_file, html)


if __name__ == "__main__":
    runner = Runner(
        description="Create prediction figures (interactive HTML files) for every image in the validation or test set."
    )
    runner.add_argument("--test")
    runner.add_argument("--test-looc")

    if runner.args.test:
        TestClass = TestLeaveOneOutPredictor if runner.args.test_looc else TestPredictor
        runner.start(TestClass, ImageFigureConsumer)
    else:
        runner.start(ValidationPredictor, ImageFigureConsumer)
