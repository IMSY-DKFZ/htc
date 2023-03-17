# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
import torch

from htc.evaluation.evaluate_images import evaluate_images
from htc.model_processing.ImageConsumer import ImageConsumer
from htc.model_processing.Runner import Runner
from htc.model_processing.TestLeaveOneOutPredictor import TestLeaveOneOutPredictor
from htc.model_processing.TestPredictor import TestPredictor
from htc.model_processing.ValidationPredictor import ValidationPredictor
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.general import apply_recursive
from htc.utils.helper_functions import get_nsd_thresholds
from htc.utils.LabelMapping import LabelMapping


class ImageNSDConsumer(ImageConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get tolerance values for the NSD with the label mapping of the run
        self.tolerances = get_nsd_thresholds(LabelMapping.from_config(self.config))

    def handle_image_data(self, image_data: dict) -> None:
        path = self.path_from_image_data(image_data)
        dataset = DatasetImage([path], train=False, config=self.config)
        sample = dataset[0]

        predictions = torch.from_numpy(image_data["predictions"]).argmax(dim=0).unsqueeze(dim=0)
        nsd = evaluate_images(
            predictions,
            sample["labels"].unsqueeze(dim=0),
            sample["valid_pixels"].unsqueeze(dim=0),
            tolerances=self.tolerances,
            metrics=["NSD"],
        )[0]
        apply_recursive(lambda x: x.cpu().numpy() if type(x) == torch.Tensor else x, nsd)
        results = {"image_name": image_data["image_name"], "metric_data": nsd}
        if "fold_name" in image_data:
            results["fold_name"] = image_data["fold_name"]

        self.results_list.append(results)

    def run_finished(self) -> None:
        tolerance_name = settings_seg.nsd_aggregation.split("_")[-1]
        key_nsd = f"surface_dice_metric_{tolerance_name}"
        key_nsd_image = f"surface_dice_metric_image_{tolerance_name}"

        df_val = pd.read_pickle(self.run_dir / "validation_table.pkl.xz")
        df_test = pd.read_pickle(self.run_dir / "test_table.pkl.xz")
        image_names = [x["image_name"] for x in self.results_list]

        if self.test:
            assert all([image_name in df_test["image_name"].values for image_name in image_names])

            df = df_test
            filename = "test_table.pkl.xz"
            df.drop(
                columns=[key_nsd, key_nsd_image], errors="ignore", inplace=True
            )  # In case the column already exists, we delete it and start fresh

            # Collect NSD results
            rows = []
            for values in self.results_list:
                image_name = values["image_name"]
                result = values["metric_data"]
                assert len(df.query("image_name == @image_name")) == 1, (
                    "There must be exactly one entry in the test table for each image (no unique match for the image"
                    f" id {image_name})"
                )

                rows.append([image_name, result["surface_dice_metric"], result["surface_dice_metric_image"]])

            df_metric = pd.DataFrame(rows, columns=["image_name", key_nsd, key_nsd_image])
            set_image_names = set(df["image_name"].values.tolist())
        else:
            assert all([image_name in df_val["image_name"].values for image_name in image_names])

            df = df_val
            filename = "validation_table.pkl.xz"
            df.drop(
                columns=[key_nsd, key_nsd_image], errors="ignore", inplace=True
            )  # In case the column already exists, we delete it and start fresh

            # We only add the NSD to the best epoch and the first dataset
            dataset_index = 0
            df_best = df.query("epoch_index == best_epoch_index and dataset_index == @dataset_index")

            # Collect NSD results
            rows = []
            for values in self.results_list:
                image_name = values["image_name"]
                fold_name = values["fold_name"]
                result = values["metric_data"]

                df_img = df_best.query("image_name == @image_name and fold_name == @fold_name")
                assert len(df_img) == 1, f"No unique match for the image id {image_name} from fold {fold_name}"

                rows.append(
                    [
                        df_img["epoch_index"].item(),
                        image_name,
                        fold_name,
                        dataset_index,
                        result["surface_dice_metric"],
                        result["surface_dice_metric_image"],
                    ]
                )

            # Add new results to existing table
            df_metric = pd.DataFrame(
                rows, columns=["epoch_index", "image_name", "fold_name", "dataset_index", key_nsd, key_nsd_image]
            )
            set_image_names = set(df_best["image_name"].values.tolist())

        missing_image_names = set_image_names - set(df_metric["image_name"].values.tolist())
        assert len(missing_image_names) == 0, (
            "There should be an NSD value for every image in the table but the following images do not have an NSD"
            f" value: {missing_image_names}"
        )

        # Add new results to existing table
        df_new = df.merge(df_metric, how="left")

        assert len(df) == len(df_new)
        assert pd.isna(df_new[key_nsd]).sum() == len(df) - len(df_metric)
        assert pd.isna(df_new[key_nsd_image]).sum() == len(df) - len(df_metric)

        df_new.to_pickle(self.run_dir / filename)


if __name__ == "__main__":
    runner = Runner(description="Add the NSD metric to every image in either the test or validation table.")
    runner.add_argument("--test")
    runner.add_argument("--test-looc")

    if runner.args.test:
        if (runner.run_dir / "test_table.pkl.xz").exists():
            TestClass = TestLeaveOneOutPredictor if runner.args.test_looc else TestPredictor
            runner.start(TestClass, ImageNSDConsumer)
        else:
            settings.log.warning(
                f"The run directory {runner.run_dir} does not contain a test table. If you want to add the NSD for the"
                " test data, please first generate the test table (via the run_test_table.py script). No NSD metrics"
                " will be computed for the test data"
            )
    else:
        runner.start(ValidationPredictor, ImageNSDConsumer)
