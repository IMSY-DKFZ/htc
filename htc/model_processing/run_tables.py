# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy

import numpy as np
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


class ImageTableConsumer(ImageConsumer):
    def __init__(self, *args, test_table_name: str = "test_table", **kwargs):
        """Adds all post-hoc metrics (e.g. ASD, NSD) to the validation or test table."""
        super().__init__(*args, **kwargs)

        # Get tolerance values for the NSD with the label mapping of the run

        if "NSD" in self.metrics:
            self.tolerances = get_nsd_thresholds(LabelMapping.from_config(self.config))
        else:
            self.tolerances = None

        self.test_table_name = test_table_name

    def handle_image_data(self, image_data: dict) -> None:
        config = copy.copy(self.config)
        config["input/preprocessing"] = None
        config["input/no_features"] = True  # As we only need labels from the sample

        path = image_data["path"]
        sample = DatasetImage([path], train=False, config=config)[0]

        predictions = torch.from_numpy(image_data["predictions"]).unsqueeze(dim=0)
        labels = sample["labels"].unsqueeze(dim=0)
        valid_pixels = sample["valid_pixels"].unsqueeze(dim=0)

        if valid_pixels.sum() == 0:
            settings.log.info(f"The image {path.image_name()} is skipped because it contains no valid pixels")
            return

        metric_data = {}

        # this script can also be used without setting the metrics parameter i.e. in case the metrics don't have to
        # computed, but only the predictions are to be stored
        if len(self.metrics) > 0:
            metric_data |= evaluate_images(
                predictions.float(),
                labels,
                valid_pixels,
                n_classes=predictions.shape[1],
                tolerances=self.tolerances,
                metrics=self.metrics + ["ECE", "CM"],
            )[0]

        metric_data["image_name"] = path.image_name()
        metric_data |= path.image_name_typed()

        if "fold_name" in image_data:
            metric_data["fold_name"] = image_data["fold_name"]

        apply_recursive(lambda x: x.cpu().numpy() if type(x) == torch.Tensor else x, metric_data)

        self.results_list.append(metric_data)

    def run_finished(self) -> None:
        if self.test:
            self._save_test_table()
        else:
            self._save_validation_table()

    def _save_validation_table(self) -> None:
        # Adding the new results to the validation table is a bit complicated since we only have results for the best epoch and the first dataset (but still want to keep the per-epoch results)
        df_results = pd.DataFrame(list(self.results_list))
        assert len(df_results) > 0, "No results calculated"
        assert "image_name" in df_results and "fold_name" in df_results

        # Add the distance aggregation of the nsd to the key
        if "NSD" in self.metrics:
            tolerance_name = settings_seg.nsd_aggregation.split("_")[-1]
            key_nsd = f"surface_dice_metric_{tolerance_name}"
            key_nsd_image = f"surface_dice_metric_image_{tolerance_name}"
            df_results.rename(
                columns={"surface_dice_metric": key_nsd, "surface_dice_metric_image": key_nsd_image}, inplace=True
            )

        df_results.sort_values(by=["fold_name", "image_name"], inplace=True, ignore_index=True)

        # We load and extend the validation table. This is a bit tricky since we only want to extend the best epoch per fold and only for the first dataset
        df_val = pd.read_pickle(self.run_dir / "validation_table.pkl.xz")
        df_val_best = df_val.query("epoch_index == best_epoch_index and dataset_index == 0").copy()
        df_val_best.sort_values(by=["fold_name", "image_name"], inplace=True, ignore_index=True)
        assert (
            df_val_best.index.identical(df_results.index)
            and np.all(df_val_best["image_name"].values == df_results["image_name"].values)
            and np.all(df_val_best["fold_name"].values == df_results["fold_name"].values)
        ), "Results and validation table are not aligned"
        if "used_labels" in df_results:
            assert np.all(
                np.concatenate(
                    np.equal(df_results["used_labels"].values, df_val_best["used_labels"].values, dtype=object)
                )
            ), "Used labels do not match between tables"

        if "dice_metric_image" in df_results:
            # The dice is always calculated during training, here we just check if we get a similar result as during training
            dice_old = df_val_best["dice_metric_image"].values
            dice_new = df_results["dice_metric_image"].values
            dice_abs_diff = np.abs(dice_new - dice_old)
            threshold = 0.01
            if not np.all(dice_abs_diff <= threshold):
                settings.log.warning(
                    "Differences between old (calculated during training) and new (calculated via predictions) dice"
                    f" scores are larger than {threshold} for {np.sum(dice_abs_diff > threshold)} out of"
                    f" {len(dice_abs_diff)} images (mean abs diff: {np.mean(dice_abs_diff)})"
                )

        # We need to copy all id columns to the results table so that the merge works
        id_columns = ["epoch_index", "best_epoch_index", "dataset_index"]
        for copy_column in id_columns:
            df_results[copy_column] = df_val_best[copy_column].values

        # These columns are already available in the results table but we still need them for the join
        id_columns += ["fold_name", "image_name"]

        # Merge results into existing validation table, will result in duplicate columns
        df_val_new = df_val.merge(
            df_results, how="left", on=id_columns, validate="one_to_one", suffixes=("_old", "_new")
        )

        # Overwrite _old columns with _new ones but only for the best epoch and the first dataset (for the remaining rows we keep the old data)
        for c in df_results.columns:
            if c not in id_columns and f"{c}_new" in df_val_new and f"{c}_old" in df_val_new:
                df_val_new[c] = df_val_new[f"{c}_new"].fillna(df_val_new[f"{c}_old"])

        df_val_new.drop(columns=df_val_new.filter(regex="_(?:old|new)$").columns.tolist(), inplace=True)
        assert not any(c.endswith(("_old", "_new")) for c in df_val_new.columns), "Merge columns are still present"
        assert all(c in df_val_new for c in df_results.columns), "Some columns are missing"

        # Some checks that the merge actually worked
        df_changed = df_val_new.query("epoch_index == best_epoch_index and dataset_index == 0")
        df_unchanged = df_val_new.query("not (epoch_index == best_epoch_index and dataset_index == 0)")
        assert len(df_changed) + len(df_unchanged) == len(df_val_new), "Tables do not add up"

        assert (
            pd.isna(df_unchanged[[c for c in df_results.columns if c not in df_val]]).all().all()
        ), "Old values for non-existing columns must be nan"
        assert not pd.isna(df_changed[df_results.columns]).all().all(), "All new values must be non-nan"
        assert len(df_val) == len(df_val_new), "The validation table must not change in length"

        # Finally, overwrite existing validation table
        df_val_new.reset_index(inplace=True, drop=True)
        df_val_new.to_pickle(self.target_dir / "validation_table.pkl.xz")

    def _save_test_table(self) -> None:
        # For the test table it is easier, as we create it from scratch (and overwrite an existing one, if available)
        df_results = pd.DataFrame(list(self.results_list))
        assert len(df_results) > 0, "No results calculated"
        assert len(df_results) == len(df_results["image_name"].unique()), "There must be exactly one row per image"

        # Add the distance aggregation of the nsd to the key

        if "NSD" in self.metrics:
            tolerance_name = settings_seg.nsd_aggregation.split("_")[-1]
            key_nsd = f"surface_dice_metric_{tolerance_name}"
            key_nsd_image = f"surface_dice_metric_image_{tolerance_name}"
            df_results.rename(
                columns={"surface_dice_metric": key_nsd, "surface_dice_metric_image": key_nsd_image}, inplace=True
            )

        df_results.to_pickle(self.target_dir / f"{self.test_table_name}.pkl.xz")


if __name__ == "__main__":
    runner = Runner(description="Re-create the test or validation table for a run with all requested metrics.")
    runner.add_argument("--test")
    runner.add_argument("--test-looc")
    runner.add_argument("--metrics")
    runner.add_argument("--output-dir")

    if runner.args.test:
        TestClass = TestLeaveOneOutPredictor if runner.args.test_looc else TestPredictor
        runner.start(TestClass, ImageTableConsumer)
    else:
        runner.start(ValidationPredictor, ImageTableConsumer)
