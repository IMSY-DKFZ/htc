# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import gc
from pathlib import Path

import pandas as pd
import torch

from htc.model_processing.Runner import Runner
from htc.model_processing.TestPredictor import TestPredictor
from htc.model_processing.ValidationPredictor import ValidationPredictor
from htc.models.common.HTCLightning import HTCLightning
from htc.settings_seg import settings_seg
from htc.utils.helper_functions import get_nsd_thresholds
from htc.utils.LabelMapping import LabelMapping
from htc_projects.context.models.ContextEvaluationMixin import ContextEvaluationMixin


# We are using the same validation as during training. For this, we need a class which behaves like a HTCLightning class, i.e.
# it must have a predict_step() method but the predictions actually comes from the predictors
class ContextValidationPredictor(ContextEvaluationMixin, ValidationPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = {k: [] for k in self.context_keys}
        self.config["input/no_labels"] = False
        self.evaluation_kwargs = {
            "metrics": ["DSC", "ASD", "NSD", "ECE", "CM"],
            "tolerances": get_nsd_thresholds(LabelMapping.from_config(self.config)),
        }

    def produce_predictions(
        self, model: HTCLightning, batch: dict[str, torch.Tensor], fold_name: str, best_epoch_index: int, **kwargs
    ) -> None:
        self.model = model

        for k in self.context_keys:
            rows = self._validation_context(batch, batch_idx=-1, dataloader_idx=0, context_key=k)
            for r in rows:
                r["fold_name"] = fold_name
                r["epoch_index"] = best_epoch_index  # The validation predictor only uses the best epoch
                r["best_epoch_index"] = best_epoch_index

            self.rows[k] += rows

        # There might be memory overflows without explicit garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> dict[str, torch.Tensor]:
        prediction = self.model.predict_step(batch)
        prediction["class"] = prediction["class"].softmax(dim=1)
        return prediction

    def save_table(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for k in self.context_keys:
            df = pd.DataFrame(self.rows[k])
            df.to_pickle(output_dir / f"validation_table_{k}.pkl.xz")


class ContextTestPredictor(ContextEvaluationMixin, TestPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = {k: [] for k in self.context_keys}
        self.config["input/no_labels"] = False
        self.evaluation_kwargs = {
            "metrics": ["DSC", "ASD", "NSD", "ECE", "CM"],
            "tolerances": get_nsd_thresholds(LabelMapping.from_config(self.config)),
        }

    def produce_predictions(self, model: HTCLightning, batch: dict[str, torch.Tensor], **kwargs) -> None:
        self.model = model

        for k in self.context_keys:
            rows = self._validation_context(batch, batch_idx=-1, dataloader_idx=0, context_key=k)
            self.rows[k] += rows

        gc.collect()
        torch.cuda.empty_cache()

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> dict[str, torch.Tensor]:
        return self.model.predict_step(batch)

    def save_table(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for k in self.context_keys:
            df = pd.DataFrame(self.rows[k])
            if "surface_dice_metric" in df:
                # Same renaming as in the run_tables script
                tolerance_name = settings_seg.nsd_aggregation.split("_")[-1]
                key_nsd = f"surface_dice_metric_{tolerance_name}"
                key_nsd_image = f"surface_dice_metric_image_{tolerance_name}"
                df.rename(
                    columns={"surface_dice_metric": key_nsd, "surface_dice_metric_image": key_nsd_image}, inplace=True
                )

            df.to_pickle(output_dir / f"test_table_{k}.pkl.xz")


if __name__ == "__main__":
    runner = Runner(
        description=(
            "Computes a validation or test table (if the --test argument is supplied) for the given context"
            " transformations. This is an alternative to the run_ttt.py and run_experiment.py scripts and is much"
            " faster because it loads the data only once and performs all transformations and evaluations on the GPU."
            " However, it is limited to the isolation transformations and the DSC metric."
        )
    )
    runner.add_argument("--test")
    runner.add_argument(
        "--transformation-name",
        type=str,
        nargs="+",
        required=True,
        choices=[
            "isolation_0",
            "isolation_cloth",
            "removal_0",
            "removal_cloth",
        ],
        help="Name of the transformation for which the experiment is going to be carried out.",
    )

    transforms = {}
    for name in runner.args.transformation_name:
        if name == "isolation_0":
            transforms[name] = [{"class": "htc_projects.context.context_transforms>OrganIsolation", "fill_value": "0"}]
        elif name == "isolation_cloth":
            transforms[name] = [
                {"class": "htc_projects.context.context_transforms>OrganIsolation", "fill_value": "cloth"}
            ]
        elif name == "removal_0":
            transforms[name] = [{"class": "htc_projects.context.context_transforms>OrganRemoval", "fill_value": "0"}]
        elif name == "removal_cloth":
            transforms[name] = [
                {"class": "htc_projects.context.context_transforms>OrganRemoval", "fill_value": "cloth"}
            ]
        else:
            raise ValueError(f"Invalid name {name}")

    assert len(transforms) > 0, "At least one transformation required"

    config = runner.config
    config["validation/context_transforms_gpu"] = transforms

    if runner.args.test:
        predictor = ContextTestPredictor(runner.run_dir, paths=runner.paths, config=config)
    else:
        assert runner.args.input_dir is None, (
            "Using paths from an arbitrary input directory can only be used with the --test switch"
        )
        predictor = ContextValidationPredictor(runner.run_dir, config=config)

    with torch.autocast(device_type="cuda"):
        predictor.start(task_queue=None, hide_progressbar=False)

    predictor.save_table(runner.output_dir)
