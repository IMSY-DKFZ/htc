# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from htc.evaluation.evaluate_images import evaluate_images
from htc.models.common.MetricAggregation import MetricAggregation
from htc.models.common.utils import multi_label_condensation
from htc.settings import settings
from htc.tivita.DataPath import DataPath


class EvaluationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_kwargs = {}  # Additional arguments for the evaluate_images function (e.g. additional metrics)
        self.df_validation_results = pd.DataFrame()
        self.validation_results_epoch = []  # Also used for storing the test results

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0 and dataloader_idx == 0:
            assert len(self.validation_results_epoch) == 0, "Validation results are not properly cleared"

        self.validation_results_epoch.append(self._validate_batch(batch, dataloader_idx))

    def on_validation_epoch_end(self) -> None:
        # First level (list): batches
        # Second level (list): images
        # Third level (dict): results per image
        df_epoch = pd.DataFrame(list(itertools.chain.from_iterable(self.validation_results_epoch)))

        agg = MetricAggregation(df_epoch, config=self.config)
        self.log_checkpoint_metric(agg.checkpoint_metric())
        self.df_validation_results = pd.concat([self.df_validation_results, df_epoch])
        self.df_validation_results.to_pickle(Path(self.logger.save_dir) / "validation_results.pkl.xz")

        # Start clean for the next validation round
        self.validation_results_epoch = []

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        df_test = pd.DataFrame(list(itertools.chain.from_iterable(self.validation_results_epoch)))
        df_test.drop(columns=["epoch_index"], inplace=True)
        df_test.to_pickle(Path(self.logger.save_dir) / "test_results.pkl.xz")
        self.validation_results_epoch = []

    def _validate_batch(self, batch: dict[str, torch.Tensor], dataloader_idx: int) -> list[dict[str, Any]]:
        batch_clean = {k: v for k, v in batch.items() if not k.startswith("labels")}

        logits = self.predict_step(batch_clean)
        if self.config["model/activations"] == "sigmoid":
            predictions = multi_label_condensation(logits["class"], self.config)["predictions"]
        else:
            predictions = F.softmax(logits["class"], dim=1)

        valid_images = batch["valid_pixels"].any(dim=(1, 2))
        if not valid_images.all():
            settings.log.error(
                "There are some images which do not contain any valid pixel. Evaluation cannot be carried out:"
                f" {np.asarray(batch['image_name_annotations'])[(~valid_images).cpu().numpy()]}"
            )

        batch_results_class = evaluate_images(
            predictions,
            batch["labels"],
            batch["valid_pixels"],
            n_classes=logits["class"].shape[1],
            **self.evaluation_kwargs,
        )

        domains = self.config.get("input/target_domain", [])
        predicted_domains = {}
        for domain in domains:
            if domain in logits:
                # We may use the domain but to not necessarily also predict it
                predicted_domains[domain] = logits[domain].argmax(dim=1)
                assert len(batch_results_class) == len(predicted_domains[domain])

        rows = []
        for b in range(len(batch_results_class)):
            path = DataPath.from_image_name(batch["image_name_annotations"][b])

            current_row = {}
            if hasattr(self, "current_epoch"):
                current_row["epoch_index"] = self.current_epoch
            current_row["dataset_index"] = dataloader_idx
            current_row["image_name"] = path.image_name()
            current_row |= path.image_name_typed()

            for key, value in batch_results_class[b].items():
                if type(value) == torch.Tensor:
                    current_row[key] = value.cpu().numpy()
                else:
                    current_row[key] = value

            for domain in domains:
                current_row[domain] = batch[domain][b].item()
                if domain in predicted_domains:
                    current_row[f"{domain}_predicted"] = predicted_domains[domain][b].item()

            rows.append(current_row)

        return rows
