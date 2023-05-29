# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from htc.evaluation.evaluate_images import evaluate_images
from htc.models.common.MetricAggregation import MetricAggregation
from htc.tivita.DataPath import DataPath


class EvaluationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        softmaxes = F.softmax(logits["class"], dim=1)
        batch_results_class = evaluate_images(
            softmaxes, batch["labels"], batch["valid_pixels"], n_classes=softmaxes.shape[1]
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
            image_name = batch["image_name"][b]
            path = DataPath.from_image_name(image_name)

            current_row = {}
            if hasattr(self, "current_epoch"):
                current_row["epoch_index"] = self.current_epoch
            current_row["dataset_index"] = dataloader_idx
            current_row["image_name"] = image_name
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
