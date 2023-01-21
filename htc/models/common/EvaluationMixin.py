# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
from pathlib import Path

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

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataset_idx: int = 0) -> list[dict]:
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

            current_row = {
                "epoch_index": self.current_epoch,
                "dataset_index": dataset_idx,
                "image_name": image_name,
            }
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

    def validation_epoch_end(self, outputs: list[list[list[dict]]]) -> None:
        # First list: datasets
        # Second list: batches
        # Third list: images
        if len(self.datasets_val) > 1:
            df_epoch = pd.concat([pd.DataFrame(list(itertools.chain(*output))) for output in outputs])
        else:
            df_epoch = pd.DataFrame(list(itertools.chain(*outputs)))

        agg = MetricAggregation(df_epoch, config=self.config)
        self.log_checkpoint_metric(agg.checkpoint_metric())
        self.df_validation_results = pd.concat([self.df_validation_results, df_epoch])

    def on_validation_epoch_end(self) -> None:
        assert self.checkpoint_metric_logged[self.current_epoch], (
            "log_checkpoint_metric was not called! This is strictly necessary so that Lightning knows which model"
            " should be saved"
        )
        self.df_validation_results.to_pickle(Path(self.logger.save_dir) / "validation_results.pkl.xz")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: list[dict]) -> None:
        df_test = pd.DataFrame(list(itertools.chain(*outputs)))
        df_test = df_test.drop(columns=["epoch_index"])
        df_test.to_pickle(Path(self.logger.save_dir) / "test_results.pkl.xz")
