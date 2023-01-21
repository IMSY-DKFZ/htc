# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

from htc.evaluation.metrics.scores import accuracy_from_cm, confusion_matrix_groups
from htc.models.common.class_weights import calculate_class_weights
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.utils import get_n_classes
from htc.models.pixel.ModelPixel import ModelPixel
from htc.tissue_atlas.median_pixel.DatasetMedianPixel import DatasetMedianPixel


class LightningMedianPixel(HTCLightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = ModelPixel(self.config)

        if self.config["model/class_weight_method"] and self.dataset_train is not None:
            weights = calculate_class_weights(self.config, *self.dataset_train.label_counts())
        else:
            weights = torch.ones(get_n_classes(self.config))

        self.ce_loss_weighted = nn.CrossEntropyLoss(weight=weights)

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        return DatasetMedianPixel(**kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.config["input/oversampling"]:
            config = copy.copy(self.config)
            config["model/class_weight_method"] = "1∕m"  # This gives the "true" values needed for oversampling
            config[
                "model/background_weight"
            ] = None  # Disable manual background weight as this could lead to a higher background sampling (since the 1/m values are usually smaller)

            weights = calculate_class_weights(config, *self.dataset_train.label_counts())
            sample_weights = weights[self.dataset_train.labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=self.config["input/epoch_size"])
        else:
            sampler = RandomSampler(self.dataset_train, replacement=True, num_samples=self.config["input/epoch_size"])

        return DataLoader(
            self.dataset_train, sampler=sampler, persistent_workers=True, **self.config["dataloader_kwargs"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["class"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        labels = batch["labels"]
        features = batch["features"]

        predictions = self(features)
        ce_loss = self.ce_loss_weighted(predictions, labels)
        self.log("train/ce_loss", ce_loss, on_epoch=True)

        return {"loss": ce_loss}

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        predictions = self(batch["features"]).argmax(dim=1)

        return {"labels": batch["labels"], "predictions": predictions, "image_names": batch["image_name"]}

    def validation_epoch_end(self, outputs: list[dict]) -> None:
        labels = torch.cat([x["labels"] for x in outputs if x is not None])
        predictions = torch.cat([x["predictions"] for x in outputs if x is not None])
        image_names = np.concatenate([x["image_names"] for x in outputs if x is not None]).tolist()

        cm_pigs = confusion_matrix_groups(predictions, labels, image_names, get_n_classes(self.config))

        rows = []
        for subject_name, cm in cm_pigs.items():
            accuracy = accuracy_from_cm(cm)
            rows.append(
                {
                    "epoch_index": self.current_epoch,
                    "dataset_index": 0,
                    "image_name": subject_name,  # We do not store image-level metrics here, only on the subject level
                    "subject_name": subject_name,
                    "accuracy": accuracy,
                    "confusion_matrix": cm.cpu().numpy(),
                }
            )

        df_epoch = pd.DataFrame(rows)
        self.df_validation_results = pd.concat([self.df_validation_results, df_epoch])
        self.log_checkpoint_metric(self.df_validation_results["accuracy"].mean())

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        labels = batch["labels"]
        features = batch["features"]
        image_names = batch["image_name"]

        logits = self(features)

        return {"image_names": image_names, "labels": labels, "logits": logits}

    def test_epoch_end(self, outputs: list[dict]) -> None:
        results = {}
        results["logits"] = torch.cat([x["logits"] for x in outputs if x is not None]).cpu().numpy()
        results["labels"] = torch.cat([x["labels"] for x in outputs if x is not None]).cpu().numpy()
        results["image_names"] = np.concatenate([x["image_names"] for x in outputs if x is not None])

        np.savez_compressed(Path(self.logger.save_dir) / "test_results.npz", **results)
