# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as metrics

from htc.models.common.utils import get_n_classes
from htc.tivita.DataPath import DataPath


class SepsisEvaluationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.validation_results_epoch = {"image_labels": [], "predictions": [], "image_names": []}

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if batch_idx == 0:
            assert all(len(values) == 0 for values in self.validation_results_epoch.values()), (
                "Validation results are not properly cleared"
            )

        # logits
        predictions = self(batch)
        image_labels = batch["image_labels"]

        self.validation_results_epoch["image_labels"].append(image_labels)
        self.validation_results_epoch["predictions"].append(predictions)
        self.validation_results_epoch["image_names"].append(batch["image_name_annotations"])

    def on_validation_epoch_end(self) -> None:
        image_labels = torch.cat(self.validation_results_epoch["image_labels"])
        predictions = torch.cat(self.validation_results_epoch["predictions"])
        predictions_labels = torch.argmax(predictions, dim=1)
        image_names = np.concatenate(self.validation_results_epoch["image_names"]).tolist()
        paths = [DataPath.from_image_name(image_name) for image_name in image_names]

        self.log("confidence:", torch.softmax(predictions, dim=1).max(dim=1).values.mean(), prog_bar=True)

        checkpoint_metric = self.config["validation/checkpoint_metric"]
        checkpoint_metric_score = getattr(metrics, checkpoint_metric)(
            predictions_labels,
            image_labels,
            task="multiclass",
            num_classes=get_n_classes(self.config),
            **self.config.get("validation/checkpoint_metric_kwargs", {}),
        )
        self.log_checkpoint_metric(checkpoint_metric_score)

        image_labels = image_labels.cpu().numpy()
        predictions = predictions.cpu().numpy()

        image_meta = [p.image_name_typed() for p in paths]
        keys = set()
        for t in image_meta:
            keys.update(t.keys())
        image_meta = {key: [value.get(key) for value in image_meta] for key in keys}
        image_meta["image_name"] = [p.image_name() for p in paths]

        df_epoch = pd.DataFrame(
            {
                "epoch_index": [self.current_epoch] * len(paths),
                "dataset_index": [0] * len(paths),
                "image_labels": image_labels.tolist(),
                "predictions": [x.astype(np.float32) for x in predictions],
            }
            | image_meta
        )

        self.df_validation_results = pd.concat([self.df_validation_results, df_epoch])
        self.df_validation_results.to_pickle(Path(self.logger.save_dir) / "validation_results.pkl.xz")

        # Start clean for the next validation round
        self.validation_results_epoch = {"image_labels": [], "predictions": [], "image_names": []}
