# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import itertools
from pathlib import Path

import pandas as pd
import torch

from htc.cpp import nunique
from htc.evaluation.evaluate_images import evaluate_images
from htc.models.common.MetricAggregation import MetricAggregation
from htc.models.common.torch_helpers import copy_sample
from htc.models.common.transforms import HTCTransformation
from htc.tivita.DataPath import DataPath
from htc.utils.LabelMapping import LabelMapping


class ContextEvaluationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_kwargs = {}  # Additional arguments for the evaluate_images function (e.g. additional metrics)
        self.df_validation_results_baseline = pd.DataFrame()

        # One context transformation type per key (e.g. isolation_0 or isolation_cloth)
        # For each key, a separate evaluation will be carried out yielding a validation table per key
        if self.config["validation/context_transforms_gpu"]:
            self.context_keys = list(self.config["validation/context_transforms_gpu"].keys())
        else:
            self.context_keys = []
        self.df_validation_results_context = {k: pd.DataFrame() for k in self.context_keys}

        self.context_transforms = {}
        self.is_isolation_transform = {}  # bool for each context transform whether it is an isolation transform
        for k in self.context_keys:
            self.context_transforms[k] = {
                label_index: HTCTransformation.parse_transforms(
                    self.config["validation/context_transforms_gpu"][k],
                    target_label=label_index,
                    config=self.config,
                )
                for label_index in LabelMapping.from_config(self.config).label_indices()
            }
            self.is_isolation_transform[k] = False
            for transform in self.config["validation/context_transforms_gpu"][k]:
                if transform["class"] == "htc.context.context_transforms>OrganIsolation":
                    self.is_isolation_transform[k] = True

        self.validation_results_epoch = {"baseline": []}
        for k in self.context_keys:
            self.validation_results_epoch[k] = []

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0 and dataloader_idx == 0:
            assert all(
                len(values) == 0 for values in self.validation_results_epoch.values()
            ), "Validation results are not properly cleared"

        self.validation_results_epoch["baseline"].append(self._validation_baseline(batch, batch_idx, dataloader_idx))
        for k in self.context_keys:
            self.validation_results_epoch[k].append(self._validation_context(batch, batch_idx, dataloader_idx, k))

    def on_validation_epoch_end(self) -> None:
        # First level (dict): validation types (baseline, isolation_0, etc.)
        # Second level (list): batches
        # Third level (list): images
        # Fourth level (dict): results per image
        df_baseline = pd.DataFrame(list(itertools.chain.from_iterable(self.validation_results_epoch["baseline"])))
        context_tables = {
            k: pd.DataFrame(list(itertools.chain.from_iterable(self.validation_results_epoch[k])))
            for k in self.context_keys
        }

        # Check for isolation context transforms that baseline and context tables have the same length:
        for k in context_tables.keys():
            if self.is_isolation_transform[k]:
                assert len(df_baseline) == len(
                    context_tables[k]
                ), "Baseline and context tables must have the same length for isolation context transforms"
                assert (
                    df_baseline["image_name"] == context_tables[k]["image_name"]
                ).all(), "Images must align in both baseline and context tables for isolation context transforms"
                assert all(
                    (df_baseline["used_labels"].iloc[i] == context_tables[k]["used_labels"].iloc[i]).all()
                    for i in range(len(df_baseline))
                ), "used_labels must be identical in both baseline and context tables for isolation context transforms"

        # Only the default results will be reported (with "checkpoint_saving": "last" it does not matter anyway)
        agg = MetricAggregation(df_baseline, config=self.config)
        self.log_checkpoint_metric(agg.checkpoint_metric())

        self.df_validation_results_baseline = pd.concat([self.df_validation_results_baseline, df_baseline])
        for k in self.context_keys:
            self.df_validation_results_context[k] = pd.concat(
                [self.df_validation_results_context[k], context_tables[k]]
            )

        # We keep the baseline name so that the rest of the pipeline still works as expected
        self.df_validation_results_baseline.to_pickle(Path(self.logger.save_dir) / "validation_results.pkl.xz")
        for k in self.context_keys:
            self.df_validation_results_context[k].to_pickle(
                Path(self.logger.save_dir) / f"validation_results_{k}.pkl.xz"
            )

        # Start clean for the next validation round
        self.validation_results_epoch = {"baseline": []}
        for k in self.context_keys:
            self.validation_results_epoch[k] = []

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        raise NotImplementedError()

    def on_test_epoch_end(self) -> None:
        raise NotImplementedError()

    def _validation_baseline(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int) -> list[dict]:
        batch_clean = {k: v for k, v in batch.items() if not k.startswith("labels")}

        logits = self.predict_step(batch_clean)
        results = evaluate_images(
            logits["class"].argmax(dim=1),
            batch["labels"],
            batch["valid_pixels"],
            n_classes=logits["class"].shape[1],
            **self.evaluation_kwargs,
        )

        rows = []
        for b in range(len(results)):
            image_name = batch["image_name"][b]
            path = DataPath.from_image_name(image_name)

            current_row = {}
            if hasattr(self, "current_epoch"):
                current_row["epoch_index"] = self.current_epoch
            current_row["dataset_index"] = dataloader_idx
            current_row["image_name"] = image_name
            current_row |= path.image_name_typed()

            for key, value in results[b].items():
                if type(value) == torch.Tensor:
                    current_row[key] = value.cpu().numpy()
                else:
                    current_row[key] = value

            rows.append(current_row)

        assert len(rows) == batch["labels"].size(0)
        return rows

    def _validation_context(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int, context_key: str
    ) -> list[dict]:
        # The validation is done for every label in the image
        # The current approach operates on the batch level aware of the fact that not every label is included in every image. However, performing the transformations on the image level is much more inefficient even though less calculations are needed in theory (probably because more device sync points are necessary)
        label_results = {}
        for label_index in batch["labels"][batch["valid_pixels"]].unique():
            label_index = label_index.item()

            batch_copy = copy_sample(batch)
            for t in self.context_transforms[context_key][label_index]:
                batch_copy = t(batch_copy)
            batch_copy_clean = {k: v for k, v in batch_copy.items() if not k.startswith("labels")}

            logits = self.predict_step(batch_copy_clean)
            res = evaluate_images(
                logits["class"].argmax(dim=1),
                batch_copy["labels"],
                batch_copy["valid_pixels"],
                n_classes=logits["class"].shape[1],
                **self.evaluation_kwargs,
            )

            # We don't need the validation data on the GPU anymore
            for batch_item in res:
                for key, values in batch_item.items():
                    if type(values) == torch.Tensor:
                        batch_item[key] = values.cpu()

            label_results[label_index] = res

        # The table generation is a bit more complex in the context case because we have separate results for each label per image
        # because we apply the transformation on a per label basis
        rows = []
        for b in range(batch["labels"].size(0)):
            image_name = batch["image_name"][b]
            path = DataPath.from_image_name(image_name)

            current_row = {}
            if hasattr(self, "current_epoch"):
                current_row["epoch_index"] = self.current_epoch
            current_row["dataset_index"] = dataloader_idx
            current_row["image_name"] = image_name
            current_row |= path.image_name_typed()

            if self.is_isolation_transform[context_key]:
                # The aggregation of the results is different for the isolation case because here we have only one row per image instead of one row per image and label
                combined_values = {}
                for label_index, batch_results in label_results.items():
                    # Not every image has every label, so we need to check whether the current label is part of the validation
                    if label_index in batch_results[b]["used_labels"]:
                        # Now we can go through the results for this image
                        for key, values in batch_results[b].items():
                            # We first collect all the results and then combine it later (e.g. concatenate all dice_metric values)
                            if type(values) == torch.Tensor:
                                if key not in combined_values:
                                    combined_values[key] = []
                                combined_values[key].append(values)

                for key, values in combined_values.items():
                    if key == "confusion_matrix":
                        current_row[key] = torch.stack(values).sum(dim=0).numpy()
                    elif key == "used_labels":
                        labels = torch.cat(values)
                        assert len(labels) == nunique(labels)
                        current_row[key] = labels.numpy()
                    else:
                        current_row[key] = torch.cat(values).numpy()

                rows.append(current_row)
            else:
                # Result per image and label. This is not directly comparable to the usual test tables and must be combined separately

                # All used labels from all label indices because a transformation may remove an image (e.g. removal_0)
                image_labels = torch.cat([r[b]["used_labels"] for r in label_results.values()])
                for label_index, batch_results in label_results.items():
                    # Not every image has every label, so we need to check whether the current label is part of the validation
                    if label_index in image_labels:
                        current_row_label = copy.deepcopy(current_row)
                        current_row_label["target_label"] = label_index

                        for key, values in batch_results[b].items():
                            if key == "confusion_matrix":
                                current_row_label[key] = values.numpy()
                            elif key == "used_labels":
                                assert len(values) == nunique(values)
                                current_row_label[key] = values.numpy()
                            else:
                                if type(values) == torch.Tensor:
                                    values = values.numpy()
                                current_row_label[key] = values

                        rows.append(current_row_label)

        return rows
