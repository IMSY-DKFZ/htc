# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.cpp import spxs_predictions
from htc.evaluation.evaluate_images import evaluate_images
from htc.settings import settings
from htc.settings_seg import settings_seg


class EvaluateSuperpixelImage:
    def evaluate_cpp(self, sample: dict) -> dict:
        # Assign each pixel the label of the superpixel, i.e. the mode of the reference annotation
        predictions, spx_label_counts = spxs_predictions(sample["spxs"], sample["labels"], sample["valid_pixels"])

        return {
            "evaluation": self._evaluate(predictions, sample),
            "predictions": predictions,
            "label_counts": spx_label_counts,
        }

    def spxs_predictions_py(
        self, spxs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # WARNING: this function exists just for performance comparison. Don't use it!

        # Assign each pixel the label of the superpixel, i.e. the mode of the reference annotation
        predictions = torch.ones(labels.shape, dtype=torch.int64) * len(settings_seg.labels)
        spx_indices = spxs.unique()
        spx_label_counts = torch.zeros(len(spx_indices), len(settings_seg.labels), dtype=torch.int64)

        for s in spx_indices:
            spx_labels = labels[spxs == s]
            spx_mask = mask[spxs == s]

            spx_labels = spx_labels[spx_mask]
            assert torch.all(spx_labels < settings.label_index_thresh), "Invalid label"

            if spx_labels.numel() == 0:
                # Superpixel contains only invalid pixels --> assign background
                label = 0
            else:
                l, c = spx_labels.unique(return_counts=True)
                spx_label_counts[s, l] = c

                label = spx_label_counts[s].argmax().item()
                assert label == spx_labels.mode().values.item(), "Wrong superpixel label"

            predictions[spxs == s] = label

        assert torch.all(predictions != len(settings_seg.labels)), "All pixels must get a label assigned"

        return predictions, spx_label_counts

    def evaluate_py(self, sample: dict) -> dict:
        # WARNING: this function exists just for performance comparison. Don't use it!
        predictions, spx_label_counts = self.spxs_predictions_py(
            sample["spxs"], sample["labels"], sample["valid_pixels"]
        )

        return {
            "evaluation": self._evaluate(predictions, sample),
            "predictions": predictions,
            "label_counts": spx_label_counts,
        }

    def _evaluate(self, predictions: torch.Tensor, sample: dict) -> dict:
        predictions = predictions.unsqueeze(dim=0)
        labels = sample["labels"].unsqueeze(dim=0)
        mask = sample["valid_pixels"].unsqueeze(dim=0)
        result = evaluate_images(predictions, labels, mask)[0]

        return result
