# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_calibration_error

from htc.evaluation.metrics.ECE import ECE


class TestECE:
    def test_forward(self) -> None:
        ece = ECE(n_bins=3)
        logits = torch.tensor([[10, 5, 101.999, 1.2], [30, -4, 102, 2]])
        softmaxes = F.softmax(logits, dim=0)
        confidences = softmaxes.max(0).values
        labels = torch.tensor([0, 1, 1, 1])  # Predicted labels: [1, 0, 1, 1]
        result = ece(softmaxes, labels)

        accuracies_bins = [0, 1, 1]
        confidences_bins = [0, confidences[2].item(), confidences[[0, 1, 3]].sum().item()]
        probabilities_bins = [0, 1, 3]
        prob_sum = sum(probabilities_bins)
        error = (
            abs(confidences_bins[1] / probabilities_bins[1] - accuracies_bins[1] / probabilities_bins[1])
            * probabilities_bins[1]
            / prob_sum
            + abs(confidences_bins[2] / probabilities_bins[2] - accuracies_bins[2] / probabilities_bins[2])
            * probabilities_bins[2]
            / prob_sum
        )

        assert result["error"] == pytest.approx(error)
        assert result["accuracies"] == accuracies_bins
        assert result["confidences"] == pytest.approx(confidences_bins)
        assert result["probabilities"] == probabilities_bins

        ece_torchmetrics = multiclass_calibration_error(
            softmaxes.permute(1, 0), labels, num_classes=2, n_bins=3, norm="l1"
        )
        assert ece_torchmetrics == pytest.approx(result["error"])

    def test_aggregate_vectors(self) -> None:
        torch.manual_seed(0)
        logits = torch.rand(5, 100)
        labels = torch.randint(0, 5, (100,))
        softmaxes = F.softmax(logits, dim=0)

        # Calculate the ece with half of the samples each and check whether the result is the same as if all samples were used at once
        ece = ECE()
        result1 = ece(softmaxes[:, :50], labels[:50])
        result2 = ece(softmaxes[:, 50:], labels[50:])
        result_full = ece(softmaxes, labels)

        acc_mat = np.stack([result1["accuracies"], result2["accuracies"]])
        conf_mat = np.stack([result1["confidences"], result2["confidences"]])
        prob_mat = np.stack([result1["probabilities"], result2["probabilities"]])
        result_aggregated = ECE.aggregate_vectors(acc_mat, conf_mat, prob_mat)

        assert result_full["error"] == pytest.approx(result_aggregated["error"])

        # Check whether normalization works
        prob = np.array(result_full["probabilities"]).astype(np.float64)
        valid = prob > 0
        acc_normalized = np.array(result_full["accuracies"]).astype(np.float64)
        acc_normalized[valid] = acc_normalized[valid] / prob[valid]
        conf_normalized = np.array(result_full["confidences"])
        conf_normalized[valid] = conf_normalized[valid] / prob[valid]
        prob[valid] = prob[valid] / np.sum(prob)

        assert np.all(result_aggregated["accuracies"] == pytest.approx(acc_normalized))
        assert np.all(result_aggregated["confidences"] == pytest.approx(conf_normalized))
        assert np.all(result_aggregated["probabilities"] == pytest.approx(prob))
