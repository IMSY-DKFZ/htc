# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


class BrierScore(nn.Module):
    def __init__(self, n_classes: int, variant: str = "multiclass"):
        """
        Computes the [Brier Score](https://metrics-reloaded.dkfz.de/metric?id=brier_score).

        Args:
            n_classes: The number of classes.
            variant: There is a different definition of the Brier score for binary classification which yields values in [0; 1] whereas the multiclass variant (which can also be used in the binary case) yields values in [0; 2]. Set either to "binary" or "multiclass".
        """
        super().__init__()
        self.n_classes = n_classes
        self.variant = variant
        assert self.n_classes >= 2, "At least two classes are required for the Brier Score"
        assert self.variant in ["binary", "multiclass"], "Invalid variant"
        if self.variant == "binary":
            assert self.n_classes == 2, "Binary variant requires exactly two classes"

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Brier score for a set of samples.

        Args:
            predictions: The predicted probabilities for each class (batch, classes).
            labels: The reference labels (batch).
        """
        assert predictions.size(1) == self.n_classes, (
            "The second dimensions of the predictions tensor does not match with the number of classes"
        )
        assert labels.ndim == 1, "Labels should be a vector"

        if self.variant == "binary":
            return torch.functional.F.mse_loss(predictions[:, 1], labels)
        else:
            labels = torch.functional.F.one_hot(labels, num_classes=self.n_classes)
            return torch.mean(torch.sum((predictions - labels) ** 2, dim=1))
