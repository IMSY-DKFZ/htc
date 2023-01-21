# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLossWeighted(nn.KLDivLoss):
    """
    Weighted version of the KLDivLoss allowing to account for class imbalances.

    Only the 'batchmean' reduction method is implemented (as weighted version).
    """

    def __init__(self, weight: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(reduction="none", *args, **kwargs)
        self.register_buffer(
            "weight", weight
        )  # See _WeightedLoss in PyTorch; ensures that weights get transferred to the GPU if necessary

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(input, target)

        # Weight both, the loss and the target
        loss = loss * self.weight
        target = target * self.weight

        # Normal kl div batchmean would weight by target.size(0)
        # However, if we reduce the numerator by a factor x, we should also reduce the denominator by the same factor. This ensures that the scaling of the weighted kl div loss is comparable to the weighted ce loss (it is actually identical with one-hot targets, cf. tests)
        loss = loss.sum() / target.sum()  # weighted batchmean

        return loss


class SuperpixelLoss(nn.Module):
    """Loss to ensure that all predictions inside a superpixel are coherent."""

    def forward(self, predictions: torch.Tensor, spxs: torch.Tensor) -> torch.Tensor:
        assert len(predictions.shape) == 2, "predictions should be a matrix of shape (batch, classes)"
        assert len(spxs.shape) == 1 and spxs.dtype == torch.int64, "spxs should be an index vector of shape (batch)"
        assert predictions.size(0) == spxs.size(0), "Each prediction should have a superpixel label id"

        predictions = F.softmax(
            predictions, dim=1
        )  # We need probabilities which we want to enforce to be unique (e.g. [1, 0, 0] instead of [0.3, 0.4, 0.3]) [N, 19]

        spxs = spxs.unsqueeze(dim=1).expand(spxs.size(0), predictions.size(1))  # [N, 19]
        spx_indices, spx_counts = spxs.unique(dim=0, return_counts=True)  # [N, 19], [N]

        # Calculate the group average based on https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/2
        spx_avg = torch.zeros(
            spx_indices.max() + 1, predictions.size(1), device=predictions.device, dtype=predictions.dtype
        )  # Matrix which adds all probabilites per superpixel [M, 19]
        spx_avg = spx_avg.scatter_add_(
            0, spxs, predictions
        )  # Add the prediction probabilites at the positions of the superpixel ids [M, 19]
        spx_avg = spx_avg[
            spx_indices[:, 0]
        ]  # Not all superpixels are used (due to invalid pixels etc.) so some ids are missing [N, 19]
        spx_avg = spx_avg / spx_counts.unsqueeze(dim=1)  # [N, 19]

        # Gini based on the (mean) probability per superpixel. Only if all predictions inside the superpixel are in coherence, the gini will have a low value
        # Note: the mean of probability vectors is still a valid probability vector
        gini = 1 - (spx_avg * spx_avg).sum(dim=1)  # [N]
        gini = gini.mean()

        return gini  # [0;1] = [low loss;high loss] = [all pixels of the superpixel are assigned to the same class;the predictions are equally distributed across all classes]
