# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Union

import numpy as np
import torch
import torch.nn as nn


class ECELoss(nn.Module):
    def __init__(self, n_bins: int = 10):
        """
        Calculates the Expected Calibration Error (ECE) of a model.

        This metric can be used to estimate how well a model is calibrated with lower values indicating a better calibrated model (for more information see https://arxiv.org/abs/1706.04599). For this, it takes the predictions of the network (the softmax values) where the highest value in the softmax vector corresponds to the confidence for that prediction. Then, the predictions are binned into confidence intervals and for each bin the difference between the average confidence and accuracy is calculated. A model is calibrated when this difference is small. The intuition is that if there are, for example, 100 samples in the bin between 0.7 and 0.8 with an average confidence of 0.77, then around 77 % of those samples should be correctly classified assuming a classifier which knows about its unsureness.

        Note: The model's calibration can be improved by using the temperature scaling method (e.g. https://github.com/gpleiss/temperature_scaling).

        Args:
            n_bins: Number of confidence interval bins.
        """
        super().__init__()
        self.n_bins = n_bins

    def forward(self, softmaxes: torch.Tensor, labels: torch.Tensor) -> dict[str, Union[float, list, list, list]]:
        """
        Calculates the ECE values for a set of samples.

        Args:
            softmaxes: The softmax output of your network (class, batch).
            labels: Reference labels (batch).

        Returns:
            dict: Dictionary with the ECE "error" plus the unnormalized vectors ("accuracies", "confidences", "probabilities"). Please refer to the `aggregate_vectors()` method if you need to aggregate those values across multiple samples.
        """
        assert len(softmaxes.shape) == 2 and len(labels.shape) == 1, "Invalid shape"
        assert softmaxes.shape[1] == labels.shape[0], "The sample dimension does not match between softmaxes and labels"

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=softmaxes.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes_sum = torch.sum(softmaxes, dim=0)
        assert torch.allclose(softmaxes_sum, torch.ones_like(softmaxes_sum)), (
            "All of the softmax should sum upto approx. 1 in the class dimension. Are you sure that you are not sending"
            " logits rather than softmaxes?"
        )

        confidences, predictions = torch.max(softmaxes, dim=0)
        accuracies = predictions.eq(labels)

        accuracies_bins = []
        confidences_bins = []
        probabilities_bins = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin

            # Group values based on confidence
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)

            # Proportion of confidence values which ended up in this bin
            prop_in_bin = in_bin.sum().item()
            probabilities_bins.append(prop_in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].sum().item()
                confidence_in_bin = confidences[in_bin].sum().item()

                accuracies_bins.append(accuracy_in_bin)
                confidences_bins.append(confidence_in_bin)
            else:
                # No values in the bin --> add zeros
                accuracies_bins.append(0)
                confidences_bins.append(0)

        # Calculate the ece error
        prob = torch.Tensor(probabilities_bins)
        acc_normalized = torch.Tensor(accuracies_bins) / prob
        conf_normalized = torch.Tensor(confidences_bins) / prob
        prob_normalized = prob / prob.sum()
        valid_bins = prob_normalized > 0

        # ECE is the absolute difference between the confidences and accuracies in the bin weighted by the number of samples which are in this bin
        ece = torch.sum(
            torch.abs(conf_normalized[valid_bins] - acc_normalized[valid_bins]) * prob_normalized[valid_bins]
        )

        return {
            "error": ece.item(),
            "accuracies": accuracies_bins,
            "confidences": confidences_bins,
            "probabilities": probabilities_bins,
        }

    @staticmethod
    def aggregate_vectors(acc_mat: np.ndarray, conf_mat: np.ndarray, prob_mat: np.ndarray):
        """
        This function aggregates the ece vectors from multiple images. This is useful when the ece cannot be calculated for all samples at once. All matrices must have the shape (n_batches, n_bins). Note that only when the raw counts are passed to this function the real ece can be calculated. If normalized vectors are passed, then the assumption is made that the original number of samples before the normalization was the same for all images (which is an approximation).

        Args:
            acc_mat: Matrix with either the counts representing the number of correctly classified samples per bin or the accuracy per bin.
            conf_mat: Matrix with either the sum of the confidence values or the normalized confidence per bin.
            prob_mat: Matrix with either the total counts or the ratio of samples per bin.

        Returns: A dictionary with "accuracies", "confidences" and "probabilities" vectors (all normalized) as well as the ece "error".
        """
        assert (
            acc_mat.shape == conf_mat.shape and acc_mat.shape == prob_mat.shape
        ), "All matrices must have the same shape"
        assert np.all(acc_mat >= 0) and np.all(conf_mat >= 0) and np.all(prob_mat >= 0), "All matrices must be positive"

        if acc_mat.dtype == np.int64:
            # Raw counts
            probabilities = prob_mat.sum(axis=0).astype(np.float64)
            valid = probabilities > 0  # Avoid division by zero

            accuracies = acc_mat.sum(axis=0).astype(np.float64)
            accuracies[valid] = accuracies[valid] / probabilities[valid]
            confidences = conf_mat.sum(axis=0)
            confidences[valid] = confidences[valid] / probabilities[valid]
            probabilities[valid] = probabilities[valid] / np.sum(probabilities)
        else:
            # Normalized vectors --> approximate solution
            prob_mat = prob_mat + 1e-10  # Avoid division by zero

            accuracies = np.average(acc_mat, weights=prob_mat, axis=0)
            confidences = np.average(conf_mat, weights=prob_mat, axis=0)
            probabilities = np.sum(prob_mat, axis=0)
            valid = probabilities > 0
            probabilities[valid] = probabilities[valid] / np.sum(probabilities)

        ece_error = np.sum(np.abs(confidences[valid] - accuracies[valid]) * probabilities[valid]).item()

        return {
            "error": ece_error,
            "accuracies": accuracies,
            "confidences": confidences,
            "probabilities": probabilities,
        }
