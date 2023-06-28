# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Union

import numpy as np
import torch
import torch.nn as nn


class CalibrationLoss(nn.Module):
    def _parse_input(
        self, predictions: torch.Tensor, labels: torch.Tensor, confidences: torch.Tensor = None
    ) -> torch.Tensor:
        if predictions.is_floating_point():
            # Softmax input
            assert predictions.ndim >= 2, "Softmax values should be at least 2-dimensional"
            assert (
                predictions.shape[1:] == labels.shape[0:]
            ), "The sample dimension does not match between softmaxes and labels"

            softmaxes_sum = torch.sum(predictions, dim=0)
            assert torch.allclose(softmaxes_sum, torch.ones_like(softmaxes_sum)), (
                "All of the softmax should sum upto approx. 1 in the class dimension. Are you sure that you are not"
                f" sending logits rather than softmaxes? {softmaxes_sum.unique(return_counts=True) = }"
            )

            confidences, predictions = torch.max(predictions, dim=0)
            confidences = confidences.flatten()
            predictions = predictions.flatten()
            labels = labels.flatten()

            assert predictions.shape == labels.shape, "The sample dimension does not match between softmaxes and labels"
            accuracies = predictions.eq(labels).to(dtype=confidences.dtype)
            assert len(confidences.shape) == 1, "Confidences should be a vector"
        else:
            # Predicted labels input
            assert confidences is not None, "Confidences must be provided if predicted labels are passed"
            assert predictions.shape == labels.shape, "The sample dimension does not match between softmaxes and labels"
            accuracies = predictions.eq(labels).to(dtype=confidences.dtype)

            confidences = confidences.flatten()
            accuracies = accuracies.flatten()

        return confidences, accuracies


class ECE(CalibrationLoss):
    def __init__(self, n_bins: int = 10):
        """
        Calculates the Expected Calibration Error (ECE) of a model.

        This implementation is in line with the [multiclass_calibration_error()](https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html#multiclass-calibration-error) function from torchmetrics but returns the raw data (e.g. accuracy counts) in addition to the ECE error. This allows for aggregation across multiple samples (e.g. for a whole dataset) using the `aggregate_vectors()` method.

        This metric can be used to estimate how well a model is calibrated with lower values indicating a better calibrated model (for more information see https://arxiv.org/abs/1706.04599). For this, it takes the predictions of the network (the softmax values) where the highest value in the softmax vector corresponds to the confidence for that prediction. Then, the predictions are binned into confidence intervals and for each bin the difference between the average confidence and accuracy is calculated. A model is calibrated when this difference is small. The intuition is that if there are, for example, 100 samples in the bin between 0.7 and 0.8 with an average confidence of 0.77, then around 77 % of those samples should be correctly classified assuming a classifier which knows about its unsureness.

        Note: The model's calibration can be improved by using the temperature scaling method (e.g. https://github.com/gpleiss/temperature_scaling).

        Args:
            n_bins: Number of confidence interval bins.
        """
        super().__init__()
        self.n_bins = n_bins

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        confidences: torch.Tensor = None,
    ) -> dict[str, Union[float, list[int], list[float], list[int]]]:
        """
        Calculates the ECE values for a set of samples.

        Args:
            predictions: Softmax predictions of the network (class, *) or predicted labels of the network (*).
            labels: Reference labels (*).
            confidences: Confidence values of the predictions (*). Must not be None if predicted labels are passed.

        Returns:
            dict: Dictionary with the ECE "error" plus the unnormalized vectors ("accuracies", "confidences", "probabilities"). Please refer to the `aggregate_vectors()` method if you need to aggregate those values across multiple samples.
        """
        confidences, accuracies = self._parse_input(predictions, labels, confidences)
        accuracies = accuracies.to(torch.int32)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=predictions.device)

        accuracies_bins = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=torch.int32)
        confidences_bins = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)
        counts_bins = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=torch.int32)

        # For each value, the index of the corresponding bucket
        indices = torch.bucketize(confidences, bin_boundaries) - 1

        # Add every value to the corresponding bucket
        counts_bins.scatter_add_(dim=0, index=indices, src=torch.ones_like(accuracies))
        accuracies_bins.scatter_add_(dim=0, index=indices, src=accuracies)
        confidences_bins.scatter_add_(dim=0, index=indices, src=confidences)

        # Normalize via number of samples per bin
        # Setting nan values to zero basically ignores those bins in the ECE calculation
        prob_normalized = counts_bins / counts_bins.sum()
        acc_normalized = torch.nan_to_num_(accuracies_bins / counts_bins)
        conf_normalized = torch.nan_to_num_(confidences_bins / counts_bins)

        # ECE is the absolute difference between the confidences and accuracies in the bin weighted by the number of samples which are in this bin
        ece = torch.sum(torch.abs(conf_normalized - acc_normalized) * prob_normalized)

        return {
            "error": ece.item(),
            "accuracies": accuracies_bins.tolist(),
            "confidences": confidences_bins.tolist(),
            "probabilities": counts_bins.tolist(),
        }

    @staticmethod
    def aggregate_vectors(
        acc_mat: np.ndarray, conf_mat: np.ndarray, prob_mat: np.ndarray
    ) -> dict[str, Union[float, list[int], list[float], list[int]]]:
        """
        This function aggregates the ece vectors from multiple images. This is useful when the ece cannot be calculated for all samples at once. All matrices must have the shape (n_batches, n_bins). Note that only when the raw counts are passed to this function the real ece can be calculated. If normalized vectors are passed, then the assumption is made that the original number of samples before the normalization was the same for all images (which is an approximation).

        Args:
            acc_mat: Matrix with either the counts representing the number of correctly classified samples per bin or the accuracy per bin (n_batches, n_bins).
            conf_mat: Matrix with either the sum of the confidence values or the normalized confidence per bin (n_batches, n_bins).
            prob_mat: Matrix with either the total counts or the ratio of samples per bin (n_batches, n_bins).

        Returns: A dictionary with "accuracies", "confidences" and "probabilities" vectors (all normalized) as well as the ece "error".
        """
        assert (
            acc_mat.shape == conf_mat.shape and acc_mat.shape == prob_mat.shape
        ), "All matrices must have the same shape"
        assert np.all(acc_mat >= 0) and np.all(conf_mat >= 0) and np.all(prob_mat >= 0), "All matrices must be positive"

        with np.errstate(invalid="ignore"):
            if acc_mat.dtype == np.int64:
                # Raw counts
                probabilities = prob_mat.sum(axis=0).astype(np.float64)

                accuracies = acc_mat.sum(axis=0).astype(np.float64)
                accuracies = np.nan_to_num(accuracies / probabilities, copy=False)
                confidences = conf_mat.sum(axis=0)
                confidences = np.nan_to_num(confidences / probabilities, copy=False)
                probabilities = np.nan_to_num(probabilities / np.sum(probabilities), copy=False)
            else:
                # Normalized vectors --> approximate solution
                prob_mat = prob_mat + 1e-10  # Avoid division by zero

                accuracies = np.average(acc_mat, weights=prob_mat, axis=0)
                confidences = np.average(conf_mat, weights=prob_mat, axis=0)
                probabilities = np.sum(prob_mat, axis=0)
                probabilities = np.nan_to_num(probabilities / np.sum(probabilities), copy=False)

        ece_error = np.sum(np.abs(confidences - accuracies) * probabilities).item()

        return {
            "error": ece_error,
            "accuracies": accuracies,
            "confidences": confidences,
            "probabilities": probabilities,
        }
