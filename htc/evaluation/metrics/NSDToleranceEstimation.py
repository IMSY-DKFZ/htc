# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import warnings
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics.utils import get_mask_edges, get_surface_distance


class NSDToleranceEstimation:
    def __init__(self, n_classes: int, n_groups: int = 1, distance_metric: str = "euclidean"):
        """
        Helper class which can be used to estimate the tolerances for the normalized surface distance (NSD) based on annotations from two annotators. The general idea of this class is to iteratively add pairs of segmentations (from the two annotators), collect all distances on the way and then derive a tolerance per class based on all distances and an user defined reduction method (e.g. mean).

        This class also supports grouping of distances, e.g. to collect distances for each pigs separately. To estimate the thresholds, distances are first aggregated per pig and then the aggregated results are averaged across pigs.

        >>> estimator = NSDToleranceEstimation(n_classes=2)
        >>> estimator.add_image(torch.tensor([[0, 1, 1]]), torch.tensor([[0, 0, 1]]), torch.ones(1, 3, dtype=torch.bool))
        >>> estimator.distances[0]
        array([array([0., 0., 1.]), array([1., 0., 0.])], dtype=object)
        >>> estimator.add_image(torch.tensor([[0, 0, 1]]), torch.tensor([[0, 1, 1]]), torch.ones(1, 3, dtype=torch.bool))
        >>> estimator.distances[0]
        ... # doctest: +NORMALIZE_WHITESPACE
        array([array([0., 0., 1., 0., 1., 0.]), array([1., 0., 0., 0., 1., 0.])],
        dtype=object)
        >>> estimator.class_tolerances(np.mean)[0]
        [0.3333333333333333, 0.3333333333333333]

        Args:
            n_classes: The number of classes in the dataset.
            n_groups: The number of groups (pigs) to consider. n_groups = 1 means no grouping so that all distances are aggregated at once.
            distance_metric: The distance metric between two edge points to use.
        """
        self.distances = np.empty((n_groups, n_classes), dtype=object)
        for g, c in np.ndindex(self.distances.shape):
            # We use numpy arrays to append the different distances
            self.distances[g, c] = np.empty(0, dtype=np.float64)

        self.distance_metric = distance_metric

    def add_image(self, seg1: torch.Tensor, seg2: torch.Tensor, mask: torch.Tensor, group_index: int = 0) -> None:
        """
        Adds an annotation pair to the current estimator.

        Args:
            seg1: Annotation of the first annotator (height, width).
            seg2: Annotation of the second annotator (height, width).
            mask: Pixels to include (height, width).
            group_index: The zero-index of the current group.
        """
        distances = self._distances_image(seg1, seg2, mask)
        for c, d in enumerate(distances):
            assert (
                c < self.distances.shape[1]
            ), f"The segmentations contain a label with the id {c} but only {self.distances.shape[1]} classes are used"

            if d is not None:
                self.distances[group_index, c] = np.append(self.distances[group_index, c], d)

    def class_tolerances(self, reduction_func: Callable[[np.ndarray], float]) -> tuple[list[float], list[float]]:
        """
        Calculates the final class tolerances by taking all distances per class and applies the reduction function.

        Args:
            reduction_func: User defined reduction function.

        Returns: Estimated tolerance for each class.
        """
        tolerances = []
        tolerances_std = []
        for c in range(self.distances.shape[1]):
            # For this class, reduce (e.g. median) the distances for each pig separately
            group_distances = []
            for g in range(self.distances.shape[0]):
                distances = self.distances[g, c]
                if distances.size > 0:
                    group_distances.append(reduction_func(distances))

            # Average the group distances
            if len(group_distances) > 0:
                tolerances.append(np.mean(group_distances))
                tolerances_std.append(np.std(group_distances))
            else:
                tolerances.append(None)
                tolerances_std.append(None)

        assert len(tolerances) == self.distances.shape[1] and len(tolerances) == len(tolerances_std)
        return tolerances, tolerances_std

    def _distances_image(self, seg1: torch.Tensor, seg2: torch.Tensor, mask: torch.Tensor) -> list[np.ndarray]:
        assert seg1.shape == seg2.shape and seg2.shape == mask.shape, "All tensors must have the same shape"
        assert seg1.dim() == 2, "The tensors must be two-dimensional (height, width)"
        assert seg1.dtype == torch.int64 and seg2.dtype == torch.int64, "Segmentation masks must have long type"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

        # Copy the tensors since we need to modify them for the masking
        seg1 = seg1.clone()
        seg2 = seg2.clone()

        # The invalid labels are assigned a new dummy class which does not influence the calculation
        invalid_label_index = max(seg1[mask].max(), seg2[mask].max()) + 1
        seg1[~mask] = invalid_label_index
        seg2[~mask] = invalid_label_index
        n_labels = invalid_label_index + 1

        # Make one-hot encodings
        seg1_hot = F.one_hot(seg1, num_classes=n_labels)
        seg2_hot = F.one_hot(seg2, num_classes=n_labels)

        class_distances = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(ground truth|prediction) is all 0, this may result in nan/inf distance\.",
                category=UserWarning,
            )

            for c in range(n_labels - 1):  # The last class (invalid_label_index) is ignored
                (edges_pred, edges_gt) = get_mask_edges(seg1_hot[..., c], seg2_hot[..., c], always_return_as_numpy=True)

                # Calculate the distances in both directions for symmetry
                distances_pred_gt = get_surface_distance(edges_pred, edges_gt, distance_metric=self.distance_metric)
                distances_gt_pred = get_surface_distance(edges_gt, edges_pred, distance_metric=self.distance_metric)
                distances = np.concatenate([distances_pred_gt, distances_gt_pred])
                distances = distances[~np.isinf(distances)]  # If a class does not occur at all, it is ignored

                class_distances.append(distances if distances.size > 0 else None)

        return class_distances
