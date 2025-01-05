# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np
import torch

from htc.evaluation.metrics.NSDToleranceEstimation import NSDToleranceEstimation


class TestNSDToleranceEstimation:
    def test_distances(self) -> None:
        seg1 = torch.zeros(10, 10, dtype=torch.int64)
        seg1[:, 5:] = 2
        seg2 = torch.zeros(10, 10, dtype=torch.int64)
        seg2[:, 6:] = 2
        mask = torch.ones(10, 10, dtype=torch.bool)

        class_distances = NSDToleranceEstimation(n_classes=3)._distances_image(seg1, seg2, mask)
        assert len(class_distances) == 3
        assert np.quantile(class_distances[0], q=0.95) == 1.0
        assert class_distances[1] is None
        assert np.quantile(class_distances[2], q=0.95) == 1.0

    def test_distances_mask(self) -> None:
        seg1 = torch.zeros(10, 10, dtype=torch.int64)
        seg1[:, 5:] = 1
        seg2 = torch.zeros(10, 10, dtype=torch.int64)
        seg2[:, 6:] = 1
        mask = torch.ones(10, 10, dtype=torch.bool)
        mask[:, 5:6] = 0

        class_distances = NSDToleranceEstimation(n_classes=2)._distances_image(seg1, seg2, mask)
        assert len(class_distances) == 2
        assert np.all(class_distances[0] == 0.0)
        assert np.all(class_distances[1] == 0.0)

    def test_multiple_images(self) -> None:
        seg1 = torch.zeros(10, 10, dtype=torch.int64)
        seg1[:, 5:] = 2
        seg2 = torch.zeros(10, 10, dtype=torch.int64)
        seg2[:, 6:] = 2
        mask = torch.ones(10, 10, dtype=torch.bool)

        estimator = NSDToleranceEstimation(n_classes=3)
        estimator.add_image(seg1, seg2, mask)
        n1 = estimator.distances[0].size

        seg1[:, 5:] = 1
        seg2[:, 6:] = 1
        estimator.add_image(seg1, seg2, mask)
        assert np.all([x.ndim == 1 for x in estimator.distances[0]])
        assert np.all([x.size > 0 for x in estimator.distances[0]])
        assert estimator.distances[0, 0].size > n1

        tolerances, tolerances_std = estimator.class_tolerances(partial(np.quantile, q=0.95))
        assert all(t == 1.0 for t in tolerances)
        assert all(t == 0.0 for t in tolerances_std)

    def test_groups(self) -> None:
        seg1 = torch.zeros(10, 10, dtype=torch.int64)
        seg1[:, 5:] = 1
        seg2 = torch.zeros(10, 10, dtype=torch.int64)
        seg2[:, 5:] = 1
        mask = torch.ones(10, 10, dtype=torch.bool)

        estimator = NSDToleranceEstimation(n_classes=2, n_groups=2)
        estimator.add_image(seg1, seg2, mask, group_index=0)
        seg2[:, 4:] = 1
        estimator.add_image(seg1, seg2, mask, group_index=1)
        assert np.all(estimator.distances[0, 0] == 0) and np.all(estimator.distances[0, 1] == 0)
        assert np.all(np.unique(estimator.distances[1, 0]) == np.array([0, 1])) and np.all(
            np.unique(estimator.distances[1, 1]) == np.array([0, 1])
        )
