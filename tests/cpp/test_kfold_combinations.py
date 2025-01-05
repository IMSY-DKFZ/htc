# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np

from htc.cpp import kfold_combinations


def test_kfold_combinations() -> None:
    min_labels = 0
    subject_indices = [0, 1, 2, 3, 4, 5]
    subject_labels = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2], 5: [0, 1, 2]}
    n_groups = 2
    folds = kfold_combinations(subject_indices, subject_labels, min_labels, n_groups)

    expected_combinations = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 3, 2, 4, 5],
        [0, 1, 4, 2, 3, 5],
        [0, 1, 5, 2, 3, 4],
        [0, 2, 3, 1, 4, 5],
        [0, 2, 4, 1, 3, 5],
        [0, 2, 5, 1, 3, 4],
        [0, 3, 4, 1, 2, 5],
        [0, 3, 5, 1, 2, 4],
        [0, 4, 5, 1, 2, 3],
    ]

    assert np.array(folds).shape == (
        10,
        6,
    ), "6 elements, a group size of 3 and 2 groups in total should yield 10 different kfold combinations"
    assert folds == expected_combinations
