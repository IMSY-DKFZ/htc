# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
from lightning import seed_everything

from htc.cpp import hierarchical_bootstrapping, hierarchical_bootstrapping_labels


def test_hierarchical_bootstrapping() -> None:
    seed_everything(0)

    # First camera with only one pig and one image
    image_index = 0
    cam_1 = {0: [image_index]}
    image_index += 1

    cam_2 = {}
    for subject_index in range(1, 11):
        cam_2[subject_index] = []
        for _ in range(10):
            cam_2[subject_index].append(image_index)
            image_index += 1

    mapping1 = {
        0: cam_1,
        1: cam_2,
    }
    mapping2 = {
        10: cam_1,
        1: cam_2,
    }

    for mapping in [mapping1, mapping2]:
        bootstraps = hierarchical_bootstrapping(mapping, n_subjects=10, n_images=10)
        assert bootstraps.shape == (1000, 2 * 10 * 10), "(n_bootstraps, n_cams * n_subjects * n_images)"

        _, counts = bootstraps.unique(return_counts=True)
        assert counts[0] == 1000 * 100
        assert counts[1:].float().mean() == 1000
        assert 800 <= counts[1:].min() <= counts[1:].max() <= 1200


def test_hierarchical_bootstrapping_labels() -> None:
    seed_everything(0)

    domain_mapping = {
        0: {
            0: [0, 1],
            1: [2],
        },
        1: {
            2: [3, 4],
        },
    }
    label_mapping = {
        0: [0, 1, 3],
        1: [1, 2, 4],
    }

    label0 = {0, 1, 3}
    label1 = {1, 2, 4}

    domain0 = {0, 1, 2}
    domain1 = {3, 4}

    bootstraps = hierarchical_bootstrapping_labels(domain_mapping, label_mapping, n_labels=2, n_bootstraps=10)
    assert torch.all(bootstraps.unique() == torch.arange(5)), "Every image should be used"

    col1 = set(bootstraps[:, [0, 2]].flatten().tolist())
    col2 = set(bootstraps[:, [1, 3]].flatten().tolist())
    if col1.issubset(domain0):
        assert col2.issubset(domain1)
    elif col1.issubset(domain1):
        assert col2.issubset(domain0)
    else:
        raise ValueError("col1 must either be a subset of domain0 or domain1")

    for row in range(bootstraps.size(0)):
        first_block = bootstraps[row, :2]
        second_block = bootstraps[row, 2:]
        assert first_block.unique().shape == second_block.unique().shape == (2,), (
            "There must always be two different images because we always select both domains (and there is no image"
            " overlap between domains)"
        )

        # For each block, the labels must either come from label0 or label1
        first_labels = set(first_block.tolist())
        second_labels = set(second_block.tolist())
        assert first_labels.issubset(label0) or first_labels.issubset(label1)
        assert second_labels.issubset(label0) or second_labels.issubset(label1)

    # Invalid label mapping
    label_mapping = {
        0: [0, 1, 3],
        1: [1, 2],
    }
    with pytest.raises(AssertionError, match="Label 1 is not present in all domains"):
        hierarchical_bootstrapping_labels(domain_mapping, label_mapping, n_labels=2, n_bootstraps=10)

    domain_mapping = {
        0: {
            0: [0, 1, 2],
            1: [3],
            2: [4],
        },
    }
    label_mapping = {
        100: [0, 1, 2, 4],
        200: [0, 1, 2, 4],
        300: [3],
    }

    bootstraps = hierarchical_bootstrapping_labels(domain_mapping, label_mapping, n_labels=4, n_bootstraps=10)
    assert (bootstraps == 3).sum() < 20

    bootstraps = hierarchical_bootstrapping_labels(
        domain_mapping, label_mapping, n_labels=4, n_bootstraps=10, oversampling=True
    )
    assert (bootstraps == 3).sum() == 20, "Label 3 should be represented half of the times"
