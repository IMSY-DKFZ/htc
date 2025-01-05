# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch

from htc.evaluation.metrics.BrierScore import BrierScore


class TestBrierScore:
    def test_binary(self) -> None:
        brier_score = BrierScore(n_classes=2, variant="binary")

        predictions = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([1, 0, 1], dtype=torch.int64)
        assert brier_score(predictions, labels) == pytest.approx(0)

        predictions = torch.tensor(
            [
                [0.1, 0.9],
                [1, 0],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        assert brier_score(predictions, labels) == pytest.approx(0.01 / 3)

        predictions = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 0], dtype=torch.int64)
        assert brier_score(predictions, labels) == pytest.approx(1)

    def test_multiclass(self) -> None:
        brier_score = BrierScore(n_classes=3, variant="multiclass")

        predictions = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([2, 1, 0], dtype=torch.int64)
        assert brier_score(predictions, labels) == pytest.approx(0)

        predictions = torch.tensor(
            [
                [0, 0.2, 0.8],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=torch.float32,
        )
        assert brier_score(predictions, labels) == pytest.approx(0.08 / 3)
        assert (
            2 * brier_score(predictions[:2], labels[:2]) + 1 * brier_score(predictions[2:], labels[2:])
        ) / 3 == pytest.approx(0.08 / 3)

        predictions = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([1, 2, 1], dtype=torch.int64)

        assert brier_score(predictions, labels) == pytest.approx(2)
