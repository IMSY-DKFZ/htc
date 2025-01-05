# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch

from htc.models.common.distance_correlation import distance_correlation


class TestDistanceCorrelation:
    def test_basics(self) -> None:
        x = torch.tensor([[1], [2], [3], [4], [4]])
        y = torch.tensor([[1], [2.2], [-3.3], [2], [2]])

        assert distance_correlation(x, x) == 1
        assert distance_correlation(y, y) == 1
        assert distance_correlation(x, y) == pytest.approx(0.5242320806029278)
        assert distance_correlation(x, y) == distance_correlation(y, x)

    def test_zero(self) -> None:
        x = torch.tensor([1], dtype=torch.float64)
        y = torch.tensor([1], dtype=torch.float64)

        dcor = distance_correlation(x, y)
        assert dcor == 0
        assert dcor.dtype == torch.float64

    def test_y_identical(self) -> None:
        x = torch.tensor([[1, 10], [1, 8], [3, 9]])
        y = torch.tensor([[1], [1], [1]])

        assert distance_correlation(x, y) == 0

    def test_y_one_hot(self) -> None:
        x = torch.tensor([[1, 10], [1, 8], [3, 9]])
        y1 = torch.tensor([[0, 1], [1, 0], [1, 0]])
        y2 = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 0, 0]])

        assert distance_correlation(x, y1) == distance_correlation(x, y2)

    @pytest.mark.serial
    def test_gpu(self) -> None:
        x = torch.tensor([[1], [2], [3], [4]])
        y = torch.tensor([[1], [2.2], [3.3], [2]])

        dcor_cpu = distance_correlation(x, y)
        dcor_gpu = distance_correlation(x.cuda(), y.cuda())
        assert dcor_gpu.is_cuda
        assert pytest.approx(dcor_cpu.item()) == 0.7763492388369893
        assert pytest.approx(dcor_cpu.item()) == dcor_gpu.item()
