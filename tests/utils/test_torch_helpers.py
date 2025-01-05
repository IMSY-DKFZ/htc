# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
from lightning import seed_everything

from htc.models.common.torch_helpers import (
    cpu_only_tensor,
    group_mean,
    minmax_pos_neg_scaling,
    pad_tensors,
    smooth_one_hot,
)


class TestPadTensors:
    def test_default(self) -> None:
        tensor_a = torch.ones(2, 2, dtype=torch.int32)
        tensor_b = torch.ones(3, 3, dtype=torch.int32)

        stacked_ab = torch.stack(pad_tensors([tensor_a, tensor_b]))
        assert stacked_ab.dtype == torch.int32
        assert torch.all(
            stacked_ab
            == torch.tensor([
                [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ])
        )

        stacked_value_ab = torch.stack(pad_tensors([tensor_a, tensor_b], pad_value=10))
        assert stacked_value_ab.dtype == torch.int32
        assert torch.all(
            stacked_value_ab
            == torch.tensor([
                [[1, 1, 10], [1, 1, 10], [10, 10, 10]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ])
        )

    def test_size(self) -> None:
        tensor_a = torch.ones(12, 34)
        tensor_b = torch.ones(28, 29)
        stacked_ab = torch.stack(pad_tensors([tensor_a, tensor_b], size_multiple=(32, 32)))
        assert stacked_ab.shape == (2, 32, 64)
        assert torch.all(stacked_ab[0, 12:, 34:].unique() == torch.tensor([0]))
        assert torch.all(stacked_ab[0, 28:, 29:].unique() == torch.tensor([0]))

    def test_dim(self) -> None:
        tensor_a = torch.ones(20, 32, 64, 100)
        tensor_b = torch.ones(20, 32, 32, 100)
        stacked_ab = torch.stack(pad_tensors([tensor_a, tensor_b], dim=(1, 2), size_multiple=32))
        assert stacked_ab.shape == (2, 20, 32, 64, 100)
        assert torch.all(stacked_ab[1, :, :, 32:, :].unique() == torch.tensor([0]))

        stacked_ab = torch.stack(pad_tensors([tensor_a, tensor_b], dim=2, size_multiple=32))
        assert stacked_ab.shape == (2, 20, 32, 64, 100)


def test_smooth_one_hot() -> None:
    seed_everything(42)
    labels = torch.randint(0, 5, (10, 480, 640))
    labels_smooth = smooth_one_hot(labels, 5, 0.1)
    assert labels_smooth.shape == (10, 480, 640, 5)
    assert torch.all(labels_smooth >= 0) and torch.all(labels_smooth <= 1)

    class_sum = labels_smooth.sum(dim=-1)
    assert torch.allclose(class_sum, torch.ones_like(class_sum))


def test_group_mean() -> None:
    indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    values = torch.tensor([0.1, 0.2, 1, 1.5, 0.5], dtype=torch.float32)

    agg_indices, agg_values = group_mean(indices, values)
    assert torch.all(agg_indices == torch.tensor([0, 1]))
    assert torch.allclose(agg_values, torch.tensor([0.15, 1]))

    indices = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64)
    values = torch.tensor([0.1, 0.2, 1, 1.5, 0.5], dtype=torch.float32)

    agg_indices, agg_values = group_mean(indices, values)
    assert torch.all(agg_indices == torch.tensor([0, 2]))
    assert torch.allclose(agg_values, torch.tensor([0.15, 1]), equal_nan=True)


def test_minmax_pos_neg_scaling() -> None:
    x = torch.tensor(
        [
            [-1, 2, -3],
            [4, -5, 6],
        ],
        dtype=torch.float32,
    )

    x0_scaled = torch.tensor(
        [
            [-1 / 3, 1, -1],
            [4 / 6, -1, 1],
        ],
        dtype=torch.float32,
    )
    x1_scaled = torch.tensor(
        [
            [-1, 1, -1],
            [1, -1, 1],
        ],
        dtype=torch.float32,
    )

    assert torch.all(minmax_pos_neg_scaling(x, dim=0) == x0_scaled)
    assert torch.all(minmax_pos_neg_scaling(x, dim=1) == x1_scaled)

    for dim, mindim1, mindim2 in [(0, 1, 1), (1, 0, 1), (2, 0, 0)]:
        torch.manual_seed(42)
        x = torch.rand(10, 12, 20) - 0.5
        x_scaled = minmax_pos_neg_scaling(x, dim=dim)
        assert x_scaled.shape == (10, 12, 20)
        assert torch.all(x_scaled >= -1) and torch.all(x_scaled <= 1)

        x_mins = x_scaled.min(dim=mindim1).values.min(dim=mindim2).values
        x_maxs = x_scaled.max(dim=mindim1).values.max(dim=mindim2).values
        assert len(x_mins) == x.shape[dim]
        assert len(x_maxs) == x.shape[dim]
        assert torch.allclose(x_mins, torch.full_like(x_mins, -1))
        assert torch.allclose(x_maxs, torch.full_like(x_maxs, 1))


@pytest.mark.serial
def test_cpu_only_tensor() -> None:
    tensor = torch.tensor([1, 2, 3])
    cpu_only_tensor(tensor)

    tensor = tensor.cuda()
    assert tensor.device == torch.device("cpu")
    tensor = tensor.to("cuda")
    assert tensor.device == torch.device("cpu")
    tensor = tensor.to(device="cuda")
    assert tensor.device == torch.device("cpu")

    tensor = torch.tensor([1, 2, 3], device="cuda")
    with pytest.raises(AssertionError):
        cpu_only_tensor(tensor)
