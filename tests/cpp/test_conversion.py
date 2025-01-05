# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from htc.cpp import tensor_mapping


def test_conversion() -> None:
    arr_np = np.array([1, 2, 3])
    mapping = {1: 10}
    arr_mapped_np = tensor_mapping(arr_np, mapping)

    assert type(arr_mapped_np) == np.ndarray
    assert np.all(arr_mapped_np == np.array([10, 2, 3]))

    arr_torch = torch.from_numpy(arr_np)
    arr_mapped_torch = tensor_mapping(arr_torch, mapping=mapping)
    arr_mapped_torch2 = tensor_mapping(tensor=arr_torch, mapping=mapping)
    assert torch.all(arr_mapped_torch == arr_mapped_torch2)

    assert type(arr_mapped_torch) == torch.Tensor
    assert torch.all(arr_mapped_torch == torch.Tensor([10, 2, 3]))
