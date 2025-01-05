# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch

from htc.cpp import tensor_mapping


class TestTensorMapping:
    def test_mapping(self) -> None:
        # Small tensor
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        mapping = {1: 10}
        tensor_mapping(tensor, mapping)  # The mapping happens in-place

        assert torch.all(tensor == torch.tensor([10, 2, 3], dtype=torch.int64))

        # Some larger tensor
        for dtype in [torch.int32, torch.int64, torch.float32]:
            tensor = torch.randint(0, 100, (5, 100, 800), dtype=dtype)
            tensor_original = tensor.clone()

            if tensor.is_floating_point():
                mapping = {float(i): float(i + 1) for i in range(100)}
            else:
                mapping = {i: i + 1 for i in range(100)}

            tensor_mapped = tensor_mapping(tensor, mapping)

            assert tensor.dtype == dtype
            assert torch.all(tensor == tensor_mapped), "The mapping should happen in-place"
            assert torch.all(tensor_original + 1 == tensor)

    def test_errors(self) -> None:
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        mapping = {"1": "10"}

        with pytest.raises(ValueError):
            tensor_mapping(tensor, mapping)

        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        mapping = {1: 10}

        with pytest.raises(AssertionError):
            tensor_mapping(tensor, mapping)

        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        mapping = {1.0: 10.0}

        with pytest.raises(AssertionError):
            tensor_mapping(tensor, mapping)
