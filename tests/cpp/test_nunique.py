# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from htc.cpp import nunique
from htc.tivita.DataPath import DataPath


def test_nunique() -> None:
    x = torch.tensor(
        [
            [[1, 2], [3, 2]],
            [[5, 2], [2, 2]],
        ],
        dtype=torch.int64,
    )

    assert torch.all(nunique(x, dim=0) == torch.tensor([[2, 1], [2, 1]], dtype=torch.int64))
    assert torch.all(nunique(x.type(torch.int32), dim=0) == torch.tensor([[2, 1], [2, 1]], dtype=torch.int64))

    x = torch.tensor([[1.1, 1.1, 3], [2, 2.1, 2.2]], dtype=torch.float32)
    assert torch.all(nunique(x, dim=1) == torch.tensor([2, 3], dtype=torch.int64))

    x = torch.arange(100).repeat(10, 1)
    assert x.shape == (10, 100)
    assert torch.all(nunique(x, dim=0) == torch.ones(100, dtype=torch.int64))
    assert torch.all(nunique(x, dim=1) == 100 * torch.ones(10, dtype=torch.int64))

    assert nunique(torch.tensor([0, 1, 1])) == 2
    assert nunique(torch.tensor([[0, 1, 1], [0, 3, 1]])) == 3

    path = DataPath.from_image_name("P086#2021_04_15_11_38_04")
    seg = path.read_segmentation()
    assert nunique(seg) == len(np.unique(seg))

    x = torch.arange(10)[::2]
    assert not x.is_contiguous()
    assert nunique(x) == 5
