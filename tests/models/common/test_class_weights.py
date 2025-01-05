# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch

from htc.models.common.class_weights import calculate_class_weights
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


def test_class_weights() -> None:
    mapping = LabelMapping({"a": 0, "b": 1}, last_valid_label_index=1)

    weights = calculate_class_weights(Config({"label_mapping": mapping, "model/class_weight_method": None}))
    assert torch.all(weights == torch.tensor([1, 1]))

    weights = calculate_class_weights(
        Config({"label_mapping": mapping, "model/class_weight_method": "(n-m)∕n"}),
        label_indices=torch.tensor([0, 1]),
        label_counts=torch.tensor([10, 5]),
    )
    assert pytest.approx(weights.sum()) == 1
    assert torch.all(weights == torch.tensor([(15 - 10) / 15, (15 - 5) / 15]))

    weights = calculate_class_weights(
        Config({"label_mapping": mapping, "model/class_weight_method": "1∕m"}),
        label_indices=torch.tensor([0, 1]),
        label_counts=torch.tensor([10, 5]),
    )
    assert pytest.approx(weights.sum()) == 1
    unnormalized_weights = torch.tensor([1 / 10, 1 / 5])
    assert torch.all(weights == unnormalized_weights / unnormalized_weights.sum())

    weights = calculate_class_weights(
        Config({"label_mapping": mapping, "model/class_weight_method": "softmin", "model/softmin_scaling": -2}),
        label_indices=torch.tensor([0, 1]),
        label_counts=torch.tensor([10, 5]),
    )
    assert pytest.approx(weights.sum()) == 1
    exp_counts = torch.exp(-2 * torch.tensor([10 / 15, 5 / 15]))
    assert torch.all(weights == exp_counts / exp_counts.sum())

    weights = calculate_class_weights(
        Config({"label_mapping": mapping, "model/class_weight_method": "nll"}),
        label_indices=torch.tensor([0, 1]),
        label_counts=torch.tensor([10, 5]),
    )
    nll = -torch.log(torch.tensor([10 / 15, 5 / 15]))
    assert pytest.approx(weights) == nll

    with pytest.raises(ValueError, match="not a valid class weight calculation method"):
        calculate_class_weights(
            Config({"label_mapping": mapping, "model/class_weight_method": "invalid_method"}),
            label_indices=torch.tensor([0, 1]),
            label_counts=torch.tensor([10, 5]),
        )
