# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch
from torchmetrics.functional import accuracy, confusion_matrix

from htc.evaluation.metrics.scores import accuracy_from_cm, confusion_matrix_to_predictions, dice_from_cm


def test_dice_from_cm() -> None:
    target = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0])
    predictions = torch.tensor([1, 1, 1, 0, 0, 1, 0, 0])

    cm = confusion_matrix(predictions, target, task="multiclass", num_classes=2)
    assert torch.all(cm == torch.tensor([[3, 2], [1, 2]]))

    dice = dice_from_cm(cm.numpy())
    assert pytest.approx(dice) == np.mean([6 / (6 + 1 + 2), 4 / (4 + 1 + 2)])

    # Only classes which occur in the targets are relevant
    target = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0])
    predictions = torch.tensor([1, 1, 1, 0, 0, 1, 0, 2])

    cm = confusion_matrix(predictions, target, task="multiclass", num_classes=4)
    dice1 = dice_from_cm(cm.numpy())

    target = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0])
    predictions = torch.tensor([1, 1, 1, 0, 0, 1, 0, 3])
    # 2 or 3 doesn't matter since the target vector does not contain these classes

    cm = confusion_matrix(predictions, target, task="multiclass", num_classes=4)
    dice2 = dice_from_cm(cm.numpy())
    assert pytest.approx(dice1) == dice2


@pytest.mark.serial
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_confusion_matrix_to_predictions(device: str) -> None:
    cm = torch.tensor([[1, 2], [0, 4]], dtype=torch.int64, device=device)
    predictions, labels = confusion_matrix_to_predictions(cm)
    assert predictions.device.type == labels.device.type == device
    assert len(predictions) == len(labels) == 7
    assert torch.all(predictions == torch.tensor([0, 1, 1, 1, 1, 1, 1], device=device))
    assert torch.all(labels == torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device))

    assert accuracy(preds=predictions, target=labels, num_classes=2, task="multiclass") == accuracy_from_cm(cm)
