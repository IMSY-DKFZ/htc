# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Union

import numpy as np
import torch
from torchmetrics.functional import confusion_matrix


def dice_from_cm(cm: np.ndarray) -> float:
    """
    Calculates the dice metric (f1 score) based on the confusion matrix. Only classes which occur in the targets are considered (mispredictions to other classes are ignored).

    Args:
        cm: Confusion matrix. The rows must denote the targets and the columns the predictions.

    Returns: Dice metric in the range [0;1]
    """
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    numerator = tp * 2.0
    denominator = tp * 2.0 + fn + fp
    used_labels = cm.sum(axis=1) > 0

    return np.mean(numerator[used_labels] / denominator[used_labels])


def confusion_matrix_groups(
    predictions: torch.Tensor, labels: torch.Tensor, image_names: list[str], n_classes: int
) -> dict[str, torch.Tensor]:
    """
    Calculates a confusion matrix for each pig group as defined by the image identifier.

    >>> predictions = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    >>> labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    >>> image_names = ['P001#1', 'P001#2', 'P002#1', 'P002#2']
    >>> confusion_matrix_groups(predictions, labels, image_names, n_classes=2)
    {'P001': tensor([[1, 1],
            [0, 0]]), 'P002': tensor([[0, 0],
            [1, 1]])}

    Args:
        predictions: List of predicted labels.
        labels: List of reference labels.
        image_names: Image identifier for each sample in the format subject_name#other#ids (the subject_name must be the first identifier).
        n_classes: Number of classes in the dataset (defines the shape of the confusion matrix).

    Returns: Confusion matrix for each pig.
    """
    assert len(predictions) == len(labels) and len(predictions) == len(
        image_names
    ), "The tensors must agree in the number of samples"

    subject_names = np.array([image_name.split("#")[0] for image_name in image_names])
    result = {}

    for subject_name in np.unique(subject_names):
        result[subject_name] = confusion_matrix(
            predictions[subject_names == subject_name],
            labels[subject_names == subject_name],
            task="multiclass",
            num_classes=n_classes,
        )

    return result


def normalize_grouped_cm(cm_groups: np.ndarray) -> tuple[np.ndarray]:
    """
    Confusion matrix normalization accounting for the hierarchical structure in the data. It first normalizes each confusion matrix per group (e.g. pig) (row-wise, i.e. calculating recall/sensitivity) and then averages the group result yielding one final confusion matrix.

    >>> cm_groups = np.array([
    ...    [[15, 5],
    ...     [1, 9]],
    ...    [[8, 2],
    ...     [3, 5]]
    ... ])
    >>> cm_mean, cm_std = normalize_grouped_cm(cm_groups)
    >>> cm_mean  # Average of the two normalized confusion matrices
    array([[0.775 , 0.225 ],
           [0.2375, 0.7625]])
    >>> cm_std  # Standard deviation of the two normalized confusion matrices
    array([[0.025 , 0.025 ],
           [0.1375, 0.1375]])

    Can also be used in conjunction with confusion_matrix_groups:
    >>> predictions = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    >>> labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    >>> image_names = ['P001#1', 'P001#2', 'P002#1', 'P002#2']
    >>> cm_groups = confusion_matrix_groups(predictions, labels, image_names, n_classes=2)
    >>> cm_groups = np.stack([cm.numpy() for cm in cm_groups.values()])
    >>> normalize_grouped_cm(cm_groups)[0]
    array([[0.5, 0.5],
           [0.5, 0.5]])

    Args:
        cm_groups: Confusion matrices (one per group) which contain the raw counts. First dimension corresponds to the group.

    Returns: Average confusion matrix across all groups and standard deviation of confusion matrices (e.g. average pig performance and deviation of the performance across pigs).
    """
    assert len(cm_groups.shape) == 3, "The confusion matrix should be three-dimensional (one cm per group)"
    assert cm_groups.shape[1] == cm_groups.shape[2], "The second and third dimension must match"

    with np.errstate(invalid="ignore"):
        # Normalize per group
        cm_groups = cm_groups / np.sum(cm_groups, axis=2, keepdims=True)  # Row-normalization (= recall/sensitivity)
    cm_std = np.nanstd(cm_groups, axis=0)
    cm_mean = np.nanmean(cm_groups, axis=0)  # nan values are ignored

    return cm_mean, cm_std


def accuracy_from_cm(cm: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculates the overall accuracy for a given confusion matrix (for all classes at once). Usually, the accuracy is defined as
    acc = (TP + TN) / (TP + TN + FP + FN)
    In a multiclass setting, TP + TN is simply the total number of correctly classified samples and the denominator is the total number of samples.

    >>> cm = torch.tensor([[10, 5], [0, 5]], dtype=torch.int64)
    >>> accuracy_from_cm(cm)
    0.75

    Args:
        cm: Confusion matrix.

    Returns: The accuracy across all classes.
    """
    assert len(cm.shape) == 2, "The confusion matrix must be two-dimensional"

    if type(cm) == np.ndarray:
        return np.trace(cm) / np.sum(cm)
    elif type(cm) == torch.Tensor:
        return (cm.trace() / cm.sum()).item()
    else:
        raise ValueError("Invalid type")
