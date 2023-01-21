# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import math
import warnings

import torch
import torch.nn.functional as F
from monai.metrics import SurfaceDistanceMetric, compute_dice, compute_surface_dice
from torchmetrics.functional import confusion_matrix

from htc.evaluation.metrics.ECELoss import ECELoss
from htc.settings import settings
from htc.settings_seg import settings_seg


def calc_surface_dice(
    predictions_labels: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, tolerances: list[float]
) -> list[dict]:
    """
    Calculate the normalized surface dice (NSD) with correct handling of the mask.

    Args:
        predictions_labels: Predicted labels of the model (batch, height, width).
        labels: Target labels (batch, height, width).
        mask: Pixels to include (batch, height, width).
        tolerances: The tolerance threshold in pixels for each class. All pixels in the distance of this threshold will count as correct. Higher numbers yield better results.

    Returns: Batch of dictionaries with labels of the image, surface dice per label and image surface dice score.
    """
    assert (
        predictions_labels.shape == labels.shape and predictions_labels.shape == mask.shape
    ), "All input tensors must have the same shape"
    assert predictions_labels.dim() == 3, "Each tensor must have three dimensions (batch, height, width)"
    assert (
        predictions_labels.dtype == torch.int64 and labels.dtype == torch.int64
    ), "Predictions and labels must be label index values"
    assert mask.dtype == torch.bool, "The mask must be a boolean tensor"
    assert all([t >= 0 for t in tolerances]), "The tolerance values must be non-negative"

    # Copy the tensors since we need to modify them for the masking
    predictions_labels = predictions_labels.clone()
    labels = labels.clone()

    # The invalid labels are assigned a new dummy class which does not influence the calculation
    invalid_label_index = max(predictions_labels[mask].max(), labels[mask].max()) + 1
    predictions_labels[~mask] = invalid_label_index
    labels[~mask] = invalid_label_index
    n_labels = invalid_label_index + 1
    assert len(tolerances) >= invalid_label_index, (
        f"There must be a threshold for each class (there are {invalid_label_index} classes but only"
        f" {len(tolerances)} tolerance values available)"
    )
    if len(tolerances) > invalid_label_index:
        # It is possible that classes get removed by the masking
        tolerances = tolerances[:invalid_label_index]

    # Make one-hot encodings
    labels_hot = F.one_hot(labels, num_classes=n_labels).permute(0, 3, 1, 2)  # [BCHW]
    predictions_hot = F.one_hot(predictions_labels, num_classes=n_labels).permute(0, 3, 1, 2)

    # SD per batch and class (similar to the average surface distance)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"the (ground truth|prediction) .* is all 0, this may result in nan/inf distance\.",
            category=UserWarning,
        )

        # + [0] as tolerance value for the invalid class (irrelevant since the class is removed later anyway)
        surface_dices = compute_surface_dice(
            predictions_hot, labels_hot, class_thresholds=tolerances + [0], include_background=True
        )
        surface_dices = surface_dices[:, :-1]  # Remove last dice value (corresponds to the mask class)
        surface_dices[torch.isnan(surface_dices)] = 0  # If a class is not present in the image, nans will be used

    batch_results = []
    for b in range(labels.shape[0]):
        used_labels = labels[b, mask[b]].unique()
        assert torch.all(used_labels < invalid_label_index)

        sd = surface_dices[b, used_labels]
        batch_results.append(
            {"used_labels": used_labels, "surface_dice_metric": sd, "surface_dice_metric_image": sd.mean().item()}
        )

    return batch_results


def calc_dice_metric(predictions_labels: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> list[dict]:
    """
    Helper function to compute the dice similarity coefficient for a images with correct handling of the mask.

    Args:
        predictions_labels: Predicted labels of the model (batch, height, width).
        labels: Target labels (batch, height, width).
        mask: Pixels to include (batch, height, width).

    Returns: Batch of dictionaries with labels of the image, dice per label and image dice score.
    """
    assert (
        predictions_labels.shape == labels.shape and predictions_labels.shape == labels.shape
    ), "All input tensors must have the same shape"
    assert predictions_labels.dim() == 3, "Each tensor must have three dimensions (batch, height, width)"
    assert (
        predictions_labels.dtype == torch.int64 and labels.dtype == torch.int64
    ), "Predictions and labels must be label index values"
    assert mask.dtype == torch.bool, "The mask must be a boolean tensor"

    # Copy the tensors since we need to modify them for the masking
    predictions_labels = predictions_labels.clone()
    labels = labels.clone()

    # Add mask class (will be removed later)
    invalid_label_index = max(predictions_labels[mask].max(), labels[mask].max()) + 1
    predictions_labels[~mask] = invalid_label_index
    labels[~mask] = invalid_label_index
    n_labels = invalid_label_index + 1

    # Make one-hot encodings
    labels_hot = F.one_hot(labels, num_classes=n_labels).permute(0, 3, 1, 2)
    predictions_hot = F.one_hot(predictions_labels, num_classes=n_labels).permute(0, 3, 1, 2)

    dice = compute_dice(
        y_pred=predictions_hot, y=labels_hot, include_background=True
    )  # The background can be excluded via the mask
    dice = dice[:, :-1]  # Remove last dice value (corresponds to the mask class)
    dice[torch.isnan(dice)] = 0  # If a class is not present in the image, nans will be used

    batch_results = []
    for b in range(labels.shape[0]):
        used_labels = labels[b, mask[b]].unique()
        assert torch.all(used_labels < invalid_label_index)

        d = dice[b, used_labels]
        batch_results.append({"used_labels": used_labels, "dice_metric": d, "dice_metric_image": d.mean().item()})

    return batch_results


def calc_surface_distance(predictions_labels: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> list[dict]:
    """
    Calculate the surface distance with correct handling of the mask.

    Args:
        predictions_labels: Predicted labels of the model (batch, height, width).
        labels: Target labels (batch, height, width).
        mask: Pixels to include (batch, height, width).

    Returns: Batch of dictionaries with labels of the image, surface distance per label and image surface distance score.
    """
    assert (
        predictions_labels.shape == labels.shape and predictions_labels.shape == labels.shape
    ), "All input tensors must have the same shape"
    assert predictions_labels.dim() == 3, "Each tensor must have three dimensions (batch, height, width)"
    assert (
        predictions_labels.dtype == torch.int64 and labels.dtype == torch.int64
    ), "Predictions and labels must be label index values"
    assert mask.dtype == torch.bool, "The mask must be a boolean tensor"

    # Copy the tensors since we need to modify them for the masking
    predictions_labels = predictions_labels.clone()
    labels = labels.clone()

    # The invalid labels are assigned a new dummy class which does not influence the calculation
    invalid_label_index = max(predictions_labels[mask].max(), labels[mask].max()) + 1
    predictions_labels[~mask] = invalid_label_index
    labels[~mask] = invalid_label_index
    n_labels = invalid_label_index + 1

    # Make one-hot encodings
    labels_hot = F.one_hot(labels, num_classes=n_labels).permute(0, 3, 1, 2)  # [BCHW]
    predictions_hot = F.one_hot(predictions_labels, num_classes=n_labels).permute(0, 3, 1, 2)

    # Distances for all classes
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"the (ground truth|prediction) of class \d+ is all 0, this may result in nan/inf distance\.",
            category=UserWarning,
        )  # We handle inf and nans ourself
        surface_distances = SurfaceDistanceMetric(symmetric=True, reduction="none", include_background=True)(
            predictions_hot, labels_hot
        )  # The background can be excluded via the mask

    # Average distance for each batch taking into account the relevant labels
    batch_results = []
    max_image_distance = math.sqrt(predictions_labels.size(1) ** 2 + predictions_labels.size(2) ** 2)
    for b in range(labels.shape[0]):
        used_labels = labels[b, mask[b]].unique()
        assert torch.all(used_labels < invalid_label_index)

        distances = surface_distances[b, used_labels]

        invalid_distances = torch.isnan(distances) | torch.isinf(distances)
        if torch.all(invalid_distances):
            # If we have only nan or inf values we set the distance to the maximal possible distance of the image size (the diagonal)
            distances[:] = max_image_distance
        else:
            # Set nan/inf values (structure to small or class did not occur at all) to the maximum value so that we can calculate a meaningful average value
            distances[invalid_distances] = distances[~invalid_distances].max()

        batch_results.append(
            {
                "used_labels": used_labels,
                "surface_distance_metric": distances,
                "surface_distance_metric_image": distances.mean().item(),
            }
        )

    return batch_results


def check_invalid_input(tensor: torch.Tensor, name: str):
    invalid_input = False
    if torch.any(torch.isnan(tensor)):
        invalid_input = True
        settings.log.warning(f"nan values detected in the tensor {name}")
    if torch.any(torch.isinf(tensor)):
        invalid_input = True
        settings.log.warning(f"inf values detected in the tensor {name}")

    if invalid_input:
        tensor.nan_to_num_()
        settings.log.info(f"nan_to_num applied on the tensor {name}")

    # we add this check here, as in case of invalid inputs, the vectors have to be re-normalized
    # otherwise they don't sum upto 1 (which is a condition for softmaxes)
    if invalid_input and len(tensor.shape) == 4:
        tensor = tensor.softmax(dim=1)
        settings.log.info(f"softmax applied to class dimension of the tensor {name}")

    return tensor


def evaluate_images(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    metrics: list = None,
    n_classes: int = None,
    tolerances: list[float] = None,
) -> list[dict]:
    """
    Evaluate all images in the batch dimension of the tensors.

    The goal is to provide a function which can be used by every model (even when validating during training) so that the validation is always the same. The metrics calculated by this function are the most common one and should always be included in the validation.

    Args:
        predictions: Predictions of the model either as labels (batch, height, width) or softmax (of logits) (batch, channel, height, width). ece loss is only calculated when providing the softmax.
        labels: Target labels (batch, height, width).
        mask: Pixels to include (batch, height, width). You can for example also use it to exclude specific classes (e.g. the background=0) in the calculation by setting the mask for these pixels to False.
        n_classes: Number of classes in the dataset. This value should be the same for every image in the dataset and is e.g. necessary for the confusion matrix where the shape is determined by the number of classes (and is only comparable if this number is the same for all images).
        metrics: A list of metrics to be calculated, this list can contain the following: DSC = dice similarity coefficient, CM = confusion matrix, ECE = expected calibration error, NSD = normalized surface dice, ASD = average surface distance. If None, DSC, ECE (if softmaxes are provided) and CM will be calculated.
        tolerances: Parameter for the NSD metric: the tolerance threshold in pixels for each class. All pixels in the distance of this threshold will count as correct. Higher numbers yield better results.

    Returns: All calculated metrics as a dict for each element in the batch.
    """
    if metrics is None:
        metrics = ["DSC", "ECE", "CM"]
    assert (
        len(predictions.shape) == 4 or len(predictions.shape) == 3
    ), "The predictions must either have the shape (batch, channel, height, width) or (batch, height, width)"
    assert len(labels.shape) == 3, "The labels must have the shape (batch, height, width)"
    assert len(mask.shape) == 3, "The mask must have the shape (batch, height, width)"
    assert labels.shape == mask.shape, "The labels and mask tensors must match in their dimensions"
    assert labels.dtype == torch.int64, "labels must be index values"
    assert mask.dtype == torch.bool, "The mask must be a boolean tensor"
    if n_classes is None:
        n_classes = len(settings_seg.labels)

    if len(predictions.shape) == 3:
        assert predictions.shape == labels.shape, "All tensors must match in the (batch, height, width) dimensions"
        assert predictions.dtype == torch.int64, "predictions are not in softmax format and must be index values"
        predictions_labels = predictions
        predictions_softmaxes = None
    else:
        assert (
            predictions.shape[:1] + predictions.shape[2:] == labels.shape
        ), "All tensors must match in the (batch, height, width) dimensions"
        assert (
            predictions.shape[1] == n_classes
        ), "The second dimensions of the predictions must match the number of classes"

        predictions = check_invalid_input(predictions, "predictions")
        predictions_softmaxes_sum = torch.sum(predictions, dim=1)

        assert torch.allclose(predictions_softmaxes_sum, torch.ones_like(predictions_softmaxes_sum)), (
            "All of the softmax should sum upto approx. 1 in the class dimension. Are you sure that you are not sending"
            " logits rather than softmax?"
        )

        predictions_labels = predictions.argmax(dim=1)
        predictions_softmaxes = predictions

    n_batch = labels.shape[0]

    # All metrics should only consider valid pixels
    valid_labels = [labels[b, mask[b]] for b in range(n_batch)]
    valid_predictions_labels = [predictions_labels[b, mask[b]] for b in range(n_batch)]

    result_batch = {}

    used_labels = [
        valid_labels[b].unique() for b in range(n_batch)
    ]  # This is important later when calculating the dice per image (to know which values to exclude)

    # dice similarity coefficient
    if "DSC" in metrics:
        dice = calc_dice_metric(predictions_labels, labels, mask)
        assert len(used_labels) == len(dice)
        assert all([torch.all(used_labels[b] == dice[b]["used_labels"]) for b in range(n_batch)])

        result_batch |= {
            "dice_metric": [b["dice_metric"] for b in dice],
            "dice_metric_image": [b["dice_metric_image"] for b in dice],
        }

    # confusion matrix
    if "CM" in metrics:
        conf_mat = [
            confusion_matrix(valid_predictions_labels[b], valid_labels[b], task="multiclass", num_classes=n_classes)
            for b in range(n_batch)
        ]
        assert all([c.shape == (n_classes, n_classes) for c in conf_mat]), "The confusion matrix has the wrong shape"

        result_batch |= {
            "confusion_matrix": conf_mat,
        }

    # expected calibration error
    if predictions_softmaxes is not None and "ECE" in metrics:
        # The losses can only be calculated if we have the softmaxes
        valid_predictions_softmaxes = [predictions_softmaxes[b, :, mask[b]] for b in range(n_batch)]
        assert all(
            [len(t.shape) == 2 for t in valid_predictions_softmaxes]
        ), "Invalid shape of the valid predicted softmaxes"

        ece = []
        ece_model = ECELoss()
        for b in range(n_batch):
            ece_result = ece_model(valid_predictions_softmaxes[b], valid_labels[b])
            ece.append(ece_result)

        result_batch["ece"] = ece

    # normalized surface dice
    if "NSD" in metrics:
        assert (
            tolerances is not None
        ), "Tolerance thresholds in pixels for each class, should be specified for calculating Surface Dice (NSD)"
        nsd = calc_surface_dice(predictions_labels, labels, mask, tolerances)

        result_batch |= {
            "surface_dice_metric": [b["surface_dice_metric"] for b in nsd],
            "surface_dice_metric_image": [b["surface_dice_metric_image"] for b in nsd],
        }

    # average surface distance
    if "ASD" in metrics:
        asd = calc_surface_distance(predictions_labels, labels, mask)

        result_batch |= {
            "surface_distance_metric": [b["surface_distance_metric"] for b in asd],
            "surface_distance_metric_image": [b["surface_distance_metric_image"] for b in asd],
        }

    result_batch |= {
        "used_labels": used_labels,
    }

    results_batch = [
        dict(zip(result_batch, t)) for t in zip(*result_batch.values())
    ]  # Convert from dict of lists to list of dicts (https://stackoverflow.com/a/33046935/2762258)

    return results_batch
