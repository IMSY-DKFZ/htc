# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
import torch

from htc.evaluation.evaluate_images import evaluate_images
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import get_nsd_thresholds


def rater_evaluation(annotation_name: str) -> tuple[pd.DataFrame, dict]:
    paths_rater = list(DataPath.iterate(settings.data_dirs.semantic, annotation_name=annotation_name))
    paths_true = [DataPath.from_image_name(p.image_name()) for p in paths_rater]
    assert len(paths_rater) == len(paths_true)
    assert len(paths_rater) == 20

    label_mapping = settings_seg.label_mapping
    config = Config({"label_mapping": label_mapping})
    dataset_rater = DatasetImage(paths_rater, train=False, config=config)
    dataset_true = DatasetImage(paths_true, train=False, config=config)
    tolerances = get_nsd_thresholds(label_mapping)

    results = {}
    additional_labels = []
    missing_labels = []
    mask_differences = []
    for sample_rater, sample_true in zip(dataset_rater, dataset_true, strict=True):
        assert sample_rater["image_name"] == sample_true["image_name"]

        predictions = sample_rater["labels"].unsqueeze(dim=0)
        labels = sample_true["labels"].unsqueeze(dim=0)
        mask = sample_true["valid_pixels"].unsqueeze(dim=0)

        # It is possible that in the new annotations an image now contains e.g. an "unsure" label. We are ignoring these pixels
        mask_diff = mask != sample_rater["valid_pixels"].unsqueeze(dim=0)
        if torch.any(mask_diff):
            diff = mask_diff.sum()
            settings.log.info(f"{sample_true['image_name']}: There are {diff} pixels different in the mask files")
            mask_differences.append(diff)

        mask[mask_diff] = False

        results[sample_true["image_name"]] = evaluate_images(
            predictions, labels, mask, tolerances=tolerances, metrics=["DSC", "NSD", "ASD", "ECE", "CM"]
        )[0]

        labels_rater = {
            label_mapping.index_to_name(l.item()) for l in predictions.unique() if label_mapping.is_index_valid(l)
        }
        labels_gt = {label_mapping.index_to_name(l.item()) for l in labels.unique() if label_mapping.is_index_valid(l)}
        if labels_rater != labels_gt:
            if len(labels_rater - labels_gt) > 0:
                settings.log.info(
                    f"{sample_true['image_name']}: Additional labels by the rater: {sorted(labels_rater - labels_gt)}"
                )
                additional_labels += list(labels_rater - labels_gt)
            if len(labels_gt - labels_rater) > 0:
                settings.log.info(
                    f"{sample_true['image_name']}: Labels missing by the rater: {sorted(labels_gt - labels_rater)}"
                )
                missing_labels += list(labels_gt - labels_rater)

    additional_labels = sorted(additional_labels)
    missing_labels = sorted(missing_labels)
    mask_differences = sorted(mask_differences)

    if len(additional_labels) > 0:
        settings.log.info(f"{len(additional_labels)} total additional labels: {additional_labels}")
    if len(missing_labels) > 0:
        settings.log.info(f"{len(missing_labels)} total missing labels: {missing_labels}")
    if len(mask_differences) > 0:
        settings.log.info(f"{len(mask_differences)}: total pixel difference in the masks: {sum(mask_differences)}")

    rows = []
    for image_name, res in results.items():
        subject_name, timestamp = image_name.split("#")
        rows.append({
            "subject_name": subject_name,
            "timestamp": timestamp,
            "dice_metric_image": res["dice_metric_image"],
            "surface_distance_metric_image": res["surface_distance_metric_image"],
            settings_seg.nsd_aggregation: res["surface_dice_metric_image"],
            "confusion_matrix": res["confusion_matrix"].numpy(),
        })

    stats = {
        "additional_labels": additional_labels,
        "missing_labels": missing_labels,
        "mask_differences": mask_differences,
    }

    return pd.DataFrame(rows), stats
