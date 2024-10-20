# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np
import pandas as pd
import torch

from htc.evaluation.metrics.NSDToleranceEstimation import NSDToleranceEstimation
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


def nsd_thresholds() -> pd.DataFrame:
    paths_rater = list(DataPath.iterate(settings.data_dirs.semantic, annotation_name="semantic#inter1"))
    paths_true = [DataPath.from_image_name(p.image_name()) for p in paths_rater]
    assert len(paths_rater) == len(paths_true)
    assert len(paths_rater) == 20

    pigs = sorted({p.subject_name for p in paths_rater})
    assert len(pigs) == 20

    config = Config({"label_mapping": None})  # We calculate the thresholds based on the original labels
    dataset_rater = DatasetImage(paths_rater, train=False, config=config)
    dataset_true = DatasetImage(paths_true, train=False, config=config)

    last_valid_label_index = paths_rater[0].dataset_settings["last_valid_label_index"]
    n_classes = len([
        l for l in paths_rater[0].dataset_settings["label_mapping"].values() if l <= last_valid_label_index
    ])
    estimator = NSDToleranceEstimation(n_classes, n_groups=len(pigs))

    # Collect distances for each image
    for sample_rater, sample_true in zip(dataset_rater, dataset_true, strict=True):
        assert sample_rater["image_name"] == sample_true["image_name"]

        subject_name = sample_true["image_name"].split("#")[0]
        subject_namex = pigs.index(subject_name)

        predictions = sample_rater["labels"]
        labels = sample_true["labels"]
        mask = torch.logical_and(
            labels <= last_valid_label_index, predictions <= last_valid_label_index
        )  # Ignore unlabeled and overlap pixels

        estimator.add_image(predictions, labels, mask, group_index=subject_namex)

    # We test different tolerance values by using different aggregation functions
    functions = {
        "mean": np.mean,
        "median": np.median,
        "q75": partial(np.quantile, q=0.75),
        "q95": partial(np.quantile, q=0.95),
    }
    tolerances = {}
    for name, f in functions.items():
        tolerances[name] = estimator.class_tolerances(reduction_func=f)

    # Table with the thresholds for each aggregation
    rows = []
    for label, label_index in paths_rater[0].dataset_settings["label_mapping"].items():
        if label in settings_seg.labels:
            current_row = {"label_name": label, "label_index": settings_seg.label_mapping.name_to_index(label)}

            for name, (avg_vec, std_vec) in tolerances.items():
                current_row[f"tolerance_{name}"] = avg_vec[label_index]
                current_row[f"tolerance_{name}_std"] = std_vec[label_index]

            rows.append(current_row)

    return pd.DataFrame(rows).sort_values(by=["label_index"])


if __name__ == "__main__":
    df = nsd_thresholds()
    settings_seg.nsd_tolerances_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings_seg.nsd_tolerances_path, index=False)
