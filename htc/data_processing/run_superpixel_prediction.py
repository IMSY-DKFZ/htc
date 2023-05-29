# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Any, Union

import pandas as pd
from threadpoolctl import threadpool_limits

from htc.evaluation.evaluate_images import evaluate_images
from htc.evaluation.evaluate_superpixels import EvaluateSuperpixelImage
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import get_nsd_thresholds
from htc.utils.parallel import p_map


def aggregate_results(i: int) -> dict[str, Union[dict, Any]]:
    sample = dataset_all[i]
    result = EvaluateSuperpixelImage().evaluate_cpp(sample)
    result_eval = result["evaluation"]
    subject_name, timestamp = dataset_all.image_names[i].split("#")

    predictions = result["predictions"].unsqueeze(dim=0)
    labels = sample["labels"].unsqueeze(dim=0)
    mask = sample["valid_pixels"].unsqueeze(dim=0)
    metrics = evaluate_images(predictions, labels, mask, tolerances=tolerances, metrics=["NSD", "ASD"])[0]

    return {
        "dice": result_eval["dice_metric_image"],
        "asd": metrics["surface_distance_metric_image"],
        "nsd": metrics["surface_dice_metric_image"],
        "subject_name": subject_name,
        "timestamp": timestamp,
    }


if __name__ == "__main__":
    config = Config.from_model_name("default", "superpixel_classification")
    config["input/no_features"] = True

    paths = list(DataPath.iterate(settings.data_dirs.semantic))
    dataset_all = DatasetImage(paths, train=False, config=config)
    tolerances = get_nsd_thresholds(settings_seg.label_mapping)

    with threadpool_limits(1):
        rows = p_map(aggregate_results, range(len(dataset_all)))

    df = pd.DataFrame(rows)
    target_folder = settings.results_dir / "superpixel_gt"
    target_folder.mkdir(parents=True, exist_ok=True)
    df.to_pickle(target_folder / "spxs_predictions.pkl.xz")
