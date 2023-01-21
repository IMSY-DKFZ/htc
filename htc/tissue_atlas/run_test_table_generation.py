# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import get_valid_run_dirs
from htc.utils.parallel import p_map


def save_test_table(run_dir: Path) -> None:
    table_path = run_dir / "test_table.pkl.xz"
    if table_path.exists():
        # Skip run if results are already aggregated
        return None

    labels = None
    image_names = None
    all_logits = None

    folds = sorted(run_dir.glob("fold*"))
    for i, fold_dir in enumerate(folds):
        test_file = fold_dir / "test_results.npz"
        data = np.load(test_file, allow_pickle=True)

        # Basic checks
        if labels is None:
            labels = data["labels"]
        else:
            assert all(labels == data["labels"])
        if image_names is None:
            image_names = data["image_names"]
        else:
            assert all(image_names == data["image_names"])

        # Collect predictions for ensembling
        logits = data["logits"]
        if all_logits is None:
            all_logits = np.empty((len(folds), *logits.shape), dtype=np.float32)

        all_logits[i] = logits

    # all_logits.shape = [n_folds, n_samples, n_classes]
    ensemble_mode = stats.mode(np.argmax(all_logits, axis=2), axis=0).mode[0]
    ensemble_logits = np.mean(all_logits, axis=0)
    ensemble_softmax = np.mean(softmax(all_logits, axis=2), axis=0)

    df = {
        "label": labels,
        "ensemble_mode": ensemble_mode,
        "ensemble_logits": [
            np.take(ensemble_logits, i, axis=0) for i in range(ensemble_logits.shape[0])
        ],  # Will save it as vectors in the table
        "ensemble_softmax": [np.take(ensemble_softmax, i, axis=0) for i in range(ensemble_softmax.shape[0])],
        "image_name": image_names,
    }

    # Already expand the image name
    meta = pd.DataFrame([DataPath.from_image_name(image_name).image_name_typed() for image_name in image_names])
    meta = meta.to_dict(orient="list")
    df |= meta

    df = pd.DataFrame(df)
    df.to_pickle(table_path)


if __name__ == "__main__":
    run_dirs = [r for r in get_valid_run_dirs() if r.parent.name == "median_pixel"]

    p_map(save_test_table, run_dirs)
