# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from htc.model_processing.Runner import Runner
from htc.model_processing.SinglePredictor import SinglePredictor
from htc.models.median_pixel.DatasetMedianPixel import DatasetMedianPixel

if __name__ == "__main__":
    # htc median_test_table --model median_pixel --run-folder 2024-02-23_14-08-16_median_18classes --spec tissue-atlas_loocv_test-8_seed-0_cam-118.json --table-name test_table_pigs
    runner = Runner(
        description="Create a test table based on a trained median spectra model for a new set of paths.",
        default_output_to_run_dir=True,
    )
    runner.add_argument("--table-name", default="test_table_new", type=str, help="Name of the generated table")

    # Inference for the median spectra is super fast, so we just use the single predictor here
    predictor = SinglePredictor(runner.args.model, runner.run_dir.name, test=True, config=runner.config)
    predictor.config["dataloader_kwargs/num_workers"] = 1
    dataset = DatasetMedianPixel(runner.paths, config=predictor.config, train=False)
    dataloader = DataLoader(dataset, **predictor.config["dataloader_kwargs"])

    logits = []
    labels = []
    for batch in dataloader:
        logits.append(predictor.predict_batch(batch)["class"])
        labels.append(batch["labels"])
    logits = torch.cat(logits, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    assert len(dataset.paths) == logits.shape[0], "Number of paths and logits do not match"
    assert len(dataset.paths) == labels.shape[0], "Number of paths and labels do not match"

    # Create a test table similar as the run_test_table_generation.py script
    df = {
        "label": labels,
        "ensemble_logits": [
            np.take(logits, i, axis=0) for i in range(logits.shape[0])
        ],  # Will save it as vectors in the table
        "image_name": [p.image_name() for p in dataset.paths],
    }

    # Already expand the image name
    meta = pd.DataFrame([p.image_name_typed() for p in dataset.paths])
    meta = meta.to_dict(orient="list")
    df |= meta

    df = pd.DataFrame(df)

    df.to_pickle(runner.output_dir / f"{runner.args.table_name}.pkl.xz")
