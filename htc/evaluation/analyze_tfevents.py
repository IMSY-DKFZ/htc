# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Union

import pandas as pd
from scipy import interpolate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from htc.settings import settings


def read_tfevent_losses(run_dir: Path) -> Union[pd.DataFrame, None]:
    """
    Read loss values from tfevent files (from all folds).

    Args:
        run_dir: Path to the experiment folder containing a subfolder for each fold.

    Returns: Dataframe with all stored scalars.
    """
    rows = []
    fold_dirs = sorted(run_dir.glob("fold*"))

    if len(fold_dirs) == 0:
        settings.log.warning(f"Could not find any fold in the folder {run_dir}")
        return None

    loss_tags = None
    for fold_dir in fold_dirs:
        if len(os.listdir(fold_dir)) == 0:
            settings.log.warning(f"The folder {run_dir} does not contain valid results")
            return None

        acc = EventAccumulator(str(fold_dir))
        acc.Reload()

        tags = acc.Tags()["scalars"]
        current_loss_tags = sorted(
            t for t in tags if t not in ["hp_metric", "step", "epoch"]
        )  # Keep every train metric which is calculated on a per epoch basis

        if loss_tags is None:
            loss_tags = current_loss_tags
        else:
            assert loss_tags == current_loss_tags, f"Each fold must have the same loss tag (run_dir={run_dir})"

        assert len(loss_tags) > 0, f"At least one loss key required for {run_dir}"

        # Map steps to epochs
        steps_to_epoch = {}
        for e in acc.Scalars("epoch"):
            # There might be more than step for the same epoch in which case we only take the first value (https://github.com/PyTorchLightning/pytorch-lightning/issues/12851)
            if e.step not in steps_to_epoch:
                steps_to_epoch[e.step] = int(e.value)

        # First create a mapping from steps to values for each tag (the steps may differ between the tags)
        steps = set()
        tag_values = {}
        for tag in loss_tags:
            values = {}
            for event in acc.Scalars(tag):
                values[event.step] = event.value
                steps.add(event.step)
            tag_values[tag] = values

        # Interpolate missing epoch values based on the existing step to epoch mappings
        steps = sorted(steps)
        steps_at_epoch = [step for step in steps if step in steps_to_epoch]
        epochs = [steps_to_epoch[step] for step in steps_at_epoch]
        epoch_interpolate = interpolate.interp1d(steps_at_epoch, epochs, kind="nearest", fill_value="extrapolate")

        # Then create a row for each step
        for step in steps:
            epoch_index = steps_to_epoch[step] if step in steps_to_epoch else epoch_interpolate(step)
            step_values = {
                "fold_name": fold_dir.stem,
                "epoch_index": int(epoch_index),
                "step": step,
            }

            for tag in loss_tags:
                if step in tag_values[tag]:
                    step_values[tag] = tag_values[tag][step]

            rows.append(step_values)

    return pd.DataFrame(rows)
