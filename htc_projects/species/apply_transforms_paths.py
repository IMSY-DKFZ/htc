# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
from lightning import seed_everything

from htc.models.common.torch_helpers import copy_sample
from htc.models.common.transforms import HTCTransformation
from htc.models.image.DatasetImageBatch import DatasetImageBatch
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


def apply_transforms_paths_median(paths: list[DataPath], config: Config, epoch_size: int = 1) -> pd.DataFrame:
    """
    Apply transformations on some images and compute the median spectra.

    Args:
        paths: The data paths to apply the transforms to.
        config: The configuration object which defines the loading the the transformation of the data (`input/transforms_gpu`).
        epoch_size: The number of times the transformation should be applied to the same image (to mimic a similar behavior as during training).

    Returns: A table with the computed median spectrum based on the transformed data.
    """
    seed_everything(0, workers=True)
    dataloader = DatasetImageBatch.batched_iteration(paths, config)
    mapping = LabelMapping.from_config(config)
    aug = HTCTransformation.parse_transforms(config["input/transforms_gpu"], config=config, device="cuda")

    rows = []
    for batch in dataloader:
        for e in range(epoch_size):
            # Apply the transformation multiple times to the same image as also done during training
            batch_copy = copy_sample(batch)
            batch_copy = HTCTransformation.apply_valid_transforms(batch_copy, aug)

            for b in range(batch_copy["features"].size(0)):
                for label_index in batch_copy["labels"][b, batch_copy["valid_pixels"][b]].unique():
                    selection = batch_copy["labels"][b] == label_index
                    spectra = batch_copy["features"][b][selection]

                    path = DataPath.from_image_name(batch_copy["image_name"][b])
                    current_row = {"image_name": path.image_name()}
                    current_row |= path.image_name_typed()

                    current_row |= {
                        "epoch_index": e,
                        "label_name": mapping.index_to_name(label_index),
                        "median_normalized_spectrum": spectra.quantile(q=0.5, dim=0).cpu().numpy(),
                    }

                    rows.append(current_row)

    return pd.DataFrame(rows)
