# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping


class DatasetMedianPixel(HTCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load precomputed spectra
        label_mapping = LabelMapping.from_config(self.config)

        df = median_table(image_names=self.image_names, label_mapping=label_mapping)
        assert not df.duplicated(["image_name", "label_index_mapped"]).any(), (
            "Found duplicated rows (same (image_name, label_index_mapped) combination found more than once). Cannot use"
            " this table because it is unclear which median spectra should be used in this case"
        )

        # We need to set these variables again because an image may contain more than one median spectra and we want to use all (and find the corresponding path to each annotation)
        self.image_names = df["image_name"].tolist()
        self.paths = [DataPath.from_image_name(image_name) for image_name in self.image_names]

        if self.config["input/normalization"] == "L1":
            self.features = df["median_normalized_spectrum"].values
        else:
            self.features = df["median_spectrum"].values

        self.labels = torch.from_numpy(df["label_index_mapped"].values)
        self.features = torch.from_numpy(np.stack(self.features))
        self.features = self.apply_transforms(self.features)

        assert (
            len(self.features) == len(self.labels) == len(self.paths) == len(self.image_names)
        ), "All arrays must have the same length"

    def label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.labels.unique(return_counts=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "features": self.features[index, :],
            "labels": self.labels[index],
        }

        if not self.train:
            sample["image_name"] = self.image_names[index]
            sample["image_index"] = index

        return sample
