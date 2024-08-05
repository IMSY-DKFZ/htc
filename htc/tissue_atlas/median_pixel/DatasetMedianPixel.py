# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import median_table
from htc.utils.Task import Task


class DatasetMedianPixel(HTCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load precomputed spectra
        df = median_table(image_names=self.image_names, config=self.config)

        self.labels = torch.from_numpy(df["label_index_mapped"].values) if self.config["label_mapping"] else None
        self.image_labels = (
            torch.from_numpy(np.stack(df["image_labels"])) if self.config["input/image_labels"] else None
        )

        if self.labels is not None:
            assert not df.duplicated(["image_name", "label_index_mapped"]).any(), (
                "Found duplicated rows (same (image_name, label_index_mapped) combination found more than once). Cannot"
                " use this table because it is unclear which median spectra should be used in this case"
            )

        # We need to set these variables again because an image may contain more than one median spectra and we want to use all (and find the corresponding path to each annotation)
        self.image_names = df["image_name"].tolist()
        self.paths = [DataPath.from_image_name(image_name) for image_name in self.image_names]

        if self.config["input/normalization"] == "L1" or "L1" in self.config["input/preprocessing"]:
            self.features = df["median_normalized_spectrum"].values
        else:
            self.features = df["median_spectrum"].values

        self.features = torch.from_numpy(np.stack(self.features))
        self.features = self.apply_transforms(self.features)

        if self.config["input/meta"]:
            self.meta = torch.stack([self.read_meta(path) for path in self.paths])
            assert len(self.meta) == len(self.features), "Meta and features must have the same length"
        else:
            self.meta = None

        assert len(self.features) == len(self.paths) == len(self.image_names), "All arrays must have the same length"
        if self.labels is not None:
            assert len(self.labels) == len(self.features), "Labels and features must have the same length"
        if self.image_labels is not None:
            assert len(self.image_labels) == len(self.features), "Image labels and features must have the same length"

    def label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates for each unique label in the dataset the number of occurrences based on the task, i.e. either based on the labels or image_labels attribute.

        Compared to the parent class, this method counts the number of annotations and not pixels.

        Returns: Tuple with label values and corresponding counts.
        """
        task = Task.from_config(self.config)
        return getattr(self, task.labels_name()).unique(return_counts=True)

    def __len__(self) -> int:
        task = Task.from_config(self.config)
        return len(getattr(self, task.labels_name()))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {"features": self.features[index, :]}

        if self.labels is not None:
            sample["labels"] = self.labels[index]
        if self.image_labels is not None:
            sample["image_labels"] = self.image_labels[index]

        if self.meta is not None:
            sample["meta"] = self.meta[index, :]

        if not self.train:
            sample["image_name"] = self.image_names[index]
            sample["image_index"] = index

        return sample
