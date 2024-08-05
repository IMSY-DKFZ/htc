# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from enum import Enum, unique
from typing_extensions import Self


@unique
class Task(Enum):
    """This enum can be used to distinguish between a segmentation task (with pixel-level labels) or a classification task (with image-level labels)."""

    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"

    def labels_name(self) -> str:
        """Returns the name of the labels attribute (e.g., used in DatasetMedianPixel) or the name of the key in the batch which stores the labels."""
        if self == Task.SEGMENTATION:
            return "labels"
        elif self == Task.CLASSIFICATION:
            return "image_labels"
        else:
            raise ValueError(f"Unknown task: {self}")

    @classmethod
    def from_config(cls, config) -> Self:
        return cls(config.get("task", "segmentation"))
