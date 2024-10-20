# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from abc import abstractmethod
from pathlib import Path

import numpy as np

from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.LabelMapping import LabelMapping


class SpecsGeneration:
    def __init__(self, name: str):
        self.name = name

    def generate_dataset(self, target_folder: Path = None):
        """
        Generates the folds for the dataset.

        Args:
            target_folder: Path to the folder where the data specification file should be stored (defaults to the models/data folder).
        """
        if target_folder is None:
            target_folder = Path(__file__).parent

        # Folds
        folds = self.generate_folds()
        specs_path = target_folder / f"{self.name}.json"
        with specs_path.open("w") as f:
            json.dump(folds, f, indent=4)
            f.write("\n")  # Add newline at the end of the file so that all JSON files are formatted the same way

        # Make sure we can read the specs file
        DataSpecification(specs_path)

    @abstractmethod
    def generate_folds(self) -> list[dict]:
        """
        Generate the fold datasets according to the data specification.

        Returns: List with different folds.
        """

    @staticmethod
    def compute_label_distribution(paths: list[DataPath], mapping: LabelMapping) -> tuple[list[str], np.ndarray]:
        """
        Computes the label distribution per subject for a list of paths. This is useful to create stratified folds for multi-labelled images (i.e., segmentation images).

        Args:
            paths: A list of data paths.
            mapping: The label mapping object.

        Returns: List of subjects and the label distribution as count array (n_subjects, n_classes).
        """
        subjects = sorted({p.subject_name for p in paths})
        label_distribution = np.zeros((len(subjects), len(mapping)), dtype=np.int64)

        for p in paths:
            for label_name in p.annotated_labels():
                label_index = mapping.name_to_index(label_name)
                if mapping.is_index_valid(label_index):
                    label_distribution[subjects.index(p.subject_name), label_index] = 1

        return subjects, label_distribution
