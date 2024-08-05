# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from abc import abstractmethod
from pathlib import Path

from htc.models.data.DataSpecification import DataSpecification


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
        pass
