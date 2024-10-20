# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.utils.helper_functions import median_table


class SpecsGenerationKidneyProjection(SpecsGeneration):
    def __init__(self, train_subjects: list[str]):
        self.train_subjects = train_subjects
        super().__init__(name=f"kidney_projection_train={','.join(self.train_subjects)}")

        self.df = median_table(dataset_name="2023_04_22_Tivita_multiorgan_kidney", annotation_name="semantic#primary")
        self.df = self.df.query("label_name == 'kidney'")

    def generate_folds(self) -> list[dict]:
        data_specs = []

        imgs_train = self.df.query("subject_name in @self.train_subjects")["image_name"].unique().tolist()
        imgs_test = self.df.query("subject_name not in @self.train_subjects")["image_name"].unique().tolist()

        fold_specs = {
            "fold_name": "fold_all",
            "train": {
                "image_names": sorted(imgs_train),
            },
            "test": {
                "image_names": sorted(imgs_test),
            },
        }

        data_specs.append(fold_specs)

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates the data specification file for the kidney projection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory where the resulting data specification file should be stored.",
    )
    parser.add_argument(
        "--train-subjects",
        nargs="+",
        type=str,
        default=["P091", "P095", "P097", "P098"],
        help="List of subjects to include in the training. The remaining subjects compose the test set.",
    )
    args = parser.parse_args()

    SpecsGenerationKidneyProjection(train_subjects=args.train_subjects).generate_dataset(args.output_dir)
