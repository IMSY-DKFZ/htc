# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import numpy as np

from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.settings import settings
from htc.tissue_atlas.tables import median_cam_table


class SpecsGenerationTissueAtlas(SpecsGeneration):
    def __init__(self, n_pigs: int, seed: int = 0):
        super().__init__(name=f"tissue-atlas_loocv_test-{n_pigs}_seed-{seed}_cam-118")
        self.df = median_cam_table()

        np.random.seed(seed)

        df_118 = self.df.query('camera_name == "0202-00118_correct-1"')
        pigs_118 = sorted(df_118["subject_name"].unique())
        n_labels = len(self.df["label_name"].unique())

        self.test_pigs = np.random.choice(pigs_118, n_pigs, replace=False)
        n_test_labels = len(df_118.query("subject_name in @self.test_pigs")["label_name"].unique())
        n_train_labels = len(df_118.query("subject_name not in @self.test_pigs")["label_name"].unique())
        assert (
            n_test_labels == n_labels and n_train_labels == n_labels
        ), "All labels must be represented in the train and the test set"

    def generate_folds(self) -> list[dict]:
        data_specs = []

        train_pigs = sorted(set(self.df["subject_name"].unique()) - set(self.test_pigs))
        imgs_test = self.df.query("subject_name in @self.test_pigs")["image_name"].unique().tolist()

        for subject_name in train_pigs:
            imgs_train = (
                self.df.query("subject_name != @subject_name and subject_name not in @self.test_pigs")["image_name"]
                .unique()
                .tolist()
            )
            imgs_val = self.df.query("subject_name == @subject_name")["image_name"].unique().tolist()

            fold_specs = {
                "fold_name": f"fold_{subject_name}",
                "train": {
                    "data_path_module": "htc.tivita.DataPath",
                    "data_path_class": "DataPath",
                    "image_names": sorted(imgs_train),
                },
                "val": {
                    "data_path_module": "htc.tivita.DataPath",
                    "data_path_class": "DataPath",
                    "image_names": sorted(imgs_val),
                },
                "test": {
                    "data_path_module": "htc.tivita.DataPath",
                    "data_path_class": "DataPath",
                    "image_names": sorted(imgs_test),
                },
            }

            data_specs.append(fold_specs)

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates the data specification file for the tissue atlas paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.htc_package_dir / "tissue_atlas/data",
        help="Directory where the resulting data specification file should be stored.",
    )
    parser.add_argument("--pigs", type=int, default=8, help="Number of pigs to include in the test set.")
    args = parser.parse_args()

    SpecsGenerationTissueAtlas(n_pigs=args.pigs).generate_dataset(args.output_dir)
