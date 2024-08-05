# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import itertools
from pathlib import Path

import numpy as np

from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.LabelMapping import LabelMapping
from htc.utils.paths import filter_semantic_labels_only


def extract_paths_known(paths_train: list[DataPath], fold_index: int) -> list[DataPath]:
    np.random.seed(fold_index)

    pigs = set()
    for data_path in paths_train:
        pigs.add(data_path.subject_name)

    paths_val_known = []
    for pig in sorted(pigs):
        # Select a random image for each pig from the training dataset to validate generalizability
        paths_pig = [p for p in paths_train if p.subject_name == pig]
        random_index = np.random.randint(0, len(paths_pig))

        # Add the pig to the new validation dataset
        paths_val_known.append(paths_pig[random_index])

        # Finally, remove pig from the training set
        paths_train.remove(paths_pig[random_index])

    return paths_val_known


folds_pigs = [
    ["P041", "P060", "P069"],
    ["P044", "P050", "P059"],
    ["P045", "P061", "P071"],
    ["P047", "P049", "P070"],
    ["P048", "P057", "P058"],
]
train_set = sorted(itertools.chain(*folds_pigs))
test_set = ["P043", "P046", "P062", "P068", "P072"]


def filter_train(path: "DataPath") -> bool:
    """
    This filter can be used in conjunction with the DataPath.iterate method to only receive paths from the training set.

    Args:
        path: Data path to be tested.

    Returns: True when the path is from the training set.
    """
    return path.subject_name not in test_set


def filter_test(path: "DataPath") -> bool:
    """Similar to filter_train but yielding only paths from the test set."""
    return path.subject_name in test_set


class SpecsGenerationPig(SpecsGeneration):
    def __init__(self, name: str, label_mapping: LabelMapping = None):
        super().__init__(name=name)
        self.label_mapping = settings_seg.label_mapping if label_mapping is None else label_mapping


class SpecsGenerationPigLOOCV(SpecsGenerationPig):
    def __init__(self):
        super().__init__(name="pigs_semantic-only_loocv")

    def generate_folds(self) -> list[dict]:
        data_specs = []
        imgs_test = [p.image_name() for p in DataPath.iterate(settings.data_dirs.semantic, filters=[filter_test])]

        for subject_name in train_set:
            paths = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))
            validation_pigs = subject_name.split(",")

            paths_train = [p for p in paths if p.subject_name not in validation_pigs]
            paths_val_unknown = [p for p in paths if p.subject_name in validation_pigs]
            paths_val_known = extract_paths_known(paths_train)

            fold_specs = {
                "fold_name": f"fold_{subject_name}",
                "train_semantic": {
                    "image_names": [p.image_name() for p in paths_train],
                },
                "val_semantic_unknown": {
                    "image_names": [p.image_name() for p in paths_val_unknown],
                },
                "val_semantic_known": {
                    "image_names": [p.image_name() for p in paths_val_known],
                },
                "test": {
                    "image_names": imgs_test,
                },
            }

            data_specs.append(fold_specs)

        return data_specs


class SpecsGenerationPigKFolds(SpecsGenerationPig):
    def __init__(self):
        super().__init__(name=f"pigs_semantic-only_{len(folds_pigs)}foldsV2")

    def generate_folds(self) -> list[dict]:
        data_specs = []
        imgs_test = [p.image_name() for p in DataPath.iterate(settings.data_dirs.semantic, filters=[filter_test])]

        for i, fold in enumerate(folds_pigs):
            paths = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))

            paths_train = [p for p in paths if p.subject_name not in fold]
            paths_val_unknown = [p for p in paths if p.subject_name in fold]
            paths_val_known = extract_paths_known(paths_train, fold_index=i)

            fold_specs = {
                "fold_name": "fold_" + ",".join(fold),
                "train_semantic": {
                    "image_names": [p.image_name() for p in paths_train],
                },
                "val_semantic_unknown": {
                    "image_names": [p.image_name() for p in paths_val_unknown],
                },
                "val_semantic_known": {
                    "image_names": [p.image_name() for p in paths_val_known],
                },
                "test": {
                    "image_names": imgs_test,
                },
            }

            data_specs.append(fold_specs)

        return data_specs


class SpecsGenerationPigKFoldsWithMasks(SpecsGenerationPig):
    def __init__(self):
        super().__init__(name=f"pigs_semantic-masks_{len(folds_pigs)}foldsV1")

    def generate_folds(self) -> list[dict]:
        data_specs = []
        imgs_test = [p.image_name() for p in DataPath.iterate(settings.data_dirs.semantic, filters=[filter_test])]

        for fold in folds_pigs:
            paths = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))
            paths_masks = list(
                DataPath.iterate(settings.data_dirs.masks, filters=[filter_train, filter_semantic_labels_only])
            )

            paths_train = [p for p in paths if p.subject_name not in fold]
            paths_val_unknown = [p for p in paths if p.subject_name in fold]
            paths_val_known = extract_paths_known(paths_train)
            paths_train_masks = [p for p in paths_masks if p.subject_name not in fold]

            fold_specs = {
                "fold_name": "fold_" + ",".join(fold),
                "train_semantic": {
                    "image_names": [p.image_name() for p in paths_train],
                },
                "train_masks": {
                    "image_names": [p.image_name() for p in paths_train_masks],
                },
                "val_semantic_unknown": {
                    "image_names": [p.image_name() for p in paths_val_unknown],
                },
                "val_semantic_known": {
                    "image_names": [p.image_name() for p in paths_val_known],
                },
                "test": {
                    "image_names": imgs_test,
                },
            }

            data_specs.append(fold_specs)

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates the data specification file for semantic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.htc_package_dir / "models/data",
        help="Directory where the resulting data specification file should be stored.",
    )
    args = parser.parse_args()

    SpecsGenerationPigKFolds().generate_dataset(args.output_dir)
