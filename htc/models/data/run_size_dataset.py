# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from htc.models.data.run_pig_dataset import SpecsGenerationPig, filter_test, filter_train
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import basic_statistics
from htc.utils.LabelMapping import LabelMapping
from htc.utils.sqldf import sqldf


def label_mapping_dataset_size() -> LabelMapping:
    df_stats = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2.json")

    df_organs = df_stats.query("label_name in @settings_seg.labels")
    df_organ_counts = sqldf("""
        SELECT label_name, COUNT(DISTINCT subject_name) AS n_pigs
        FROM df_organs
        WHERE set_type = 'train'
        GROUP BY label_name
        ORDER BY n_pigs
    """)

    # We select all the labels which occur in every pig in the train dataset
    n_pigs_total = df_stats.query('set_type == "train"')["subject_name"].nunique()
    labels = df_organ_counts.query("n_pigs == @n_pigs_total")["label_name"].tolist()

    # Use the same background mapping
    mapping_settings = settings_seg.label_mapping
    mapping = {
        label_name: label_index
        for label_name, label_index in mapping_settings.mapping_name_index.items()
        if label_index == mapping_settings.name_to_index("background")
    }

    # All organs come after the background
    for i, label in enumerate(labels):
        mapping[label] = i + 1

    last_valid_label_index = max(mapping.values())

    # The rest are invalid labels
    i = 0
    for label_name in mapping_settings.label_names(include_invalid=True):
        if label_name not in mapping:
            mapping[label_name] = settings.label_index_thresh + i
            i += 1

    return LabelMapping(mapping, last_valid_label_index=last_valid_label_index)


def filter_full_classes(path: DataPath, mapping: LabelMapping) -> bool:
    used_classes = [
        label_name
        for label_name, label_index in mapping.mapping_name_index.items()
        if label_index > 0 and label_index < settings.label_index_thresh
    ]  # These classes are available in every pig
    available_labels = path.annotated_labels()

    return any(l in used_classes for l in available_labels)


def filter_min_pixels(path: DataPath, mapping: LabelMapping) -> bool:
    sample = DatasetImage(
        [path], train=False, config=Config({"input/no_features": True, "label_mapping": mapping.mapping_name_index})
    )[0]

    # Similar to DatasetPatchStream (we need enough pixels for patch extraction)
    patch_size_half = 32  # patch_64 is currently the largest model
    relevant_pixels = sample["valid_pixels"]
    relevant_pixels[sample["labels"] == mapping.name_to_index("background")] = False
    relevant_pixels[:patch_size_half, :] = False
    relevant_pixels[:, :patch_size_half] = False
    relevant_pixels[-patch_size_half:, :] = False
    relevant_pixels[:, -patch_size_half:] = False

    return relevant_pixels.sum().item() > 10000


class SpecsGenerationPigTrainData(SpecsGenerationPig):
    def __init__(self, n_repetitions: int):
        self.n_repetitions = n_repetitions
        super().__init__(
            name=f"pigs_semantic-only_dataset-size_repetitions={n_repetitions}V2",
            label_mapping=label_mapping_dataset_size(),
        )

    def generate_folds(self) -> list[dict]:
        from htc.models.data.run_pig_dataset import train_set

        data_specs = []

        filters_train = [
            filter_train,
            partial(filter_full_classes, mapping=self.label_mapping),
            partial(filter_min_pixels, mapping=self.label_mapping),
        ]
        filters_test = [filter_test, partial(filter_full_classes, mapping=self.label_mapping)]
        paths_train = list(DataPath.iterate(settings.data_dirs.semantic, filters=filters_train))
        paths_test = list(DataPath.iterate(settings.data_dirs.semantic, filters=filters_test))

        max_tries = 100
        for n_pigs in range(1, len(train_set) + 1):
            seed_pigs = {}
            for seed in range(self.n_repetitions):
                np.random.seed(seed)

                n = 0
                while True:
                    n += 1
                    if n >= max_tries:
                        raise ValueError(f"Could not find a disjunct pig set for {n_pigs} n_pigs and seed {seed}")

                    pigs = sorted(np.random.choice(train_set, n_pigs, replace=False).tolist())
                    if pigs not in seed_pigs.values():
                        # We want to have different set of pigs across seeds
                        break

                paths_fold = [p for p in paths_train if p.subject_name in pigs]
                seed_pigs[seed] = pigs

                fold_specs = {
                    "fold_name": f"fold_pigs={n_pigs}_seed={seed}",
                    "train_semantic": {
                        "data_path_module": "htc.tivita.DataPath",
                        "data_path_class": "DataPath",
                        "image_names": [p.image_name() for p in paths_fold],
                    },
                    "val_semantic_test": {
                        "data_path_module": "htc.tivita.DataPath",
                        "data_path_class": "DataPath",
                        "image_names": [p.image_name() for p in paths_test],
                    },
                }

                data_specs.append(fold_specs)

                if n_pigs == len(train_set):
                    # We don't need repetitions if we use the full dataset
                    break

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Creates the data specification file for the dataset size experiment (varying the number of training pigs)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.htc_package_dir / "models/data",
        help="Directory where the resulting data specification file should be stored.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="The number of repetitions for each n_pigs (number of different seeds).",
    )
    args = parser.parse_args()

    SpecsGenerationPigTrainData(n_repetitions=args.repetitions).generate_dataset(args.output_dir)
