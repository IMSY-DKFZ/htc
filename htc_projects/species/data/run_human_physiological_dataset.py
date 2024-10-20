# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np

from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import median_table
from htc.utils.import_extra import requires_extra
from htc.utils.paths import filter_labels
from htc_projects.species.settings_species import settings_species
from htc_projects.species.tables import baseline_table, ischemic_table

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    _missing_library = ""
except ImportError:
    _missing_library = "iterative-stratification"


class SpecsGenerationHumanPhysiological(SpecsGeneration):
    mapping = settings_species.label_mapping

    @requires_extra(_missing_library)
    def __init__(
        self,
        test_ratio: float = 0.3,
        test_subjects: list[str] = None,
        seed: int = 0,
        additional_species: list[str] = None,
    ):
        additional_species = additional_species if additional_species is not None else []
        if len(additional_species) > 0:
            additional_species_str = "+" + "+".join(additional_species)
        else:
            additional_species_str = ""

        self.n_folds = 5
        super().__init__(
            name=f"human_semantic-only{additional_species_str}_physiological-{','.join(settings_species.malperfused_labels)}_{self.n_folds}folds_mapping-{len(self.mapping)}_seed-{seed}"
        )
        np.random.seed(seed)

        self.paths = self.get_paths()

        if test_subjects is None:
            # Test set split (also stratified)
            subjects, label_distribution = self.compute_label_distribution(self.paths, self.mapping)
            mskf = MultilabelStratifiedKFold(n_splits=math.ceil(1 / test_ratio), shuffle=True)
            _, test_indices = next(iter(mskf.split(subjects, label_distribution)))

            self.test_subjects = [subjects[i] for i in test_indices]
        else:
            self.test_subjects = test_subjects
            subjects = sorted({p.subject_name for p in self.paths})

        self.main_subjects = sorted(set(subjects) - set(self.test_subjects))

        assert len(self.main_subjects) + len(self.test_subjects) == len(subjects)
        assert set(self.main_subjects).isdisjoint(self.test_subjects)

        self.imgs_species = {}
        for species in additional_species:
            if species.endswith("-p"):
                # Only physiological images from this species requested
                df_species = baseline_table(self.mapping)
                species_name = species[:-2]
            else:
                df_species = ischemic_table(self.mapping)
                df_species = df_species[
                    df_species.baseline_dataset | df_species.label_name.isin(settings_species.malperfused_labels)
                ]
                species_name = species

            df_species = df_species[df_species.species_name == species_name]

            imgs = [p.image_name_annotations() for p in DataPath.from_table(df_species)]
            assert len(imgs) > 0, f"No images found for species {species_name}"
            self.imgs_species[f"train_{species}"] = {"image_names": sorted(imgs)}

    def generate_folds(self) -> list[dict]:
        data_specs = []

        imgs_test = [p.image_name_annotations() for p in self.paths if p.subject_name in self.test_subjects]
        paths_train = [p for p in self.paths if p.subject_name in self.main_subjects]

        subjects, label_distribution = self.compute_label_distribution(paths_train, self.mapping)
        assert subjects == self.main_subjects

        mskf = MultilabelStratifiedKFold(n_splits=self.n_folds, shuffle=True)
        for i, (train_indices, val_indices) in enumerate(mskf.split(self.main_subjects, label_distribution)):
            train_subjects = [self.main_subjects[i] for i in train_indices]
            val_subjects = [self.main_subjects[i] for i in val_indices]
            assert len(train_subjects) + len(val_subjects) == len(self.main_subjects)
            assert set(train_subjects).isdisjoint(val_subjects)
            assert set(train_subjects).isdisjoint(self.test_subjects)
            assert set(val_subjects).isdisjoint(self.test_subjects)

            fold_specs = {
                "fold_name": f"fold_{i}",
                "train": {
                    "image_names": sorted([
                        p.image_name_annotations() for p in paths_train if p.subject_name in train_subjects
                    ]),
                },
                "val": {
                    "image_names": sorted([
                        p.image_name_annotations() for p in paths_train if p.subject_name in val_subjects
                    ]),
                },
                "test": {
                    "image_names": sorted(imgs_test),
                },
            } | self.imgs_species

            data_specs.append(fold_specs)

        return data_specs

    @staticmethod
    def get_paths() -> list[DataPath]:
        # Exclude every subject with a malperfused annotation
        df_mal = median_table(
            dataset_name="2021_07_26_Tivita_multiorgan_human",
            annotation_name="polygon#malperfused",
            label_mapping=SpecsGenerationHumanPhysiological.mapping,
            keep_mapped_columns=False,
        )
        df_mal = df_mal[df_mal.label_name.isin(settings_species.malperfused_labels)]
        malperfused_subjects = set(df_mal.subject_name)
        # and every manually excluded kidney subject
        malperfused_subjects.update(settings_species.malperfused_kidney_subjects)

        paths = list(
            DataPath.iterate(
                settings.data_dirs.human,
                annotation_name="semantic#primary",
                filters=[
                    (lambda p: p.subject_name not in malperfused_subjects),
                    partial(filter_labels, mapping=SpecsGenerationHumanPhysiological.mapping),
                ],
            )
        )
        assert len(paths) > 0, "No paths found for the human dataset"

        return paths


def generate_nested(n_nested_folds: int, output_dir: Path, seed: int = 0, **kwargs) -> None:
    paths = SpecsGenerationHumanPhysiological.get_paths()

    np.random.seed(seed)
    subjects, label_distribution = SpecsGenerationHumanPhysiological.compute_label_distribution(
        paths, SpecsGenerationHumanPhysiological.mapping
    )
    mskf = MultilabelStratifiedKFold(n_splits=n_nested_folds, shuffle=True)
    for i, (_, test_indices) in enumerate(mskf.split(subjects, label_distribution)):
        settings.log.info(f"Generating nested fold {i}")
        test_subjects = [subjects[i] for i in test_indices]

        spec_generator = SpecsGenerationHumanPhysiological(test_subjects=test_subjects, seed=seed, **kwargs)
        spec_generator.name = spec_generator.name.replace("folds_", f"folds_nested-{i}-{n_nested_folds - 1}_")
        spec_generator.generate_dataset(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a data specification file for the human dataset containing only physiological images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory where the resulting data specification file should be stored.",
    )
    parser.add_argument(
        "--additional-species",
        type=str,
        nargs="+",
        default=None,
        help="List of additional species data which should be added to the training set.",
    )
    args = parser.parse_args()

    generate_nested(
        n_nested_folds=settings_species.n_nested_folds,
        output_dir=args.output_dir,
        additional_species=args.additional_species,
    )
