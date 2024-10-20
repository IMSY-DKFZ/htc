# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.import_extra import requires_extra
from htc.utils.paths import filter_labels
from htc_projects.species.data.run_rat_semantic_dataset import SpecsGenerationRatSemantic
from htc_projects.species.settings_species import settings_species
from htc_projects.species.tables import ischemic_table

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    _missing_library = ""
except ImportError:
    _missing_library = "iterative-stratification"


class SpecsGenerationPigSemanticNested(SpecsGenerationRatSemantic):
    @requires_extra(_missing_library)
    def __init__(self, test_subjects: list[str], seed: int = 0, additional_species: list[str] = None):
        additional_species = additional_species if additional_species is not None else []
        if len(additional_species) > 0:
            additional_species_str = "+" + "+".join(additional_species)
        else:
            additional_species_str = ""

        self.test_subjects = test_subjects
        self.n_folds = 5
        SpecsGeneration.__init__(
            self,
            f"pig_semantic-only{additional_species_str}_{self.n_folds}folds_mapping-{len(self.mapping)}_seed-{seed}",
        )
        np.random.seed(seed)

        self.paths = self.get_paths()
        subjects = sorted({p.subject_name for p in self.paths})
        self.main_subjects = sorted(set(subjects) - set(self.test_subjects))

        self.imgs_species = {}
        for species in additional_species:
            df_ischemic = ischemic_table(self.mapping)
            df_ischemic = df_ischemic[
                df_ischemic.baseline_dataset | df_ischemic.label_name.isin(settings_species.malperfused_labels)
            ]
            df_ischemic = df_ischemic[df_ischemic.species_name == species]

            imgs = [p.image_name_annotations() for p in DataPath.from_table(df_ischemic)]
            self.imgs_species[f"train_{species}"] = {"image_names": sorted(imgs)}

    @staticmethod
    def get_paths() -> list[DataPath]:
        paths = list(
            DataPath.iterate(
                settings.data_dirs.semantic,
                annotation_name="semantic#primary",
                filters=[partial(filter_labels, mapping=SpecsGenerationRatSemantic.mapping)],
            )
        )
        assert len(paths) > 0, "No paths found"

        return paths


def generate_nested(n_nested_folds: int, output_dir: Path, seed: int = 0, **kwargs) -> None:
    paths = SpecsGenerationPigSemanticNested.get_paths()

    np.random.seed(seed)
    subjects, label_distribution = SpecsGenerationPigSemanticNested.compute_label_distribution(
        paths, SpecsGenerationPigSemanticNested.mapping
    )
    mskf = MultilabelStratifiedKFold(n_splits=n_nested_folds, shuffle=True)
    for i, (_, test_indices) in enumerate(mskf.split(subjects, label_distribution)):
        settings.log.info(f"Generating nested fold {i}")
        test_subjects = [subjects[i] for i in test_indices]

        spec_generator = SpecsGenerationPigSemanticNested(test_subjects=test_subjects, seed=seed, **kwargs)
        spec_generator.name = spec_generator.name.replace("folds_", f"folds_nested-{i}-{n_nested_folds - 1}_")
        spec_generator.generate_dataset(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a data specification file for the pig semantic dataset with nested folds",
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
