# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from htc.models.data.DataSpecification import DataSpecification
from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.utils.helper_functions import median_table
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu
from htc_projects.sepsis_icu.tables import first_inclusion


class SpecsGenerationAdmission(SpecsGeneration):
    target_label = "palm"

    def __init__(
        self,
        classification_target: str,
        test_size: float = None,
        test_subjects: list[str] = None,
        seed: int = 0,
        definite_test_subjects: list[str] = None,
    ):
        self.n_folds = 5
        self.col_name = self.column_name(classification_target)

        self.df = first_inclusion(classification_target, self.target_label)
        self.df = self.df[~self.df.isin(settings_sepsis_icu.exclusion_subjects)]
        assert len(self.df) == self.df["subject_name"].nunique(), "Each subject must have exactly one image"

        if test_size is not None:
            assert test_subjects is None, "test_subjects must be None if test_size is not None"
            np.random.seed(seed)
            set_generator = StratifiedGroupKFold(n_splits=math.ceil(1 / test_size), shuffle=True)
            train_indices, test_indices = next(
                iter(set_generator.split(self.df["image_name"], self.df[self.col_name], self.df["subject_name"]))
            )

            # We may want to include specific subjects in the test set, e.g. sepsis_onset_patients
            if definite_test_subjects is None:
                definite_test_subjects = []
            self.test_subjects = sorted(
                set(self.df.iloc[test_indices]["subject_name"].unique().tolist() + definite_test_subjects)
            )
            self.train_subjects = sorted(
                set(self.df.iloc[train_indices]["subject_name"].unique().tolist()) - set(definite_test_subjects)
            )

            assert set(self.test_subjects).isdisjoint(self.train_subjects), "Test and train subjects must be disjoint"
            print(
                f"Statistics for {classification_target}: {len(self.train_subjects)} train subjects,"
                f" {len(self.test_subjects)} test subjects"
            )
            assert set(self.df.query("subject_name in @self.test_subjects")[self.col_name]) == set(
                self.df[self.col_name]
            ), "All labels must be represented in the test set"

            super().__init__(
                name=f"{classification_target}-inclusion_{self.target_label}_{self.n_folds}folds_test-{test_size}_seed-{seed}"
            )
        elif test_subjects is not None:
            assert test_size is None, "test_size must be None if test_subjects is not None"

            self.test_subjects = test_subjects
            self.train_subjects = sorted(set(self.df["subject_name"].unique()) - set(test_subjects))

            super().__init__(
                name=f"{classification_target}-inclusion_{self.target_label}_{self.n_folds}folds_seed-{seed}"
            )
        else:
            raise ValueError("Either test_size or test_subjects must be provided")

    def generate_folds(self) -> list[dict]:
        data_specs = []

        df_test = self.df.query("subject_name in @self.test_subjects")
        assert len(df_test) == df_test["image_name"].nunique(), "Images must be unique"
        imgs_test = sorted((df_test["image_name"] + "@" + df_test["annotation_name"]).tolist())

        df_splits = self.df.query("subject_name in @self.train_subjects")
        df_splits = df_splits.reset_index(drop=True)

        fold_generator = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True)
        for i, (train_indices, val_indices) in enumerate(
            fold_generator.split(df_splits["image_name"], df_splits[self.col_name], df_splits["subject_name"])
        ):
            subjects_train = df_splits.iloc[train_indices]["subject_name"].unique()
            subjects_val = df_splits.iloc[val_indices]["subject_name"].unique()
            assert set(subjects_train).isdisjoint(subjects_val), "Test and validation subjects must be disjoint"

            df_train = self.df.query("subject_name in @subjects_train")
            df_train = df_train.reset_index(drop=True)

            df_val = self.df.query("subject_name in @subjects_val")
            df_val = df_val.reset_index(drop=True)

            fold_specs = {
                "fold_name": f"fold_{i}",
                "train": {
                    "image_names": sorted((df_train["image_name"] + "@" + df_train["annotation_name"]).tolist()),
                },
                "val": {
                    "image_names": sorted((df_val["image_name"] + "@" + df_val["annotation_name"]).tolist()),
                },
                "test": {
                    "image_names": imgs_test,
                },
            }

            data_specs.append(fold_specs)

        return data_specs

    @staticmethod
    def column_name(classification_target: str) -> str:
        if classification_target == "survival":
            return "survival_30_days_post_inclusion"
        elif classification_target == "sepsis":
            return "sepsis_status"
        elif classification_target == "septic_shock":
            return "septic_shock"
        elif classification_target == "shock":
            return "shock"
        else:
            raise ValueError(f"Unknown classification_target: {classification_target}")


def generate_nested(n_nested_folds: int, classification_target: str, output_dir: Path, seed: int = 0) -> None:
    df = first_inclusion(classification_target, SpecsGenerationAdmission.target_label)
    df = df[~df.isin(settings_sepsis_icu.exclusion_subjects)]
    assert len(df) == df["subject_name"].nunique(), "Each subject must have exactly one image"

    np.random.seed(seed)
    fold_generator = StratifiedGroupKFold(n_splits=n_nested_folds, shuffle=True)
    for i, (_, test_indices) in enumerate(
        fold_generator.split(
            df["image_name"], df[SpecsGenerationAdmission.column_name(classification_target)], df["subject_name"]
        )
    ):
        print(f"Generating nested fold {i} for target {classification_target}...")
        test_subjects = sorted(set(df.iloc[test_indices]["subject_name"].unique().tolist()))
        spec_generator = SpecsGenerationAdmission(
            test_subjects=test_subjects, classification_target=classification_target, seed=seed
        )
        spec_generator.name = spec_generator.name.replace("folds_", f"folds_nested-{i}_")
        spec_generator.generate_dataset(output_dir)

        SpecsGenerationAdaptation(
            base_name=spec_generator.name,
            train_all=False,
            target_label="finger",
            classification_target=classification_target,
        ).generate_dataset(output_dir)


class SpecsGenerationAdaptation(SpecsGeneration):
    def __init__(
        self,
        base_name: str,
        train_all: bool = False,
        target_label: str = "finger",
        classification_target: str = "sepsis",
    ):
        assert classification_target in ["sepsis", "survival", "septic_shock", "shock"], (
            "classification_target must be 'sepsis', 'survival', 'septic_shock' or 'shock'"
        )
        assert base_name.startswith(classification_target), "base_name must start with classification_target"
        self.train_all = train_all
        self.spec = DataSpecification(base_name)
        self.spec.activate_test_set()

        # adapt name of spec file
        new_name = self.spec.name().replace("palm", target_label)
        if self.train_all:
            new_name = new_name.replace("-inclusion", "-inclusion-train-all")
        super().__init__(name=new_name)

        # generate dataframe for train all for given target label and target class
        self.df_all = median_table(dataset_name="2022_10_24_Tivita_sepsis_ICU#subjects")
        self.df_all = self.df_all.query("label_name == @target_label")
        if classification_target == "sepsis":
            self.df_all = self.df_all.query("sepsis_status in ['no_sepsis', 'sepsis']")
        else:
            self.df_all = self.df_all[~pd.isna(self.df_all.survival_30_days_post_inclusion)]

        # generate dataframe for training on first timepoint for target label and target class
        self.df = first_inclusion(classification_target, target_label)

    def generate_folds(self) -> list[dict]:
        data_specs = []

        for fold_name, splits in self.spec:
            fold_specs = {"fold_name": fold_name}
            for name, paths in splits.items():
                subjects = {p.subject_name for p in paths}  # subjects from "parent" (sepsis-inclusion_palm spec)
                if name.startswith("train") and self.train_all:  # train_all should only affect training, not validation
                    df_subjects = self.df_all.query("subject_name in @subjects")
                elif name == "test_all":
                    df_subjects = self.df_all.query("subject_name in @subjects")
                else:
                    df_subjects = self.df.query("subject_name in @subjects")

                fold_specs[name] = {
                    "image_names": sorted(df_subjects["image_name"] + "@" + df_subjects["annotation_name"])
                }

            data_specs.append(fold_specs)

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Creates the data specification file for the sepsis study which only includes the first image for each"
            " patient."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory where the resulting data specification file should be stored.",
    )
    parser.add_argument(
        "--test-subjects", type=float, default=0.25, help="Ratio of subjects to include in the test set."
    )

    args = parser.parse_args()

    generate_nested(n_nested_folds=5, classification_target="sepsis", output_dir=args.output_dir)
    generate_nested(n_nested_folds=5, classification_target="survival", output_dir=args.output_dir)
    generate_nested(n_nested_folds=5, classification_target="septic_shock", output_dir=args.output_dir)
    generate_nested(n_nested_folds=5, classification_target="shock", output_dir=args.output_dir)

    # generate parent sepc files
    SpecsGenerationAdmission(classification_target="sepsis", test_size=args.test_subjects).generate_dataset(
        args.output_dir
    )
    SpecsGenerationAdmission(classification_target="survival", test_size=args.test_subjects).generate_dataset(
        args.output_dir
    )
    SpecsGenerationAdmission(classification_target="septic_shock", test_size=args.test_subjects).generate_dataset(
        args.output_dir
    )
    SpecsGenerationAdmission(classification_target="shock", test_size=args.test_subjects).generate_dataset(
        args.output_dir
    )

    # generate child spec files
    for classification_target in ["sepsis", "survival", "septic_shock", "shock"]:
        for target_label in ["palm", "finger"]:
            print(f"Adapting specs for {classification_target} and {target_label} label...")
            for train_all in [True, False]:
                SpecsGenerationAdaptation(
                    base_name=f"{classification_target}-inclusion_palm_5folds_test-0.25_seed-0",
                    train_all=train_all,
                    target_label=target_label,
                    classification_target=classification_target,
                ).generate_dataset(args.output_dir)
