# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest_console_scripts import ScriptRunner

import htc_projects.sepsis_icu.data.run_sepsis_icu_datasets as run_sepsis_icu_datasets
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings


class TestSpecsSepsis:
    def test_specs_reproducible(self, tmp_path: Path, script_runner: ScriptRunner) -> None:
        res = script_runner.run(run_sepsis_icu_datasets.__file__, "--output-dir", tmp_path)
        assert res.success

        # make sure that all existing json files are regenerated
        dir_existing = settings.htc_projects_dir / "sepsis_icu/data"
        specs_existing = [p.name for p in sorted(dir_existing.glob("*.json"))]
        specs_new = [p.name for p in sorted(tmp_path.glob("*.json"))]
        assert specs_existing == specs_new, "Available sepsis data spec names differ from regenerated spec names."

        # make sure that new and existing specs are the same
        for spec in specs_new:
            spec_existing = DataSpecification(spec)
            spec_new = DataSpecification(tmp_path / spec)
            assert spec_existing == spec_new, "Regenerated sepsis data spec differs from existing spec."

    def test_specs_correctness(self) -> None:
        def check_subjects_subset(
            parent_spec: DataSpecification, child_spec: DataSpecification, train_all: bool = False
        ) -> None:
            child_df = child_spec.table()
            for fold_name, splits in parent_spec:
                for name, paths in splits.items():
                    parent_subjects = {p.subject_name for p in paths}
                    assert len(parent_subjects) == len(set(parent_subjects)), (
                        f"Subjects in parent spec {parent_spec} are not unique."
                    )
                    child_subjects = child_df[(child_df.fold_name == fold_name) & (child_df.split_name == name)][
                        "subject_name"
                    ]
                    if not train_all:
                        assert len(child_subjects) == len(set(child_subjects)), (
                            f"Subjects in child spec {child_spec} are not unique."
                        )
                    assert set(child_subjects).issubset(set(parent_subjects)), (
                        f"Subjects in child spec {child_spec} are not a subset of subjects in parent spec {parent_spec}."
                    )

        def check_overlap_timestamps(
            parent_spec: DataSpecification, child_spec: DataSpecification, groundtruth: str, train_all: bool = False
        ) -> None:
            child_df = child_spec.table()
            for fold_name, splits in parent_spec:
                for name, paths in splits.items():
                    parent_timestamps = {p.timestamp for p in paths}
                    assert len(parent_timestamps) == len(set(parent_timestamps)), (
                        f"Timestamps in parent spec {parent_spec} are not unique."
                    )
                    child_timestamps = child_df.query("fold_name == @fold_name and split_name == @name")["timestamp"]
                    assert len(child_timestamps) == len(set(child_timestamps)), (
                        f"Timestamps in child spec {child_spec} are not unique."
                    )
                    if groundtruth == "distinct":
                        assert len(set(child_timestamps)) + len(set(parent_timestamps)) == len(
                            set(child_timestamps).union(set(parent_timestamps))
                        ), f"Timestamps from child spec {child_spec} can be found in parent spec {parent_spec}."
                    elif groundtruth == "subset":
                        assert set(child_timestamps).issubset(set(parent_timestamps)), (
                            f"Timestamps in child spec {child_spec} are not a subset of timestamps in parent spec"
                            f" {parent_spec}."
                        )
                        if train_all and "val" in name:  # the validation set should not be changed due to train all
                            assert len(set(child_timestamps)) == len(
                                set(child_timestamps).union(set(parent_timestamps))
                            ), f"Timestamps in child spec {child_spec} and parent spec {parent_spec} are not identical."

        parent_spec = DataSpecification("sepsis-inclusion_palm_5folds_test-0.25_seed-0.json")
        sepsis_finger_spec = DataSpecification("sepsis-inclusion_finger_5folds_test-0.25_seed-0.json")
        check_subjects_subset(parent_spec, sepsis_finger_spec)
        check_overlap_timestamps(parent_spec, sepsis_finger_spec, groundtruth="distinct")

        sepsis_finger_train_all_spec = DataSpecification(
            "sepsis-inclusion-train-all_finger_5folds_test-0.25_seed-0.json"
        )
        check_subjects_subset(parent_spec, sepsis_finger_train_all_spec, train_all=True)
        check_overlap_timestamps(parent_spec, sepsis_finger_train_all_spec, groundtruth="distinct")
        check_overlap_timestamps(sepsis_finger_train_all_spec, sepsis_finger_spec, groundtruth="subset", train_all=True)

        sepsis_palm_train_all_spec = DataSpecification("sepsis-inclusion-train-all_palm_5folds_test-0.25_seed-0.json")
        check_subjects_subset(parent_spec, sepsis_palm_train_all_spec, train_all=True)
        check_overlap_timestamps(
            sepsis_finger_train_all_spec, sepsis_palm_train_all_spec, groundtruth="distinct", train_all=True
        )
        check_overlap_timestamps(sepsis_palm_train_all_spec, parent_spec, groundtruth="subset", train_all=True)

        parent_spec = DataSpecification("survival-inclusion_palm_5folds_test-0.25_seed-0.json")
        survival_palm_spec = DataSpecification("survival-inclusion_palm_5folds_test-0.25_seed-0.json")
        check_subjects_subset(parent_spec, survival_palm_spec)
        check_overlap_timestamps(parent_spec, survival_palm_spec, groundtruth="subset")

        survival_finger_spec = DataSpecification("survival-inclusion_finger_5folds_test-0.25_seed-0.json")
        check_subjects_subset(parent_spec, survival_finger_spec)
        check_overlap_timestamps(survival_palm_spec, survival_finger_spec, groundtruth="distinct")

        survival_finger_train_all_spec = DataSpecification(
            "survival-inclusion-train-all_finger_5folds_test-0.25_seed-0.json"
        )
        check_subjects_subset(parent_spec, survival_finger_train_all_spec, train_all=True)
        check_overlap_timestamps(
            survival_finger_train_all_spec, survival_finger_spec, groundtruth="subset", train_all=True
        )

        survival_palm_train_all_spec = DataSpecification(
            "survival-inclusion-train-all_palm_5folds_test-0.25_seed-0.json"
        )
        check_subjects_subset(parent_spec, survival_palm_train_all_spec, train_all=True)
        check_overlap_timestamps(survival_palm_train_all_spec, survival_palm_spec, groundtruth="subset", train_all=True)
        check_overlap_timestamps(
            survival_palm_train_all_spec, survival_finger_train_all_spec, groundtruth="distinct", train_all=True
        )

    def test_nested(self) -> None:
        for classification_target in ["sepsis", "survival"]:
            for target_label in ["palm", "finger"]:
                base_spec = f"{classification_target}-inclusion_{target_label}_5folds_nested-*_seed-0.json"
                all_train_subjects = set()
                all_test_subjects = set()
                for i in range(5):
                    spec = DataSpecification(base_spec.replace("*", str(i)))
                    paths_train = spec.paths()
                    train_subjects = {p.subject_name for p in paths_train}
                    all_train_subjects.update(train_subjects)

                    with spec.activated_test_set():
                        paths_test = spec.paths("test")
                    test_subjects = {p.subject_name for p in paths_test}

                    assert train_subjects.isdisjoint(test_subjects)
                    assert test_subjects.isdisjoint(all_test_subjects)

                    all_test_subjects.update(test_subjects)

                assert len(all_train_subjects) > 0
                assert all_train_subjects == all_test_subjects
