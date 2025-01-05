# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest

from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings


@pytest.mark.parametrize("specs_path", sorted(settings.src_dir.rglob("data/*.json")))
def test_specs(specs_path: Path, check_sepsis_data_accessible: Callable, check_human_data_accessible: Callable) -> None:
    check_sepsis_data_accessible()
    check_human_data_accessible()

    # Load our schema definition
    with (settings.htc_package_dir / "models" / "data" / "data_spec.schema").open() as f:
        schema = json.load(f)
    with specs_path.open() as f:
        spec_raw = json.load(f)

    try:
        jsonschema.validate(instance=spec_raw, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise AssertionError(f"Error in {specs_path.relative_to(settings.src_dir)}") from e

    # General test applied to all data specification files
    specs = DataSpecification(specs_path)
    specs.activate_test_set()

    all_fold_names = []
    for fold_name, datasets in specs:
        # Collect test set
        test_set_names = [x for x in specs.split_names() if x.startswith("test")]

        timestamps_test = set()
        subject_names_test = set()
        for name in test_set_names:
            for p in datasets[name]:
                if hasattr(p, "timestamp"):
                    timestamps_test.update([p.timestamp])
                if hasattr(p, "subject_name"):
                    subject_names_test.update([p.subject_name])

        # Check that every image occurs only once and that there is no overlap with the test set
        split_subjects = {n: set() for n in specs.split_names()}
        all_timestamps = []
        for name, paths in datasets.items():
            for path in paths:
                if hasattr(path, "subject_name"):
                    split_subjects[name].add(path.subject_name)
                if hasattr(path, "timestamp"):
                    if name.startswith("test"):
                        assert path.timestamp in timestamps_test
                        assert path.subject_name in subject_names_test
                    else:
                        assert path.timestamp not in timestamps_test
                        assert path.subject_name not in subject_names_test

                        assert path.timestamp not in all_timestamps, "Duplicate timestamp in non-test set"

                    all_timestamps.append(path.timestamp)

        # Check that borders are always at subject boundaries (except for special cases where we define exceptions here)
        exceptions_global = ["val_semantic_known", "test_first", "test_all"]
        exceptions_specs = {
            "pigs_semantic-only_5foldsV2_glove": ["val_semantic_known", "test_ood"],
        }
        split_names = [
            n
            for n in specs.split_names()
            if n not in exceptions_specs.get(specs_path.stem, []) and n not in exceptions_global
        ]
        for split_name1 in split_names:
            for split_name2 in split_names:
                if split_name1 != split_name2:
                    assert not split_subjects[split_name1].intersection(split_subjects[split_name2]), (
                        f"Overlap of subjects between {split_name1} and {split_name2} for the specs {specs_path}. If"
                        " this is expected for this specification file, add an exception to this test."
                    )

        all_fold_names.append(fold_name)

    assert sorted(set(all_fold_names)) == sorted(all_fold_names), "fold_name must be unique"
    assert all_fold_names == specs.fold_names()


@pytest.mark.parametrize(
    "base_name",
    [
        "rat_semantic-only_5folds_nested-*-2_mapping-12_seed-0.json",
        "human_semantic-only_physiological-kidney_5folds_nested-*-2_mapping-12_seed-0.json",
        "pig_semantic-only_5folds_nested-*-2_mapping-12_seed-0.json",
    ],
)
def test_specs_nested(base_name: str) -> None:
    all_subjects = set()
    all_test_subjects = set()

    match = re.search(r"nested-\*-(\d+)", base_name)
    assert match is not None
    n_nested = int(match.group(1)) + 1

    for i in range(n_nested):
        spec = DataSpecification(base_name.replace("nested-*", f"nested-{i}"))
        spec.activate_test_set()
        train_subjects = {p.subject_name for p in spec.paths()}
        test_subjects = {p.subject_name for p in spec.paths("test")}
        assert len(train_subjects) > 0
        assert len(test_subjects) > 0

        if len(all_subjects) > 0:
            assert all_subjects == train_subjects

        all_subjects.update(train_subjects)
        all_test_subjects.update(test_subjects)

        if i < n_nested - 1:
            assert len(all_test_subjects) < len(all_subjects)

    assert all_subjects == all_test_subjects
