# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from pytest_console_scripts import ScriptRunner

import htc_projects.species.data.run_human_physiological_dataset as run_human_physiological_dataset
import htc_projects.species.data.run_pig_semantic_nested_dataset as run_pig_semantic_nested_dataset
import htc_projects.species.data.run_rat_semantic_dataset as run_rat_semantic_dataset
from htc.models.data.DataSpecification import DataSpecification
from htc_projects.species.settings_species import settings_species


class TestSpecsSpecies:
    @pytest.mark.parametrize(
        "script_path, base_name",
        [
            (
                run_rat_semantic_dataset.__file__,
                "rat_semantic-only_5folds_nested-*-2_mapping-12_seed-0.json",
            ),
            (
                run_human_physiological_dataset.__file__,
                "human_semantic-only_physiological-kidney_5folds_nested-*-2_mapping-12_seed-0.json",
            ),
            (
                run_pig_semantic_nested_dataset.__file__,
                "pig_semantic-only_5folds_nested-*-2_mapping-12_seed-0.json",
            ),
        ],
    )
    def test_specs_semantic_reproducible(
        self, tmp_path: Path, script_runner: ScriptRunner, script_path: str, base_name: str
    ) -> None:
        res = script_runner.run(script_path, "--output-dir", tmp_path)
        assert res.success

        max_index = settings_species.n_nested_folds - 1
        for i in range(settings_species.n_nested_folds):
            specs_name = base_name.replace(f"nested-*-{max_index}", f"nested-{i}-{max_index}")
            specs_existing = DataSpecification(specs_name)
            specs_new = DataSpecification(tmp_path / specs_name)
            assert specs_existing == specs_new
