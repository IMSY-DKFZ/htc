# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest_console_scripts import ScriptRunner

import htc_projects.context.models.data.run_glove_dataset as run_glove_dataset
from htc.models.data.DataSpecification import DataSpecification


class TestGloveDataset:
    specs_name = "pigs_semantic-only_5foldsV2_glove.json"

    def test_specs_reproducible(self, tmp_path: Path, script_runner: ScriptRunner) -> None:
        res = script_runner.run(run_glove_dataset.__file__, "--output-dir", tmp_path)
        assert res.success

        specs_existing = DataSpecification(self.specs_name)
        specs_new = DataSpecification(tmp_path / self.specs_name)
        assert set(specs_existing.paths()).issubset(specs_new.paths())

    def test_glove_occurrence(self) -> None:
        specs = DataSpecification(self.specs_name)
        assert not any("glove" in p.annotated_labels() for p in specs.paths())

        specs.activate_test_set()
        assert not any("glove" in p.annotated_labels() for p in specs.paths("^test$"))
        assert all("glove" in p.annotated_labels() for p in specs.paths("^test_ood$"))
