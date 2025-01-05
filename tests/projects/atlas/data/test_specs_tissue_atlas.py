# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest_console_scripts import ScriptRunner

import htc_projects.atlas.data.run_atlas_dataset as run_atlas_dataset
from htc.models.data.DataSpecification import DataSpecification
from htc_projects.atlas.data.run_atlas_dataset import SpecsGenerationTissueAtlas
from htc_projects.atlas.tables import median_cam_table


class TestSpecsTissueAtlas:
    specs_name = "tissue-atlas_loocv_test-8_seed-0_cam-118.json"

    def test_specs_atlas(self) -> None:
        df = median_cam_table()
        specs = DataSpecification(self.specs_name)
        specs.activate_test_set()
        all_fold_names = []
        common_test_paths = None

        for fold_name, datasets in specs:
            # All sets must be non-overlapping
            paths_train = datasets["train"]
            paths_val = datasets["val"]
            paths_test = datasets["test"]
            assert len({p.subject_name for p in paths_test}.intersection({p.subject_name for p in paths_train})) == 0
            assert len({p.subject_name for p in paths_test}.intersection({p.subject_name for p in paths_val})) == 0
            assert len({p.subject_name for p in paths_train}.intersection({p.subject_name for p in paths_val})) == 0

            # Check that the test set is identical across all folds
            if common_test_paths is None:
                common_test_paths = paths_test
            else:
                assert common_test_paths == paths_test, "The test set must be the same across all folds"

            # Check that no pig is in train and validation (based on the fold name)
            val_pigs = fold_name.removeprefix("fold_").split(",")
            assert all(path.subject_name not in val_pigs for path in datasets["train"])

            # Check that every image occurs only once
            all_timestamps = []
            for name in specs.split_names():
                for path in datasets[name]:
                    assert path.timestamp not in all_timestamps
                    all_timestamps.append(path.timestamp)

            assert set(df["timestamp"].unique()) == set(all_timestamps), (
                "The data specification is different from the data table"
            )
            all_fold_names.append(fold_name)

        assert sorted(set(all_fold_names)) == sorted(all_fold_names), "The folds are not allowed to overlap in the pigs"

    def test_random_generation(self) -> None:
        d1_seed0 = SpecsGenerationTissueAtlas(n_pigs=8, seed=0)
        d2_seed0 = SpecsGenerationTissueAtlas(n_pigs=8, seed=0)
        d1_seed1 = SpecsGenerationTissueAtlas(n_pigs=8, seed=1)
        d2_seed1 = SpecsGenerationTissueAtlas(n_pigs=8, seed=1)

        assert set(d1_seed0.test_pigs) == set(d2_seed0.test_pigs)
        assert set(d1_seed1.test_pigs) == set(d2_seed1.test_pigs)
        assert set(d1_seed0.test_pigs) != set(d1_seed1.test_pigs)
        assert set(d2_seed0.test_pigs) != set(d2_seed1.test_pigs)

    def test_specs_reproducible(self, tmp_path: Path, script_runner: ScriptRunner) -> None:
        res = script_runner.run(run_atlas_dataset.__file__, "--output-dir", tmp_path)
        assert res.success

        specs_existing = DataSpecification(self.specs_name)
        specs_new = DataSpecification(tmp_path / self.specs_name)
        assert specs_existing == specs_new
