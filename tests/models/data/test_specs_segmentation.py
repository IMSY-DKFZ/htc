# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
from multiprocessing import Pool, set_start_method
from pathlib import Path

import pytest
from pytest_console_scripts import ScriptRunner

import htc.models.data.run_pig_dataset as run_pig_dataset
from htc.models.data.DataSpecification import DataSpecification
from htc.models.data.run_pig_dataset import extract_paths_known, filter_train
from htc.settings import settings
from htc.tivita.DataPath import DataPath


class TestSpecsSegmentation:
    specs_name = "pigs_semantic-only_5foldsV2.json"

    def test_specs_segmentation(self) -> None:
        from htc.models.data.run_pig_dataset import test_set

        specs = DataSpecification(self.specs_name)

        for p in specs.paths():
            assert p.subject_name not in test_set

        for fold_name, datasets in specs:
            # No overlap between train and validation unknown
            pigs_train = {p.subject_name for p in datasets["train_semantic"]}
            subject_names_val = fold_name.removeprefix("fold_").split(",")

            for p in datasets["val_semantic_unknown"]:
                assert p.subject_name not in pigs_train
                assert p.subject_name in subject_names_val

            for p in datasets["val_semantic_known"]:
                assert p.subject_name in pigs_train

    def test_specs_reproducible(self, tmp_path: Path, script_runner: ScriptRunner) -> None:
        res = script_runner.run(run_pig_dataset.__file__, "--output-dir", tmp_path)
        assert res.success

        specs_existing = DataSpecification(self.specs_name)
        specs_new = DataSpecification(tmp_path / self.specs_name)
        assert specs_existing == specs_new

    @pytest.mark.parametrize("fold_name", ["P041", "P044,P047,P057,P060"])
    def test_extract_paths_known(self, fold_name: str):
        set_start_method(
            "spawn", force=True
        )  # This is crucial as it simulates a fresh start of the Python interpreter which can lead to a different ordering of the set method

        # Some repetition
        paths = [self._get_paths(fold_name), self._get_paths(fold_name)]
        assert paths[0]["pid"] == paths[1]["pid"]

        # Also add some multiprocessing results
        p = Pool()
        paths_mp = p.map(self._get_paths, [fold_name, fold_name, fold_name, fold_name])
        p.close()
        p.join()

        pids = [p["pid"] for p in paths_mp]
        assert len(set(pids)) == len(pids)

        paths += paths_mp
        for p in paths:
            del p["pid"]

        # Test whether in all cases the same paths were selected
        for i in range(1, len(paths)):
            assert paths[i] == paths[0]

    def _get_paths(self, fold_name: str) -> dict[str, list[DataPath] | int]:
        paths = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))
        validation_pigs = fold_name.split(",")

        paths_train = [p for p in paths if p.subject_name not in validation_pigs]
        paths_val_unknown = [p for p in paths if p.subject_name in validation_pigs]
        paths_val_known = extract_paths_known(paths_train, fold_index=1337)

        return {
            "train": paths_train,
            "val_unknown": paths_val_unknown,
            "val_known": paths_val_known,
            "pid": os.getpid(),
        }
