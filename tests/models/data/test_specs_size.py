# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest_console_scripts import ScriptRunner

import htc.models.data.run_size_dataset as run_size_dataset
from htc.models.data.DataSpecification import DataSpecification
from htc.models.data.run_size_dataset import label_mapping_dataset_size
from htc.settings_seg import settings_seg


class TestSpecsSize:
    specs_name = "pigs_semantic-only_dataset-size_repetitions=5V2.json"

    def test_specs_size(self) -> None:
        from htc.models.data.run_pig_dataset import train_set

        specs = DataSpecification(self.specs_name)

        n_seed_pigs = len(train_set) - 1
        n_seeds = 5
        assert len(specs) == n_seed_pigs * n_seeds + 1, "For the full dataset we have only one seed"
        assert specs.split_names() == ["train_semantic", "val_semantic_test"]

        paths_test = None
        for fold_name, datasets in specs:
            assert "val_semantic_test" in datasets
            if paths_test is None:
                paths_test = datasets["val_semantic_test"]
            else:
                assert paths_test == datasets["val_semantic_test"], (
                    "The validation test set must be the same across folds"
                )

            assert "train_semantic" in datasets

        for n_pigs in range(1, n_seed_pigs + 1):
            for seed in range(n_seeds):
                paths_seed = sorted(specs.folds[f"fold_pigs={n_pigs}_seed={seed}"]["train_semantic"])
                for seed_other in range(n_seeds):
                    if seed != seed_other:
                        paths_seed_other = sorted(
                            specs.folds[f"fold_pigs={n_pigs}_seed={seed_other}"]["train_semantic"]
                        )
                        assert paths_seed != paths_seed_other, (
                            f"There should be different paths across seeds ({seed} and {seed_other} are the same for"
                            f" n_pigs={n_pigs})"
                        )

    def test_label_mapping_dataset_size(self) -> None:
        mapping = label_mapping_dataset_size()
        assert len(mapping) == 8
        assert mapping.label_names() == [
            "background",
            "colon",
            "liver",
            "peritoneum",
            "skin",
            "small_bowel",
            "spleen",
            "stomach",
        ]
        assert mapping.last_valid_label_index == 7

        mapping_settings = settings_seg.label_mapping
        background_names = [
            label_name
            for label_name, label_index in mapping.mapping_name_index.items()
            if label_index == mapping.name_to_index("background")
        ]
        background_names_settings = [
            label_name
            for label_name, label_index in mapping_settings.mapping_name_index.items()
            if label_index == mapping_settings.name_to_index("background")
        ]
        assert background_names == background_names_settings

    def test_specs_reproducible(self, tmp_path: Path, script_runner: ScriptRunner) -> None:
        res = script_runner.run(run_size_dataset.__file__, "--output-dir", tmp_path)
        assert res.success

        specs_existing = DataSpecification(self.specs_name)
        specs_new = DataSpecification(tmp_path / self.specs_name)
        assert specs_existing == specs_new
