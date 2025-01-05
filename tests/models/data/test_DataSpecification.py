# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import io
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from htc.models.data.DataSpecification import DataSpecification
from htc.models.data.run_pig_dataset import filter_train
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.Config import Config


class TestDataSpecification:
    @pytest.fixture(scope="function")
    def data_specs(self) -> DataSpecification:
        return DataSpecification("pigs_semantic-only_5foldsV2.json")

    def test_name(self, data_specs: DataSpecification) -> None:
        assert data_specs.name() == "pigs_semantic-only_5foldsV2"

    def test_paths(self, data_specs: DataSpecification) -> None:
        paths_train = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))
        paths_spec = data_specs.paths()

        assert type(paths_spec) == list
        assert paths_spec == paths_train
        assert 0 < len(data_specs.paths("val_semantic_known")) < len(data_specs.paths("val_semantic_unknown"))
        assert len(set(data_specs.paths("train") + data_specs.paths("val_semantic_known"))) == len(
            data_specs.paths("val")
        )
        assert len(data_specs.paths("train|val_semantic_known")) == len(data_specs.paths("val"))
        assert len(data_specs.paths("val")) == len(data_specs.paths("val_semantic"))

        assert data_specs.fold_paths("fold_P041,P060,P069", "train_semantic") == sorted(
            data_specs.folds["fold_P041,P060,P069"]["train_semantic"]
        )

        data_specs.activate_test_set()
        for fold_name in data_specs.fold_names():
            assert data_specs.fold_paths(fold_name, "^test") == data_specs.paths("^test")

    def test_paths_test(self, data_specs: DataSpecification) -> None:
        specs_json = """
        [
            {
                "fold_name": "fold",
                "train": {
                    "image_names": ["P044#2020_02_01_09_51_15"]
                },
                "val": {
                    "data_path_class": "htc.tivita.DataPath>DataPath",
                    "image_names": ["P045#2020_02_05_16_51_41"]
                },
                "test": {
                    "image_names": ["P059#2020_05_14_12_50_10"]
                }
            }
        ]
        """
        specs = DataSpecification(io.StringIO(specs_json))
        specs.deactivate_test_set()
        specs.deactivate_test_set()  # Does not do anything if already deactivated
        assert specs.paths() == [
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
            DataPath.from_image_name("P045#2020_02_05_16_51_41"),
        ]

        specs.activate_test_set()
        assert specs.paths() == [
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
            DataPath.from_image_name("P045#2020_02_05_16_51_41"),
            DataPath.from_image_name("P059#2020_05_14_12_50_10"),
        ]
        specs.activate_test_set()  # No change when activating a second time
        assert specs.paths() == [
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
            DataPath.from_image_name("P045#2020_02_05_16_51_41"),
            DataPath.from_image_name("P059#2020_05_14_12_50_10"),
        ]

        specs.deactivate_test_set()
        assert specs.paths() == [
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
            DataPath.from_image_name("P045#2020_02_05_16_51_41"),
        ]

        with specs.activated_test_set():
            assert specs.paths() == [
                DataPath.from_image_name("P044#2020_02_01_09_51_15"),
                DataPath.from_image_name("P045#2020_02_05_16_51_41"),
                DataPath.from_image_name("P059#2020_05_14_12_50_10"),
            ]
        assert specs.paths() == [
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
            DataPath.from_image_name("P045#2020_02_05_16_51_41"),
        ]

        # Activate again for the rest of the test
        specs.activate_test_set()
        assert specs.paths() == [
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
            DataPath.from_image_name("P045#2020_02_05_16_51_41"),
            DataPath.from_image_name("P059#2020_05_14_12_50_10"),
        ]

        # Ordering of the folds should stay the same
        folds_before = copy.deepcopy(data_specs.folds)
        data_specs.activate_test_set()
        assert list(folds_before.keys()) == list(data_specs.folds.keys())

        df_table = pd.DataFrame(
            [
                ["fold", "train", "P044#2020_02_01_09_51_15", "P044", "2020_02_01_09_51_15", "semantic#primary"],
                ["fold", "val", "P045#2020_02_05_16_51_41", "P045", "2020_02_05_16_51_41", "semantic#primary"],
                ["fold", "test", "P059#2020_05_14_12_50_10", "P059", "2020_05_14_12_50_10", "semantic#primary"],
            ],
            columns=["fold_name", "split_name", "image_name", "subject_name", "timestamp", "annotation_name"],
        )
        assert_frame_equal(df_table, specs.table())

        assert specs.fold_paths("fold") == specs.paths()
        assert specs.fold_paths("fold", "^train") == specs.paths("^train")
        assert specs.fold_paths("fold", "non_existing_dataset") == []

    def test_fold_names(self, data_specs: DataSpecification) -> None:
        fold_names = data_specs.fold_names()
        assert "fold_P041,P060,P069" in fold_names
        assert all(f.startswith("fold") for f in fold_names)
        assert len(data_specs) == len(fold_names)

    def test_dataset_names(self, data_specs: DataSpecification) -> None:
        assert data_specs.split_names() == ["train_semantic", "val_semantic_unknown", "val_semantic_known"]

    def test_iterator(self, data_specs: DataSpecification) -> None:
        fold_name, fold_datasets = next(iter(data_specs))
        assert fold_name == "fold_P041,P060,P069"
        assert list(fold_datasets.keys()) == data_specs.split_names()

        for name in data_specs.split_names():
            assert all(isinstance(p, DataPath) for p in fold_datasets[name])

    def test_path_specs_variations(self) -> None:
        DataSpecification("pigs_semantic-only_5foldsV2.json")
        DataSpecification("data/pigs_semantic-only_5foldsV2.json")

        with pytest.raises(FileNotFoundError):
            DataSpecification("some/invalid/path")

    def test_from_config_to_json(self, data_specs, tmp_path: Path) -> None:
        config = Config({"input/data_spec": "pigs_semantic-only_5foldsV2.json"})
        spec_config = DataSpecification.from_config(config)
        assert spec_config == data_specs
        assert isinstance(config["input/data_spec"], DataSpecification)

        json_str = json.dumps(config.data, cls=AdvancedJSONEncoder)
        assert json.loads(json_str) == {"input": {"data_spec": "pigs_semantic-only_5foldsV2.json"}}

        shutil.copy2(spec_config.path, tmp_path / spec_config.path.name)
        config_tmp = Config({"input/data_spec": tmp_path / spec_config.path.name})
        spec_tmp = DataSpecification.from_config(config_tmp)
        assert spec_tmp == spec_config

        # The absolute path to the config is used when the data specs lies outside of this repository
        json_str = json.dumps(config_tmp.data, cls=AdvancedJSONEncoder)
        assert json.loads(json_str) == {"input": {"data_spec": str(tmp_path / spec_config.path.name)}}
