# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import json
import os
import pprint
from multiprocessing import Pool, set_start_method
from multiprocessing.managers import DictProxy
from pathlib import Path

import jsonschema

from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.transforms import HTCTransformation
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


class ConfigAccessor:
    def __init__(self, config: Config):
        self.config = config

    def access_keys(self, key: str) -> None:
        self.config[key]
        return os.getpid()


class TestConfig:
    def test_used_keys(self) -> None:
        config = Config({"a/b": 1, "a/bb": 2, "c": 3, "e/f/g": 4, "i": 5})

        assert config.keys() == ["a/b", "a/bb", "c", "e/f/g", "i"]

        config["c"]
        assert config.used_keys() == ["c"]
        assert config.unused_keys() == sorted(["a/b", "a/bb", "e/f/g", "i"])

        config["a"]
        assert config.used_keys() == sorted(["c", "a"])
        assert config.unused_keys() == sorted(["e/f/g", "i"])

        config["e/f"]
        assert config.used_keys() == sorted(["c", "a", "e/f"])
        assert config.unused_keys() == sorted(["i"])

        config["h"] = 6
        assert config.used_keys() == sorted(["c", "a", "e/f", "h"])
        assert config.unused_keys() == sorted(["i"])

        del config["i"]
        assert config.used_keys() == sorted(["c", "a", "e/f", "h", "i"])
        assert config.unused_keys() == []

        assert "j" not in config
        assert config.used_keys() == sorted(["c", "a", "e/f", "h", "i", "j"])
        assert config.unused_keys() == []

    def test_nested(self) -> None:
        config = Config({"a": 1, "a_x": 2})

        config["a"]
        config["a_x"]
        assert config.unused_keys() == []

        config = Config({"a": 1, "a_x": 2})

        config["a"]
        assert config.unused_keys() == ["a_x"]

    def test_multiprocessing(self) -> None:
        set_start_method("spawn", force=True)

        config = Config({"a/b": 1, "c": 2}, use_shared_dict=True)

        assert config.keys() == ["a/b", "c"]

        ca = ConfigAccessor(config)

        p = Pool()
        pids = p.map(ca.access_keys, ["a/b", "c"])
        p.close()
        p.join()

        assert len(set(pids)) == 2, "Different processes should have been used"
        assert config.used_keys() == ["a/b", "c"]

    def test_inheritance(self) -> None:
        config1 = Config.from_model_name(config_name="default_rgb", model_name="pixel")
        config2 = Config({"inherits": "pixel/configs/default_rgb"})
        config3 = Config({"inherits": "models/pixel/configs/default_rgb"})

        assert config1 == config2 and config1 == config3
        assert config1["input/n_channels"] == 3
        assert type(config1._used_keys) == dict

        config4 = Config({"inherits": "pixel/configs/default_rgb"}, use_shared_dict=True)
        assert type(config4._used_keys) == DictProxy

    def test_from_model_name(self) -> None:
        config1 = Config.from_model_name(config_name="default", model_name="pixel")
        config2 = Config.from_model_name(model_name="pixel")
        config3 = Config.from_model_name(config_name=config2.path_config)
        config4 = Config.from_model_name(config_name="pixel/configs/default.json")
        config5 = Config.from_model_name(config_name="models/pixel/configs/default.json")

        assert config1 == config2 == config3 == config4 == config5

    def test_config_loading(self) -> None:
        config1 = Config(settings.htc_package_dir / "models" / "pixel" / "configs" / "default.json")
        config2 = Config("pixel/configs/default.json")
        config3 = Config("models/pixel/configs/default.json")
        config4 = Config("htc/models/pixel/configs/default.json")
        config5 = Config.from_model_name(config_name="default", model_name="pixel")

        assert config1 == config2 == config3 == config4 == config5

    def test_validate_configs_schema(self) -> None:
        # Load our schema definition
        with (settings.htc_package_dir / "utils" / "config.schema").open() as f:
            schema = json.load(f)

        errors = []
        n_lightning_checks = 0
        n_mapping_checks = 0
        n_spec_checks = 0
        n_transform_checks = 0
        n_image_mapping_checks = 0
        files = sorted(settings.src_dir.rglob("configs/*.json"))
        assert len(files) > 0, "Could not find any config files"
        for config_file in files:
            if config_file.name.startswith("generated"):
                continue

            # We use the Config class to load the json data so that the inheritance gets resolved
            config = Config(config_file)
            config_rel_path = config_file.relative_to(settings.src_dir)

            try:
                jsonschema.validate(instance=config.data, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                errors.append(f"{config_rel_path}: {e}")

            # Check whether references to classes are valid
            if "lightning_class" in config:
                n_lightning_checks += 1
                try:
                    HTCLightning.class_from_config(config)
                except Exception as e:
                    errors.append(f"{config_rel_path}: {e}")

            if "label_mapping" in config:
                n_mapping_checks += 1
                try:
                    LabelMapping.from_config(config)
                except Exception as e:
                    errors.append(f"{config_rel_path}: {e}")

            if "input/data_spec" in config:
                n_spec_checks += 1
                try:
                    DataSpecification.from_config(config)
                except Exception as e:
                    errors.append(f"{config_rel_path}: {e}")

            for t_key in [
                "input/transforms_cpu",
                "input/transforms_gpu",
                "input/test_time_transforms_cpu",
                "input/test_time_transforms_gpu",
            ]:
                if t_key in config:
                    n_transform_checks += 1
                    try:
                        HTCTransformation.parse_transforms(config[t_key], config=config, device="cpu")
                    except Exception as e:
                        errors.append(f"{config_rel_path}: {e}")

            if "input/image_labels" in config:
                for i, label_settings in enumerate(config["input/image_labels"]):
                    if "image_label_mapping" in label_settings:
                        n_image_mapping_checks += 1
                        try:
                            LabelMapping.from_config(config, image_label_entry_index=i)
                        except Exception as e:
                            errors.append(f"{config_rel_path}: {e}")

        assert n_lightning_checks > 0, "No lightning class checks performed"
        assert n_mapping_checks > 0, "No label mapping checks performed"
        assert n_spec_checks > 0, "No data specification checks performed"
        assert n_transform_checks > 0, "No transformation checks performed"
        assert n_image_mapping_checks > 0, "No image label mapping checks performed"
        assert len(errors) == 0, pprint.pformat(errors)

    def test_copy(self) -> None:
        config = Config({
            "my/value": 1,
            "my/list": [1, 2],
            "my/nested_list": [[1, 2]],
            "input/data_spec": "pigs_semantic-only_5foldsV2.json",
        })
        DataSpecification.from_config(config)
        config_copy = copy.copy(config)
        assert config == config_copy

        assert config_copy["my/value"] == 1
        config_copy["my/value"] = 2
        assert config_copy["my/value"] == 2
        assert config["my/value"] == 1

        config_copy["my/list"].append(3)
        assert config_copy["my/list"] == [1, 2, 3]
        assert config["my/list"] == [1, 2]

        config_copy["my/nested_list"][0].append(3)
        assert config_copy["my/nested_list"][0] == [1, 2, 3]
        assert config["my/nested_list"][0] == [1, 2]

        assert config["input/data_spec"] == config_copy["input/data_spec"]

    def test_suffix(self) -> None:
        config1 = Config("htc_projects/context/models/configs/organ_transplantation_0.8.json")
        config2 = Config("htc_projects/context/models/configs/organ_transplantation_0.8")
        config3 = Config({"inherits": "htc_projects/context/models/configs/organ_transplantation_0.8.json"})
        config4 = Config({"inherits": "htc_projects/context/models/configs/organ_transplantation_0.8"})

        assert config1 == config2 == config3 == config4

    def test_extend_lists(self, tmp_path: Path) -> None:
        config = Config({
            "input/my_list": [1, 2],
            "input/my_list_extends": [3],
            "input/skip_me": 10,
        })
        assert config.keys() == ["input/my_list", "input/skip_me"]
        assert config["input/my_list"] == [1, 2, 3]

        config.save_config(tmp_path / "config.json")
        config2 = Config({
            "inherits": str(tmp_path / "config.json"),
            "inherits_skip": ["input/skip_me"],
            "input/my_list_extends": [4],
        })
        assert config2.keys() == ["config_name", "input/my_list"]
        assert config2["input/my_list"] == [1, 2, 3, 4]
        assert "input/my_list_extends" not in config2

    def test_merge(self) -> None:
        config_base = Config({
            "input/old_list": [1, 2],
        })
        config_extend = Config({
            "input/new_list": [10, 20],
            "input/old_list_extends": [3],
        })

        config_merged = config_base.merge(config_extend)
        assert config_merged.keys() == ["input/new_list", "input/old_list"]
        assert config_merged["input/old_list"] == [1, 2, 3]
        assert config_merged["input/new_list"] == [10, 20]

        assert config_base.keys() == ["input/old_list"]
        assert config_base["input/old_list"] == [1, 2]

    def test_multiple_inheritance(self, tmp_path: Path) -> None:
        config1 = Config({
            "input/old_list": [1, 2],
        })
        config2 = Config({
            "input/new_list": [10, 20],
            "input/old_list_extends": [3],
        })

        config1.save_config(tmp_path / "config1.json")
        config2.save_config(tmp_path / "config2.json")

        config3 = Config({
            "inherits": [str(tmp_path / "config1.json"), str(tmp_path / "config2.json")],
            "input/old_list_extends": [4],
        })

        assert config3.keys() == ["config_name", "input/new_list", "input/old_list"]
        assert config3["input/old_list"] == [1, 2, 3, 4]
        assert config3["input/new_list"] == [10, 20]
