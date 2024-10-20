# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import functools
import json
import pprint
from collections.abc import Callable, Iterator
from multiprocessing import Manager
from pathlib import Path
from typing import Any

import commentjson
from typing_extensions import Self

from htc.settings import settings
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.general import merge_dicts_deep
from htc.utils.unify_path import unify_path


# Decorator to count key usage (useful to determine which keys have not been used during program execution)
def track_key_usage(method: Callable) -> Callable:
    @functools.wraps(method)
    def _track_key_usage(self, identifier: str, *args, **kwargs):
        if self._used_keys is not None:
            self._used_keys[identifier] = 1  # We re-use a dict, but we are only interested in the unique set of keys
        return method(self, identifier, *args, **kwargs)

    return _track_key_usage


class Config:
    def __init__(self, path_or_dict: str | Path | dict, use_shared_dict=False):
        """
        This class can be used to work with the configuration files. It can be used as a dict but has some conveniences like accessing nested dicts via 'key1/key2' identifier. For the semantic meaning of common attributes, please take a look at the config.schema file.

        This class uses the commentjson library (https://github.com/vaidik/commentjson) to load the json configuration file. This means that comments are supported in the configuration file. However, they are lost when the configuration file is saved to a file.

        >>> config = Config(settings.models_dir / "image" / "configs" / "default.json")
        >>> config["input/n_channels"]
        100

        Adding the config file extension is optional:

        >>> config = Config(settings.models_dir / "image" / "configs" / "default")  # Defaults to .json
        >>> config["input/n_channels"]
        100

        Config files can also inherit from each other (e.g. usually the rgb and param configs inherit from the default HSI config). For example, the rgb config of the image model inherits from the default config (which is used by the HSI modality).

        >>> config_rgb = Config(settings.models_dir / "image" / "configs" / "default_rgb.json")

        config_rgb is the same as config expect for the changes made by the rgb config:

        >>> config["input/preprocessing"] = (
        ...     None  # We are making the default config identical to the rgb config just for demonstration purposes
        ... )
        >>> config["input/n_channels"] = 3  # These settings are changed by the rgb config
        >>> config["config_name"] = "default_rgb"
        >>> config == config_rgb  # The rgb config is now identical to the default config from which it inherits
        True

        For list config values inheritance is difficult as usually the user wants to append to the list. To achieve this, you can add a config value with the same key as the original list but with `_extends` appended. Those values will then be merged:
        >>> config = Config({
        ...     "input/my_list": [1, 2],
        ...     "input/my_list_extends": [3],
        ... })
        >>> config["input/my_list"]
        [1, 2, 3]

        Config objects can also be constructed from existing dictionaries:

        >>> Config({"a/b": 1, "c": 3})
        {'a': {'b': 1}, 'c': 3}
        >>> Config({"a/b/c": "str_value1", "d": {"x": "str_value2"}})
        {'a': {'b': {'c': 'str_value1'}}, 'd': {'x': 'str_value2'}}

        Note: If you want to make a copy of the config to change values (without affecting the existing config), you can use copy.copy() which gives you a copy of all keys pointing to builtins (e.g. strings) but only copies the reference to objects like a data specification file (which is much better for performance). You should only deepcopy a config file if you want to make changes to e.g. the references data specs.

        Args:
            path_or_dict: Path (or string) to the configuration file to load or a dictionary with the data. If a path, several common locations are searched:
            * absolute/relative
            * relative to the model's directory (e.g. image/configs/default.json)
            * relative to the htc package directory (e.g. models/image/configs/default.json)
            * relative to the htc projects directory (e.g. species/configs/default.json)
            * relative to the repository root (e.g. htc/models/image/configs/default.json)

            use_shared_dict: If True, then a shared dictionary is used to track key usage (via multiprocessing.Manager). This is necessary to get correct usage statistics when multiprocessing is used. However, the manager object creates a subprocess and this may lead to problems if the config is created inside a processing pool as daemonic processes are not allowed to have children. The default is to use a standard Python dictionary. Note: if correct usage statistics are needed inside a processing pool, then the ProcessPoolExecutor might be an option (https://stackoverflow.com/a/61470465/2762258).
        """
        if isinstance(path_or_dict, str):
            path_or_dict = Path(path_or_dict)

        self.path_config = None
        self._used_keys = None

        if isinstance(path_or_dict, Path):
            # Add .json extension if the user forgot it
            if not path_or_dict.name.endswith(".json"):
                path_or_dict = path_or_dict.with_name(path_or_dict.name + ".json")

            # Find the location to the config file
            for p in Config.get_possible_paths(path_or_dict):
                if p.exists():
                    self.path_config = unify_path(p)
                    break
            assert self.path_config is not None, (
                f"Cannot find the config file at {self.path_config}. Please make sure that the file exists at this"
                " location"
            )

            with self.path_config.open() as fp:
                self.data = commentjson.load(fp)

            self["config_name"] = self.path_config.stem
        else:
            self.data = {}
            for k, v in path_or_dict.items():
                if type(k) == str and "/" in k:
                    # a/b style
                    self[k] = v
                else:
                    # Standard dict
                    self.data[k] = v

        if self["inherits"]:
            if type(self["inherits"]) == str:
                self["inherits"] = [self["inherits"]]

            for parent in self["inherits"]:
                extension = "" if parent.endswith(".json") else ".json"
                inherits = Path(parent + extension)

                # We try several locations to find the parent config file
                possible_paths = Config.get_possible_paths(inherits)
                if self.path_config is not None:
                    possible_paths.append(
                        self.path_config.with_name(inherits.name)
                    )  # Same directory as the child config

                parent_path = None
                for path in possible_paths:
                    if path.exists():
                        parent_path = path
                        break

                assert parent_path is not None, (
                    f"Cannot find the path to the parent configuration file {inherits}. Tried at the following"
                    f" locations: {possible_paths}"
                )

                config_parent = Config(parent_path)
                if self["inherits_skip"]:
                    for key in self["inherits_skip"]:
                        del config_parent[key]
                data_parent = config_parent.data

                # The existing data (=data from the child) has precedence over the parent data
                self.data = dict(merge_dicts_deep(data_parent, self.data))

                # Extend all config keys from the parent (but not the own class due to the possibility of multiple inherence)
                self._extend_lists(config_parent)

            del self["inherits"]
            del self["inherits_skip"]

        self._extend_lists()

        # We start counting usage from here on
        if use_shared_dict:
            # The key tracking should also work in multiprocessing environments (e.g. network training)
            self._used_keys = Manager().dict()
        else:
            self._used_keys = {}

    def _extend_lists(self, base_config: "Config" = None) -> None:
        if base_config is None:
            base_config = self

        # Users can extend additional lists by adding the same key with _extends appended
        for k, v in base_config.items():
            if k.endswith("_extends") and type(v) == list:
                k_original = k.removesuffix("_extends")
                if k_original in self:
                    assert type(self[k_original]) == list, (
                        f"The original key {k_original} is not a list but a {type(self[k_original])} which is not"
                        " supported for the extends feature"
                    )
                    self[k_original] = self[k_original] + v
                    del base_config[k]

    def _copy_data(self, dict_data: dict) -> dict:
        new_data = {}

        for key, value in dict_data.items():
            if type(value) == dict:
                # Due to the recursive call of this function, we are basically deep-copying dicts
                new_data[key] = self._copy_data(value)
            elif type(value) == list:
                # Similarly, we are also deep-copying lists
                new_data[key] = copy.deepcopy(value)
            else:
                # Everything else will just be reference-copied (e.g. data specs)
                new_data[key] = value

        return new_data

    def __copy__(self) -> Self:
        # Making a deepcopy of the config may be costly e.g. with large data specs
        # Here we are only (deep)copying builtins but keep references to objects (like a data specs)
        # Can be used via copy.copy(config)
        cls = self.__class__
        config_new = cls.__new__(cls)

        config_new.data = self._copy_data(self.data)
        config_new.path_config = self.path_config
        config_new._used_keys = self._used_keys

        return config_new

    def merge(self, other: Self) -> Self:
        """
        Merge the current config with another config or a dictionary of properties. The other config has precedence over the current config.

        >>> config = Config({"a/b": 1, "c": 3})
        >>> config_merged = config.merge(Config({"a/b": 2, "d": 4}))
        >>> config_merged["a/b"]
        2
        >>> config_merged["c"]
        3
        >>> config_merged["d"]
        4

        Args:
            other: The config to merge the current config with.

        Returns: The merged config (the old config remains untouched).
        """
        config_merge = copy.copy(self)
        config_merge.data = dict(merge_dicts_deep(config_merge.data, other.data))
        config_merge._extend_lists()

        return config_merge

    def used_keys(self) -> list[str]:
        """Returns: List of all keys which have been accessed from this config (either via getter, setter, contains check or deletion). Nested keys are not in the list if only the top-level key has been used (e.g. 'a/b' and 'a/c are not in the list if only 'a' is accessed)."""
        return sorted(self._used_keys.keys())

    def unused_keys(self) -> list[str]:
        """Returns: List of keys from this config which have never been used (either via getter, setter, contains or deletion)."""
        # First determine all used keys including nested keys
        all_used_keys = set(self._used_keys.keys())
        for used_key in self._used_keys:
            for k in self.keys():
                if k.startswith(f"{used_key}/"):
                    all_used_keys.add(k)

        return sorted(set(self.keys()) - all_used_keys)

    def __repr__(self) -> str:
        # JSON conversion ensures that any object references get resolved
        clean_data = json.loads(json.dumps(self.data, indent=4, sort_keys=True, cls=AdvancedJSONEncoder))
        return pprint.pformat(clean_data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self.data == other.data
        else:
            return False

    @track_key_usage
    def __contains__(self, identifier: str) -> bool:
        """
        Checks if the configuration has a certain key.

        >>> config = Config.from_model_name(model_name="pixel")
        >>> "model/activation_function" in config
        True
        >>> "model" in config
        True
        >>> "lr" in config
        False
        """
        keys = identifier.split("/")

        last_data = self.data
        for key in keys:
            if type(last_data) != dict or key not in last_data:
                return False
            else:
                last_data = last_data[key]

        return True

    @track_key_usage
    def __getitem__(self, identifier: str):
        """Returns the configuration value for a key."""
        keys = identifier.split("/")

        last_data = self.data
        for key in keys:
            if type(last_data) != dict:
                return None
            if key not in last_data:
                return None

            last_data = last_data[key]

        return last_data

    def get(self, identifier: str, default=None):
        """
        Same as __getitem__ but allowing a default value in case the identifier does not exist.

        >>> config = Config.from_model_name(model_name="image")
        >>> config.get("non_existing_key", 1.0)
        1.0

        Args:
            identifier: Key to search for.
            default: Default value in case the key does not exist.

        Returns: Value of the config variable or the default.
        """
        if identifier in self:
            return self[identifier]
        else:
            return default

    @track_key_usage
    def __setitem__(self, identifier: str, value) -> None:
        """
        Change the value of a configuration.

        >>> config = Config.from_model_name(model_name="pixel")
        >>> config["model/lr"] = 0.1
        >>> config["model/lr"]
        0.1
        """
        keys = identifier.split("/")

        last_data = self.data
        for i, key in enumerate(keys):
            if key not in last_data:
                last_data[key] = {}

            if i == len(keys) - 1:
                last_data[key] = value
            else:
                last_data = last_data[key]

    @track_key_usage
    def __delitem__(self, identifier: str) -> None:
        """
        Removes an identifier from the configuration.

        >>> config = Config.from_model_name(model_name="pixel")
        >>> del config["dataloader_kwargs/batch_size"]
        >>> config.keys()[:2]
        ['config_name', 'dataloader_kwargs/num_workers']
        >>> config = Config({"test": 1})
        >>> del config["test"]
        >>> config.keys()
        []

        Args:
            identifier (str): [description]
        """
        if identifier in self:
            keys = identifier.split("/")
            last_data = self.data
            for key in keys[:-1]:
                last_data = last_data[key]

            del last_data[keys[-1]]

    def keys(self) -> list[str]:
        """
        List of all (nested) keys in the configuration.

        >>> config = Config.from_model_name(model_name="pixel")
        >>> config.keys()[:2]
        ['config_name', 'dataloader_kwargs/batch_size']

        Returns: List of keys.
        """

        def get_keys(dict_data, prefix=""):
            keys = []

            for key, value in dict_data.items():
                if type(value) == dict:
                    inner_keys = get_keys(value, prefix=f"{prefix}{key}/")
                    keys += inner_keys
                else:
                    keys.append(prefix + key)

            return keys

        return sorted(get_keys(self.data))

    def items(self) -> Iterator[tuple[str, Any]]:
        """List of all (nested) (key,item) pairs in the configuration."""
        keys = self.keys()
        values = [self[k] for k in keys]

        return zip(keys, values, strict=True)

    def save_config(self, path: Path) -> None:
        with path.open("w") as fp:
            commentjson.dump(self.data, fp, indent=4, sort_keys=True, cls=AdvancedJSONEncoder)

        self.path_config = path
        self["config_name"] = self.path_config.stem

    @classmethod
    def from_model_name(cls, config_name: Path | str = None, model_name: str = None, **config_kwargs) -> Self:
        """
        Load the configuration file for a model. Several common locations are searched:

            * Relative or absolute path.
            * Relative to settings.models_dir.
            * Relative to settings.htc_package_dir.
            * Relative to settings.htc_projects_dir.
            * Relative to the model's config folder (inside settings.models_dir).

        >>> config = Config.from_model_name("default", "image")
        >>> config["input/n_channels"]
        100

        Args:
            config_name: Name or absolute/relative path to the configuration file. If None, loads the default configuration file.
            model_name: Name of the model inside settings.models_dir (e.g. pixel). If not None, will be used as additional search path.
            config_kwargs: Additional keyword arguments passed on to the Config constructor.

        Returns: The configuration object.
        """
        if config_name is None:
            config_name = "default.json"

        path_config = Path(config_name)
        if not path_config.name.endswith(".json"):
            path_config = path_config.with_name(path_config.name + ".json")

        possible_paths = Config.get_possible_paths(path_config)
        if model_name is not None:
            possible_paths.append(settings.models_dir / model_name / "configs" / path_config)

        config = None
        for p in possible_paths:
            if p.exists():
                config = Config(p, **config_kwargs)
                break

        if config is None:
            locations = "\n".join([str(p) for p in possible_paths])
            raise ValueError(
                f"Cannot find the configuration file {config_name}. Tried the following locations:\n{locations}"
            )

        return config

    @classmethod
    def from_fold(cls, experiment_folder: Path) -> Self:
        """
        Loads a config.json file from an experiment consisting of multiple config files (in corresponding subdirectories for each fold). The config files from all folds will be read and it is checked that all configs are identical and that a config file exists for each fold.

        Args:
            experiment_folder: Path to the experiment folder or path to a folder which contains a config.json file.

        Returns: The configuration object.
        """
        config = None

        for fold_dir in sorted(experiment_folder.glob("fold*")):
            config_file = fold_dir / "config.json"
            assert config_file.exists(), (
                f"There is no config file in the folder {fold_dir}. This usually happens when the corresponding run"
                " crashed"
            )

            loaded_config = cls(config_file)

            if config is None:
                config = loaded_config
            else:
                assert (
                    loaded_config == config
                ), f"The config file {config_file} does not match the previous config files"

        assert config is not None, f"Could not find a config file in the experiment folder {experiment_folder}"
        return config

    @staticmethod
    def get_possible_paths(path: Path) -> list[Path]:
        return [
            # Relative/absolute
            path,
            # Relative to the model's dir (name of the model must be provided, though, e.g. image/configs/default)
            settings.models_dir / path,
            # Relative to the htc package directory
            settings.htc_package_dir / path,
            # Relative to the htc_projects directory
            settings.htc_projects_dir / path,
            # Relative to the src directory
            settings.src_dir / path,
        ]
