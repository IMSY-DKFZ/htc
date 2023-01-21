# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import functools
import json
import pprint
import re
from collections.abc import Iterator
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable, Union

import commentjson

from htc.settings import settings
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.general import merge_dicts_deep


# Decorator to count key usage (useful to determine which keys have not been used during program execution)
def track_key_usage(method: Callable) -> Callable:
    @functools.wraps(method)
    def _track_key_usage(self, identifier: str, *args, **kwargs):
        if self._used_keys is not None:
            self._used_keys[identifier] = 1  # We re-use a dict, but we are only interested in the unique set of keys
        return method(self, identifier, *args, **kwargs)

    return _track_key_usage


class Config:
    def __init__(self, path_or_dict: Union[str, Path, dict], use_shared_dict=False):
        """
        This class can be used to work with the configuration files. It can be used as a dict but has some conveniences like accessing nested dicts via 'key1/key2' identifier. For the semantic meaning of common attributes, please take a look at the config.schema file.

        This class uses the commentjson library (https://github.com/vaidik/commentjson) to load the json configuration file. This means that comments are supported in the configuration file. However, they are lost when the configuration file is saved to a file.

        >>> config = Config(settings.models_dir / 'image' / 'configs' / 'default.json')
        >>> config['input/n_channels']
        100

        Adding the config file extension is optional:

        >>> config = Config(settings.models_dir / 'image' / 'configs' / 'default')  # Defaults to .json
        >>> config['input/n_channels']
        100

        Config files can also inherit from each other (e.g. usually the rgb and param configs inherit from the default HSI config). For example, the rgb config of the image model inherits from the default config (which is used by the HSI modality).

        >>> config_rgb = Config(settings.models_dir / 'image' / 'configs' / 'default_rgb.json')

        config_rgb is the same as config expect for the changes made by the rgb config:

        >>> config['input/preprocessing'] = None  # We are making the default config identical to the rgb config just for demonstration purposes
        >>> config['input/n_channels'] = 3        # These settings are changed by the rgb config
        >>> config['config_name'] = 'default_rgb'
        >>> config == config_rgb  # The rgb config is now identical to the default config from which it inherits
        True

        Config objects can also be constructed from existing dictionaries:

        >>> Config({'a/b': 1, 'c': 3})
        {'a': {'b': 1}, 'c': 3}
        >>> Config({'a/b/c': 'str_value1', 'd': {'x': 'str_value2'}})
        {'a': {'b': {'c': 'str_value1'}}, 'd': {'x': 'str_value2'}}

        Note: If you want to make a copy of the config to change values (without affecting the existing config), you can use copy.copy() which gives you a copy of all keys pointing to builtins (e.g. strings) but only copies the reference to objects like a data specification file (which is much better for performance). You should only deepcopy a config file if you want to make changes to e.g. the references data specs.

        Args:
            path_or_dict: Path (or string) to the configuration file to load or a dictionary with the data.
            use_shared_dict: If True, then a shared dictionary is used to track key usage (via multiprocessing.Manager). This is necessary to get correct usage statistics when multiprocessing is used. However, the manager object creates a subprocess and this may lead to problems if the config is created inside a processing pool as daemonic processes are not allowed to have children. The default is to use a standard Python dictionary. Note: if correct usage statistics are needed inside a processing pool, then the ProcessPoolExecutor might be an option (https://stackoverflow.com/a/61470465/2762258).
        """
        if isinstance(path_or_dict, str):
            path_or_dict = Path(path_or_dict)

        self.path_config = None
        self._used_keys = None

        if isinstance(path_or_dict, Path):
            # Load from file
            self.path_config = path_or_dict

            # Add .json extension if the user forgot it
            if not self.path_config.exists():
                with_extension = self.path_config.with_name(self.path_config.name + ".json")
                if with_extension.exists():
                    self.path_config = with_extension
                else:
                    raise ValueError(
                        f"Cannot find the config file at {self.path_config}. Please make sure that the file exists at"
                        " this location"
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
            match = re.search(r"\.[^.]+$", self["inherits"])
            extension = ".json" if match is None else ""
            inherits = Path(self["inherits"] + extension)

            # We try several locations to find the parent config file
            possible_paths = [
                inherits,  # Relative/absolute
                settings.models_dir
                / inherits,  # Relative to the model's dir (name of the model must be provided, though, e.g. image/configs/default)
                settings.htc_package_dir / inherits,  # Relative to the htc package directory
                settings.src_dir / inherits,  # Relative to the src directory
            ]
            if self.path_config is not None:
                possible_paths.append(self.path_config.with_name(inherits.name))  # Same directory as the child config

            parent_path = None
            for path in possible_paths:
                if path.exists():
                    parent_path = path
                    break

            assert parent_path is not None, (
                f"Cannot find the path to the parent configuration file {inherits}. Tried at the following locations:"
                f" {possible_paths}"
            )
            data_parent = Config(parent_path).data

            # The existing data (=data from the child) has precedence over the parent data
            self.data = dict(merge_dicts_deep(data_parent, self.data))

            del self["inherits"]

        # We start counting usage from here on
        if use_shared_dict:
            # The key tracking should also work in multiprocessing environments (e.g. network training)
            self._used_keys = Manager().dict()
        else:
            self._used_keys = {}

    def __copy__(self):
        # Making a deepcopy of the config may be costly e.g. with large data specs
        # Here we are only (deep)copying builtins but keep references to objects (like a data specs)
        # Can be used via copy.copy(config)
        cls = self.__class__
        config_new = cls.__new__(cls)

        def copy_data(dict_data: dict) -> dict:
            new_data = {}

            for key, value in dict_data.items():
                if type(value) == dict:
                    # Due to the recursive call of this function, we are basically deep-copying dicts
                    new_data[key] = copy_data(value)
                elif type(value) == list:
                    # Similarly, we are also deep-copying lists
                    new_data[key] = copy.deepcopy(value)
                else:
                    # Everything else will just be reference-copied (e.g. data specs)
                    new_data[key] = value

            return new_data

        config_new.data = copy_data(self.data)
        config_new.path_config = self.path_config
        config_new._used_keys = self._used_keys

        return config_new

    def used_keys(self) -> list[str]:
        """Returns: List of all keys which have been accessed from this config (either via getter, setter, contains check or deletion). Nested keys are not in the list if only the top-level key has been used (e.g. 'a/b' and 'a/c are not in the list if only 'a' is accessed).
        """
        return sorted(self._used_keys.keys())

    def unused_keys(self) -> list[str]:
        """Returns: List of keys from this config which have never been used (either via getter, setter, contains or deletion).
        """
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

        >>> config = Config.load_config(model_name='pixel')
        >>> 'model/activation_function' in config
        True
        >>> 'model' in config
        True
        >>> 'lr' in config
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

        >>> config = Config.load_config(model_name='image')
        >>> config.get('non_existing_key',1.0)
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

        >>> config = Config.load_config(model_name='pixel')
        >>> config['model/lr'] = 0.1
        >>> config['model/lr']
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

        >>> config = Config.load_config(model_name='pixel')
        >>> del config['dataloader_kwargs/batch_size']
        >>> config.keys()[:2]
        ['config_name', 'dataloader_kwargs/num_workers']
        >>> config = Config({'test': 1})
        >>> del config['test']
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

        >>> config = Config.load_config(model_name='pixel')
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

        return zip(keys, values)

    def save_config(self, path: Path) -> None:
        with path.open("w") as fp:
            commentjson.dump(self.data, fp, indent=4, sort_keys=True, cls=AdvancedJSONEncoder)

        self.path_config = path

    @classmethod
    def load_config(cls, path: Union[Path, str] = None, model_name: str = None, **config_kwargs) -> "Config":
        """
        Tries to find the configuration file in several common locations:

            * Relative or absolute path.
            * Relative to settings.models_dir.
            * Relative to settings.htc_package_dir.
            * Relative to the model's config folder (inside settings.models_dir).

        >>> config = Config.load_config("default", "image")
        >>> config["input/n_channels"]
        100

        Args:
            path: Name or absolute/relative path to the configuration file. If None, loads the default configuration file.
            model_name: Name of the model inside settings.models_dir (e.g. pixel). If not None, will be used as additional search path.
            config_kwargs: Additional keyword arguments passed on to the Config constructor.

        Returns: The configuration object.
        """
        if path is None:
            path = "default.json"

        path_config = Path(path)
        if path_config.suffix == "":
            path_config = path_config.with_suffix(".json")

        possible_paths = [
            path_config,
            settings.models_dir / path_config,
            settings.htc_package_dir / path_config,
        ]

        if model_name is not None:
            possible_paths.append(settings.models_dir / model_name / "configs" / path_config)

        config = None
        for p in possible_paths:
            if p.exists():
                config = Config(p, **config_kwargs)
                break

        if config is None:
            raise ValueError(
                f"Cannot find the configuration file {path}. Tried the following locations: {possible_paths}"
            )

        return config

    @classmethod
    def load_config_fold(cls, experiment_folder: Path) -> "Config":
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
