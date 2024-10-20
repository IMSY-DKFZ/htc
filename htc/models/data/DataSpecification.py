# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

import pandas as pd
from typing_extensions import Self

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.type_from_string import type_from_string


class DataSpecification:
    def __init__(self, path_or_file: str | Path | IO):
        """
        Reads a data specification file and creates data paths from the image identifiers.

        >>> specs = DataSpecification("pigs_semantic-only_5foldsV2.json")
        >>> path = specs.folds["fold_P041,P060,P069"]["train_semantic"][0]  # Select the first path of the training set
        >>> path.image_name()
        'P044#2020_02_01_09_51_15'

        You can also easily construct datasets from the folds:
        >>> from htc.models.image.DatasetImage import DatasetImage
        >>> first_fold = specs.folds["fold_P041,P060,P069"]
        >>> dataset_train = DatasetImage(first_fold["train_semantic"], train=True)
        >>> dataset_train[0]["image_name"]
        'P044#2020_02_01_09_51_15'
        >>> dataset_val_unknown = DatasetImage(first_fold["val_semantic_unknown"], train=False)
        >>> dataset_val_unknown[0]["image_name"]
        'P041#2019_12_14_12_00_16'

        Or get the paths from the different splits
        >>> validation_paths = specs.paths("^val")

        If you want to read the test paths, you have to explicitly enable it:
        >>> with specs.activated_test_set():
        ...     test_paths = specs.paths("^test")
        >>> len(validation_paths)
        340
        >>> len(test_paths)
        166
        >>> len({p.subject_name for p in validation_paths})
        15
        >>> len({p.subject_name for p in test_paths})
        5

        The main purpose of the data specification is to define your training setup, i.e. which paths should be used for training, which for validation etc. For this, add the name or (relative) path to your data specification in your config (`input/data_spec`) and it will iterate over your folds, use all paths for training which are part of a split starting with the name `train` for training, all with `val` for validation and  `test` for testing. For testing, however, you need to add the `--test` argument to the training script if the test images should be evaluated.

        The format of the data specification files itself is described in the data_spec.schema file.

        Args:
            path_or_file: Path (or string) to the data specification json file (path can also be relative to the models or data directories) or a file object which implements a read() method.
        """
        if isinstance(path_or_file, str):
            path_or_file = Path(path_or_file)

        if hasattr(path_or_file, "read"):
            specs = json.load(path_or_file)
            self.path = None
        else:
            # Add .json extension if the user forgot it
            if not path_or_file.name.endswith(".json"):
                path_or_file = path_or_file.with_name(path_or_file.name + ".json")

            possible_paths = [
                path_or_file,
                settings.htc_package_dir / path_or_file,
                settings.htc_projects_dir / path_or_file,
                settings.src_dir / path_or_file,
            ]

            # Check all data and model directories in this repo per default
            for data_dir in sorted(settings.htc_package_dir.rglob("data")) + sorted(
                settings.htc_projects_dir.rglob("data")
            ):
                models_dir = data_dir.parent
                possible_paths.append(data_dir / path_or_file)
                possible_paths.append(models_dir / path_or_file)

            self.path = None
            for path in possible_paths:
                if path.exists():
                    self.path = path
                    break

            if self.path is None:
                locations = "\n".join([str(p) for p in possible_paths])
                raise FileNotFoundError(
                    f"Cannot find the file {path_or_file}. Tried at the following locations:\n{locations}"
                )

            with self.path.open() as f:
                specs = json.load(f)

        assert type(specs) == list, "The data specification must be a list of folds"

        # Construct data paths from the data specification
        self.folds = {}
        self.__folds_test = {}  # We load the test path but keep it a secret until the user explicitly calls activate_test_set()
        for fold in specs:
            assert type(fold) == dict, "Each fold must be specified as a dict"

            fold_data = {}
            fold_data_test = {}
            for split_key, split_specs in fold.items():  # fold_name, train_semantic, val_semantic_unknown, ...
                if split_key == "fold_name":
                    continue

                if "data_path_class" in split_specs:
                    DataPathClass = type_from_string(split_specs["data_path_class"])
                else:
                    DataPathClass = DataPath

                paths = []
                for image_name in split_specs["image_names"]:
                    paths.append(DataPathClass.from_image_name(image_name))

                if split_key.startswith("test"):
                    fold_data_test[split_key] = paths
                else:
                    fold_data[split_key] = paths

            self.folds[fold["fold_name"]] = fold_data
            self.__folds_test[fold["fold_name"]] = fold_data_test

        # Make sure the folds are consistent
        split_names = self.split_names()
        for fold_data in self.folds.values():
            assert split_names == list(fold_data.keys()), "Every fold must use the same splits"

    def __repr__(self) -> str:
        return f"DataSpecification(name={self.name()}, folds={self.fold_names()}, splits={self.split_names()})"

    def __eq__(self, other: Self) -> bool:
        return self.folds == other.folds and self.__folds_test == other.__folds_test

    def __len__(self) -> int:
        """
        Returns: The number of folds in the data specification file.
        """
        return len(self.folds)

    def __iter__(self) -> Iterator[tuple[str, dict[str, DataPath]]]:
        """
        Iterates over all folds in the data specification file.

        >>> data_specs = DataSpecification("pigs_semantic-only_5foldsV2.json")
        >>> for fold_name, splits in data_specs:
        ...     print(fold_name)
        ...     for name, paths in splits.items():
        ...         print(name)
        ...         print(paths[0].image_name())
        ...         break
        ...     break
        fold_P041,P060,P069
        train_semantic
        P044#2020_02_01_09_51_15

        Yields: Pair of (fold_name, split_name) pairs
        """
        yield from self.folds.items()

    def name(self) -> str:
        """
        Returns: Name of the data specification file (without the file ending).
        """
        return self.path.stem

    def fold_names(self) -> list[str]:
        """
        Returns: List of the fold names in the data specification.
        """
        return list(self.folds.keys())

    def split_names(self) -> list[str]:
        """
        Returns: List of split names from all folds (it is ensured that every fold uses the same names).
        """
        return list(next(iter(self.folds.values())).keys())

    def paths(self, split_name: str = None) -> list[DataPath]:
        """
        Find all paths in this data specification file.

        Args:
            split_name: Optional regex selector to constrain the splits to be searched. If None, paths form all splits are used.

        Returns: Unique set of paths from all folds and all selected splits.
        """
        all_paths = set()

        for fold_data in self.folds.values():
            for name, paths_split in fold_data.items():
                if split_name is not None and re.search(split_name, name) is None:
                    continue

                all_paths.update(paths_split)

        return sorted(all_paths)

    def fold_paths(self, fold_name: str, split_name: str = None) -> list[DataPath]:
        """
        Find all paths for a given fold (similar to paths()).

        Args:
            fold_name: The name of the fold to look for paths.
            split_name: Optional regex selector to constrain the splits to be searched. If None, paths form all splits are used.

        Returns: Unique set of paths from the given fold and all selected splits.
        """
        assert fold_name in self.folds, f"Cannot find the fold name {fold_name}"

        fold_paths = set()
        for name, paths_split in self.folds[fold_name].items():
            if split_name is not None and re.search(split_name, name) is None:
                continue

            fold_paths.update(paths_split)

        return sorted(fold_paths)

    def table(self) -> pd.DataFrame:
        """
        Returns: The information of the data specification as a table.
        """
        rows = []
        for fold_name, splits in self:
            for name, paths in splits.items():
                for p in paths:
                    row = {
                        "fold_name": fold_name,
                        "split_name": name,
                        "image_name": p.image_name(),
                    }
                    row |= p.image_name_typed()
                    rows.append(row)

        return pd.DataFrame(rows)

    def activate_test_set(self) -> None:
        """
        This function activates all the data paths from the test set in this data specification file (they are hidden by default).

        After you call this function, you can use any of the other functions (e.g. paths()) and they will now include the test paths.
        """
        for fold_name in self.__folds_test.keys():
            self.folds[fold_name] |= self.__folds_test[fold_name]

    def deactivate_test_set(self) -> None:
        """
        Hides the test set again if it was activated before.

        Instead of using this function directly, please consider using activated_test_set() as contextmanager.
        """
        for fold_name in self.__folds_test.keys():
            for name in self.__folds_test[fold_name].keys():
                self.folds[fold_name].pop(name, None)

    @contextmanager
    def activated_test_set(self) -> Iterator[None]:
        """
        Contextmanager to temporarily access the test set.

        >>> data_specs = DataSpecification("pigs_semantic-only_5foldsV2.json")
        >>> with data_specs.activated_test_set():
        ...     print(len(data_specs.paths("^test")))
        166
        >>> print(len(data_specs.paths("^test")))
        0
        """
        self.activate_test_set()
        yield None
        self.deactivate_test_set()

    def to_json(self) -> str:
        """
        Returns: String representing the path to the data specification file.
        """
        if str(self.path).startswith(str(settings.htc_package_dir)):
            return self.path.name
        else:
            return str(self.path)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        assert "input/data_spec" in config, "There is no data specification defined in the config"

        if isinstance(config["input/data_spec"], DataSpecification):
            return config["input/data_spec"]
        else:
            try:
                spec = cls(config["input/data_spec"])
            except FileNotFoundError as e:
                data_path = config.path_config.parent / "data.json"
                if data_path.exists():
                    spec = cls(data_path)
                else:
                    raise FileNotFoundError(f"Could not find the data specification file {data_path}") from e

            config["input/data_spec"] = spec  # Cache for future use
            return spec
