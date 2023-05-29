# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Union

from htc.utils.unify_path import unify_path


class Datasets:
    def __init__(
        self,
        network_dir: Path = None,
        network_project_folder: str = "",
        network_data_folder: str = "",
    ):
        """
        This class is a helper to access datasets which are either stored locally or on a common network drive. If a dataset is not available locally, then it is automatically accessed on the network drive (which may be slow).

        >>> from htc.settings import settings
        >>> str(settings.data_dirs.semantic)  # doctest: +ELLIPSIS
        '...2021_02_05_Tivita_multiorgan_semantic/data'

        There are also special variables to access common directories on the network drive:
        >>> str(settings.datasets.network)  # doctest: +ELLIPSIS
        '.../E130-Projekte'
        >>> str(settings.datasets.network_project)  # doctest: +ELLIPSIS
        '.../E130-Projekte/Biophotonics/Projects/2021_02_05_hyperspectral_tissue_classification'
        >>> str(settings.datasets.network_data)  # doctest: +ELLIPSIS
        '.../E130-Projekte/Biophotonics/Data'

        Please do not access the parent of a data directory directly since data may be a symbolic link (e.g. due to filesystem size constraints). For example, if 2021_02_05_Tivita_multiorgan_semantic/data is a symbolic link pointing to /my_folder/data, then .parent.name is my_folder and not 2021_02_05_Tivita_multiorgan_semantic.

        >>> entry = settings.datasets.semantic
        >>> list(entry.keys())
        ['path_dataset', 'path_data', 'path_intermediates', 'location', 'has_unified_paths', 'shortcut', 'env_name']
        >>> str(entry["path_dataset"])  # doctest: +ELLIPSIS
        '...2021_02_05_Tivita_multiorgan_semantic'
        >>> str(entry["path_intermediates"])  # doctest: +ELLIPSIS
        '...2021_02_05_Tivita_multiorgan_semantic/intermediates'

        Args:
            network_dir: Path to the network directory. If None, accessing the network drive or searching for folders on the network drive will not work.
            network_project_folder: Folder on the network drive which contains the data for this project (e.g. results).
            network_data_folder: Folder on the network drive which contains the datasets.
        """
        self.network_dir = network_dir
        self.network_project_folder = network_project_folder
        self.network_data_folder = network_data_folder
        self.dataset_names = []
        self._dirs = {}

        # Avoid circular imports
        from htc.settings import settings

        self.log = settings.log
        self.log_once = settings.log_once

    def __repr__(self) -> str:
        dirs = []
        for name, entry in self:
            shortcut = (
                f"settings.data_dirs.{entry['shortcut']}" if entry["shortcut"] is not None else "(no shortcut set)"
            )
            dirs.append(f"""- {shortcut}
    * full name: {name}
    * environment name: {self.path_to_env(entry['path_data'])}
    * location: {entry['location']}""")

        msg = f"Network directory: {self.network if self.network is not None else '(not set)'}\n"
        msg += "Registered data directories:\n" + "\n".join(dirs)
        return msg

    def __iter__(self) -> Iterator[tuple[str, dict]]:
        """
        Iterate over all registered data directories. Only the dataset names as they are used on the network drive are returned. If you need the environment variable name, please use path_to_env() or entry["env_name"].

        >>> from htc.settings import settings
        >>> [(name, entry["env_name"]) for name, entry in settings.datasets]  # doctest: +ELLIPSIS
        [('2021_07_26_Tivita_multiorgan_human', 'PATH_Tivita_multiorgan_human'), ('2021_03_30_Tivita_studies', 'PATH_Tivita_studies'), ...]

        Yields: Tuple with name of the dataset folder and entry object.
        """
        for name in self.dataset_names:
            yield name, self._dirs[name]

    def add_dir(
        self,
        env_name: str,
        network_folder: str = None,
        data_folder: str = "data",
        intermediates_folder: str = "intermediates",
        shortcut: str = None,
        additional_names: list[str] = None,
    ) -> None:
        """
        Adds a data directory to this class.

        Args:
            env_name: Name of the environment variable which may contain the path to this data directory.
            network_folder: Name of the folder on the network drive (e.g. name of the folder in Biophotonics/Data).
            data_folder: Name inside the data directory which contains the data (e.g. 'data' which is often accompanied by an 'intermediates' directory).
            intermediates_folder: Name of the intermediates directory which contains preprocessed data.
            shortcut: Common shortcut name for the dataset.
            additional_names: List of additional names which can be used to access the dataset.
        """
        if env_name in self._dirs:
            self.log.warning(
                f"The environment variable {env_name} is already registered. Did you call this function twice? The"
                " dataset is still added (again)"
            )

        # User may disable an env variable via PATH_Tivita_multiorgan_semantic=''
        if path_env := os.getenv(env_name, False):
            # Here, we are only interested in the path, not the options
            path_env, _ = Datasets.parse_path_options(path_env)
            path_env = unify_path(path_env)
            path_env_data = path_env / data_folder

            if not path_env.exists():
                self.log.warning(
                    f"The environment variable {env_name} was set to {path_env} but the path does not exist"
                )
            else:
                if not path_env_data.exists():
                    self.log.warning(
                        f"The environment variable {env_name} was set to {path_env} and the path exists, but the data"
                        f" folder {data_folder} does not exist. Did you set a wrong data_folder_name?"
                    )

            path_entry = {
                "path_dataset": path_env,
                "path_data": path_env_data,
                "path_intermediates": path_env / intermediates_folder,
                "location": "local",
                "has_unified_paths": True,
                "shortcut": shortcut,
                "env_name": env_name,
            }
        elif self.network_dir is not None and network_folder is not None:
            path_dataset = self.network_data / network_folder
            path_entry = {
                "path_dataset": path_dataset,
                "path_data": path_dataset / data_folder,
                "path_intermediates": path_dataset / intermediates_folder,
                "location": "network",
                "has_unified_paths": False,
                "shortcut": shortcut,
                "env_name": env_name,
            }
        else:
            path_entry = None

        if path_entry is not None:
            if network_folder is None:
                dataset_name = path_entry["path_dataset"].name
            else:
                dataset_name = network_folder
            self.dataset_names.append(dataset_name)

            # Each directory should be at least accessible via its environment variable name (e.g. PATH_Tivita_multiorgan_semantic) or the name of the folder (e.g. 2021_02_05_Tivita_multiorgan_semantic)
            possible_names = [env_name, dataset_name]

            # Add additional names if not already there
            if additional_names is not None:
                for name in additional_names:
                    if name not in possible_names:
                        possible_names.append(name)

            for name in possible_names:
                self._dirs[name] = path_entry

                # Every uppercase version of a name should also match (--> Windows)
                name_upper = name.upper()
                if name != name_upper and name_upper not in self._dirs:
                    self._dirs[name_upper] = path_entry

    def get(self, item: str, local_only: bool = False) -> Union[dict, None]:
        """
        Access a data directory from this class.

        >>> from htc.settings import settings
        >>> str(settings.datasets.masks["path_dataset"])  # doctest: +ELLIPSIS
        '...2021_02_05_Tivita_multiorgan_masks'

        Args:
            item: Name of the environment variable, network folder name or shortcut name.
            local_only: If True, network locations will not be considered.

        Returns: Dictionary with information about the dataset (or None if the dataset could not be found):
            - path_dataset: Path to the dataset root folder, e.g. my/path/2021_02_05_Tivita_multiorgan_masks.
            - path_data: Path to the data folder, e.g. my/path/2021_02_05_Tivita_multiorgan_masks/data.
            - path_intermediates: Path to the intermediates folder, e.g. my/path/2021_02_05_Tivita_multiorgan_masks/intermediates.
            - location: Whether this path is available locally or only on the network drive.
            - shortcut: Short name for the dataset.
            - env_name: Name of the environment variable which which was used to add this dataset.
        """
        if "#" in item:
            # E.g. 2021_02_05_Tivita_multiorgan_semantic#context_experiments
            item, subdata = item.split("#")
        else:
            subdata = None

        matched_location = self._find_match(item)

        if matched_location is not None:
            if matched_location["location"] == "network":
                if local_only:
                    # In case we are not interested in network locations
                    return None
                else:
                    self.log_once.info(
                        f"The environment variable {item} is not set. Falling back to the network drive (this may be"
                        " slow)"
                    )

            if not matched_location["has_unified_paths"]:
                # We only want to unify paths once as this can be a costly operation
                matched_location["path_data"] = unify_path(matched_location["path_data"])
                matched_location["path_dataset"] = unify_path(matched_location["path_dataset"])
                matched_location["has_unified_paths"] = True

            if subdata is not None:
                match = copy.deepcopy(matched_location)
                match["path_data"] = match["path_data"] / subdata
                return match
            else:
                return matched_location
        else:
            return None

    def __getitem__(self, item: str) -> Union[dict, None]:
        return self.get(item)

    def __getattr__(self, item: str) -> Union[dict, Path, None]:
        if item.startswith("_"):
            # __getattr__ may be called for built-ins, in which we are not interested
            return super().__getattr__(item)
        elif item == "network":
            return unify_path(self.network_dir) if self.network_dir is not None else None
        elif item == "network_project":
            return unify_path(self.network_dir / self.network_project_folder) if self.network_dir is not None else None
        elif item == "network_data":
            return unify_path(self.network_dir / self.network_data_folder) if self.network_dir is not None else None
        else:
            return self.get(item)

    def __contains__(self, item: str) -> bool:
        """
        Checks whether a data directory can be found.

        Args:
            item: Name of the data directory (same as in get()).

        Returns: True if a data directory with this name exists.
        """
        return self._find_match(item) is not None

    def network_location(self, item: str, path_data: bool = True) -> Path:
        """
        Similar to get() but always returns the path on the network drive.

        >>> from htc.settings import settings
        >>> str(settings.datasets.network_location("semantic"))  # doctest: +ELLIPSIS
        '.../E130-Projekte/Biophotonics/Data/2021_02_05_Tivita_multiorgan_semantic/data'
        >>> str(settings.datasets.network_location("semantic", path_data=False))  # doctest: +ELLIPSIS
        '.../E130-Projekte/Biophotonics/Data/2021_02_05_Tivita_multiorgan_semantic'

        Args:
            item: Name to search for (similar to get()).
            path_data: If False, the path to the dataset is returned (i.e. the parent directory), else the path pointing to the data subfolder.

        Returns: Path on the network drive.
        """
        assert self.network_dir is not None, "No network directory set"

        entry = self[item]
        return (
            self.network_data / entry["path_dataset"].name / entry["path_data"].name
            if path_data
            else self.network_data / entry["path_dataset"].name
        )

    def path_to_env(self, path: Union[str, Path]) -> Union[str, None]:
        """
        Searches for the name of the environment variable which corresponds to the given path.

        >>> from htc.settings import settings
        >>> path = settings.data_dirs['2021_02_05_Tivita_multiorgan_masks']
        >>> settings.datasets.path_to_env(path)
        'PATH_Tivita_multiorgan_masks'

        Args:
            path: Path to one of the data directories.

        Returns: The name of the corresponding environment variable or None if there is no match for the path.
        """
        for _, entry in self:
            if unify_path(path) == unify_path(entry["path_data"]):
                return entry["env_name"]

        return None

    def env_keys(self) -> Iterator[str]:
        """
        Iterates over all data directory names. Only names starting with PATH will be returned. For example, only PATH_Tivita_multiorgan_semantic but not 2021_02_05_Tivita_multiorgan_semantic will be returned.

        >>> data_dirs = Datasets()
        >>> data_dirs.add_dir("PATH_Tivita_multiorgan_semantic", "2021_02_05_Tivita_multiorgan_semantic")
        >>> list(data_dirs.env_keys())
        ['PATH_Tivita_multiorgan_semantic']
        """
        for _, entry in self:
            yield entry["env_name"]

    def find_intermediates_dir(self, path: Union[str, Path], intermediates_folder: str = "intermediates") -> Path:
        """
        Searches for the intermediates directory given the path to a dataset or data folder by iterating over all known entry of this class.

        Note: If you need the intermediates directory based on the name of the dataset, use `settings.intermediates_dirs.<dataset_name>`.
        Note: Similar to `find_entry()` but expects only paths to the data or intermediates directories and there is no guarantee that the found path exists.

        Args:
            path: Path to the dataset or the data folder.
            intermediates_folder: Name of the intermediates folder.

        Returns: Path (or string) to the intermediates directory. There is no guarantee that it exists.
        """
        if isinstance(path, str):
            path = Path(path)

        if (entry := self.find_entry(path)) is not None:
            return entry["path_intermediates"]

        if "data" in path.parts:
            data_index = path.parts.index("data")
            return Path(*path.parts[:data_index]) / intermediates_folder

        # Last resort, relative to the parent
        return path.parent / intermediates_folder

    def find_entry(self, path: Union[str, Path]) -> Union[dict, None]:
        """
        Searches for a known entry given an arbitrary path (e.g. to an image folder). An entry matches if the given path is part of one of the known dataset paths.

        Args:
            path: Path to match against known entries.

        Returns: Dataset entry object similar to `get()` or None if no entry matched.
        """
        if isinstance(path, Path):
            path = str(path)

        for _, entry in self:
            if str(entry["path_dataset"]) in path:
                return entry

        return None

    def _find_match(self, item: str) -> Union[dict, None]:
        matched_location = None

        if item in self._dirs:
            matched_location = self._dirs[item]
        else:
            # Try shortcut name next
            for _, entry in self:
                if entry["shortcut"] is not None and item == entry["shortcut"]:
                    matched_location = entry
                    break

        # Uppercase version should match as well (especially relevant for Windows)
        if matched_location is None and item != item.upper():
            matched_location = self._find_match(item.upper())

        return matched_location

    @staticmethod
    def parse_path_options(path: str) -> tuple[Path, dict]:
        """
        Takes a filepath and searches for options in the form `/my/path:option=2;option2=value2`.

        >>> path, options = Datasets.parse_path_options("/my/path:option1=2;option2=value2")
        >>> str(path)
        '/my/path'
        >>> options
        {'option1': '2', 'option2': 'value2'}

        Args:
            path: Filepath string usually coming from `os.environ`.

        Returns: The filepath as Path object without the options and a dictionary with the parsed options.
        """
        options = {}
        path = Path(path)

        if ":" in path.name:
            parts = path.name.split(":")
            assert len(parts) == 2, "Only one option delimiter allowed"
            path = path.with_name(parts[0])

            for o in parts[1].split(";"):
                match = re.search(r"(\w+)=(\w+)", o)
                if match is not None:
                    options[match.group(1)] = match.group(2)

        return path, options


class DatasetAccessor:
    def __init__(self, datasets: Datasets, entry_field: str) -> None:
        self.datasets = datasets
        self.entry_field = entry_field

    def __getitem__(self, item: str) -> Union[Path, None]:
        entry = self.datasets.get(item)
        if entry is None:
            return None
        else:
            return entry[self.entry_field]

    def __getattr__(self, item: str) -> Union[Path, None]:
        return self[item]

    def __contains__(self, item: str) -> bool:
        return item in self.datasets
