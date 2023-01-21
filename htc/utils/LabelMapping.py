# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib
import itertools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from htc.cpp import automatic_numpy_conversion, tensor_mapping
from htc.settings import settings
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.Config import Config

if TYPE_CHECKING:
    from htc.tivita.DataPath import DataPath


class LabelMapping:
    def __init__(
        self,
        mapping_name_index: dict[str, int],
        last_valid_label_index: int = None,
        zero_is_invalid: bool = False,
        mapping_index_name: dict[int, str] = None,
        label_colors: dict[str, str] = None,
    ):
        """
        Small helper class to handle different label mappings (e.g. original labels as defined by the clinicians vs. labels used for training).

        Args:
            mapping_name_index: Mapping of label identifiers (name of the organs) to indices (integer values).
            last_valid_label_index: The index of the last valid label in the mapping. This is useful to distinguish between valid and invalid pixels. If None, every label index smaller than settings.label_index_thresh will be considered valid.
            zero_is_invalid: If True, 0 will be treated as invalid label (additional labels may be treated as invalid via last_valid_label_index).
            mapping_index_name: Mapping of label indices to names (used to construct a mapping based on a saved config).
            label_colors: Mapping of label names to color values (e.g. '#ffffff'). If None, settings.label_colors will be used.
        """
        self.mapping_name_index = mapping_name_index
        self.label_colors = label_colors if label_colors is not None else settings.label_colors
        self.zero_is_invalid = zero_is_invalid

        if last_valid_label_index is None:
            valid_indices = [
                i for i in mapping_name_index.values() if i < settings.label_index_thresh
            ]  # We need to find the last valid index
            self.last_valid_label_index = max(valid_indices)
        else:
            self.last_valid_label_index = last_valid_label_index

        if mapping_index_name is not None:
            self.mapping_index_name = mapping_index_name
        else:
            # Reverse mapping
            label_names = []  # Maps from label_index to the (first) label name
            label_indices = []
            for name, i in self.mapping_name_index.items():
                assert type(name) == str, "The keys in the mapping must be strings (e.g. organ names)"
                assert type(i) == int, "The values in the mapping must be indices (e.g. label index values)"

                if (
                    i not in label_indices
                ):  # In case there is more than one mapping to the same label_index, take the first (background)
                    label_names.append(name)
                    label_indices.append(i)

            assert len(label_names) == len(label_indices), "Each label id must have a corresponding label name"
            self.mapping_index_name = {i: name for name, i in zip(label_names, label_indices)}

        # Make sure the mapping is sorted
        self.mapping_index_name = dict(sorted(self.mapping_index_name.items()))  # Sort by key
        self.mapping_name_index = dict(
            sorted(self.mapping_name_index.items(), key=lambda item: item[1])
        )  # Sort by value

        assert set(self.mapping_name_index.values()) == set(
            self.mapping_index_name.keys()
        ), "Both mappings must have the same ids (it is only allowed to have more names for the same id)"

    def __len__(self):
        """Returns: The number of unique valid label indices in this mapping. This is identical to the number of classes used during training. Please note that it is possible to have more names than ids since multiple names can map to the same id.
        """
        return len(self.label_indices())

    def __contains__(self, name_or_index: Union[str, int, torch.Tensor]) -> bool:
        """Returns: True when the given name or index is part of this mapping."""
        if type(name_or_index) == torch.Tensor:
            name_or_index = name_or_index.item()

        if type(name_or_index) == str:
            return name_or_index in self.mapping_name_index
        elif type(name_or_index) == int:
            return name_or_index in self.mapping_index_name
        else:
            raise ValueError(f"Invalid type {type(name_or_index)}")

    def __eq__(self, other: "LabelMapping") -> bool:
        return (
            self.mapping_name_index == other.mapping_name_index
            and self.last_valid_label_index == other.last_valid_label_index
            and self.zero_is_invalid == other.zero_is_invalid
            and self.mapping_index_name == other.mapping_index_name
        )

    def name_to_index(self, name: str) -> int:
        assert name in self.mapping_name_index, f"Cannot find the label {name} in the mapping {self.mapping_name_index}"
        return self.mapping_name_index[name]

    def index_to_name(self, i: Union[int, torch.Tensor], all_names: bool = False) -> Union[str, list[str]]:
        """
        Maps a label_index to its name(s).

        Args:
            i: The label_index.
            all_names: If True, all names are returned if one id maps to multiple names. Otherwise, only the first name is returned.

        Returns: The (first) name for the label (all_names=False) or list of label names (all_names=True).
        """
        if type(i) == torch.Tensor:
            i = i.item()

        assert i in self.mapping_index_name, f"Cannot find the label index {i} in the mapping {self.mapping_index_name}"
        if all_names:
            return [name for name, label_index in self.mapping_name_index.items() if label_index == i]
        else:
            return self.mapping_index_name[i]

    def name_to_color(self, name: str) -> str:
        """
        Map label names to colors.

        >>> m = LabelMapping({'a': 0, 'a_second_name': 0}, last_valid_label_index=0, label_colors={'a': '#FFFFFF'})
        >>> m.name_to_color('a')
        '#FFFFFF'

        If there is more than one name for a label id, then they all map to the same color if no specific mapping is defined for the second name:
        >>> m.name_to_color('a_second_name')
        '#FFFFFF'

        Returns: The (hex) color for the label name as defined in the label_colors mapping. If the name corresponds to an invalid name, label_colors['invalid'] is returned.
        """
        if name in self.label_colors:
            return self.label_colors[name]
        else:
            name = self.index_to_name(
                self.name_to_index(name)
            )  # If there is more than one name for a label, map it to the common label name (= the first label name)
            if name in self.label_colors:
                return self.label_colors[name]
            elif not self.is_index_valid(self.name_to_index(name)) and "invalid" in self.label_colors:
                return self.label_colors["invalid"]
            else:
                raise ValueError(f"Cannot find a color for the label {name}")

    def index_to_color(self, i: Union[int, torch.Tensor]) -> str:
        return self.name_to_color(self.index_to_name(i))

    def is_index_valid(self, i: Union[int, torch.Tensor, np.ndarray]) -> Union[bool, torch.Tensor, np.ndarray]:
        """Returns: True when the given label index (or tensor with indices) is valid according to this mapping."""
        if self.zero_is_invalid:
            return (0 < i) & (i <= self.last_valid_label_index)
        else:
            return i <= self.last_valid_label_index

    def label_names(self, include_invalid: bool = False, all_names: bool = False) -> list[str]:
        """
        List of all label names as defined by this mapping.

        Args:
            include_invalid: If True, also include invalid labels.
            all_names: If True, include all names for a label if there are more than one.

        Returns: The name of each valid label name (include_invalid=False) in this mapping in the order of their corresponding label_index (0, 1, ...). If more than one name maps to the same id, only the first name is returned (all_names=False).
        """
        names = [self.index_to_name(label_index, all_names) for label_index in self.label_indices(include_invalid)]
        if all_names:
            return list(itertools.chain(*names))
        else:
            return names

    def label_indices(self, include_invalid: bool = False) -> list[int]:
        """
        List of label indices as defined by this mapping.

        Args:
            include_invalid: If True, also include invalid label ids.

        Returns: List of ids of all valid labels.
        """
        if include_invalid:
            return list(self.mapping_index_name.keys())
        else:
            return [label_index for label_index in self.mapping_index_name.keys() if self.is_index_valid(label_index)]

    @automatic_numpy_conversion
    def map_tensor(
        self, tensor: Union[torch.Tensor, np.ndarray], old_mapping: "LabelMapping"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Remaps all label ids in the tensor to the new label id as defined by the label mapping of this class (self = new_mapping). Mapping happens based on the label name.

        Args:
            tensor: Tensor or array with label id values which should be remapped.
            old_mapping: The label mapping which defines the name of the labels used in the tensor.

        Returns: The input tensor with labels remapped in-place.
        """
        old_new_mapping = {}
        for original_index in tensor.unique():
            label_name = old_mapping.index_to_name(original_index.item())
            old_new_mapping[original_index.item()] = self.name_to_index(label_name)

        return tensor_mapping(tensor, old_new_mapping)

    def rename(self, rename_dict: dict[str, str]) -> None:
        """
        Rename existing label names to new label names.

        Args:
            rename_dict: dict with key being what label should be renamed and value being the new label name.
        """
        self.mapping_name_index = {
            rename_dict.get(label_name, label_name): label_index
            for label_name, label_index in self.mapping_name_index.items()
        }
        self.label_colors = {
            rename_dict.get(label_name, label_name): color for label_name, color in self.label_colors.items()
        }
        self.mapping_index_name = {
            label_index: rename_dict.get(label_name, label_name)
            for label_index, label_name in self.mapping_index_name.items()
        }

    def to_json(self) -> dict:
        """Returns: All class attributes as dictionary so that the object can be reconstructed again from the dict."""
        return {
            "mapping_name_index": self.mapping_name_index,
            "last_valid_label_index": self.last_valid_label_index,
            "zero_is_invalid": self.zero_is_invalid,
            "mapping_index_name": self.mapping_index_name,
        }

    @classmethod
    def from_path(cls, path: "DataPath") -> "LabelMapping":
        """
        Constructs a label mapping based on the default labels of the dataset accessed via the path object.

        These are the labels as defined by the clinicians.
        """
        label_colors = path.dataset_settings["label_colors"] if "label_colors" in path.dataset_settings else None
        return cls(
            path.dataset_settings["label_mapping"],
            path.dataset_settings["last_valid_label_index"],
            label_colors=label_colors,
        )

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "LabelMapping":
        """
        Similar to from_path() but using the dataset_settings.json from the data directory directly.

        Args:
            data_dir: Path to the data directory which must contain a dataset_settings.json file.
        """
        dsettings = DatasetSettings(data_dir)
        return cls(dsettings["label_mapping"], dsettings["last_valid_label_index"])

    @classmethod
    def from_config(cls, config: Config) -> "LabelMapping":
        """
        Constructs a label mapping as defined in the config file. config['label_mapping'] can be defined as:

        * a LabelMapping instance.
        * a config definition string in the format module>variable (e.g. htc.settings_seg>label_mapping). module must be importable and variable must exist in the module.
        * a dict from a JSON file (as saved via to_class_dict()).
        * a dict with label_name:label_index definitions (like settings_seg.label_mapping) in which case settings.label_index_thresh will be used to determine invalid labels.
        """
        assert "label_mapping" in config, "There is no label mapping in the config file"
        mapping = config["label_mapping"]

        if type(mapping) == str:
            match = re.search(r"^([\w.]+)>(\w+)$", mapping)
            assert match is not None, (
                f"Could not parse the string {mapping} as a valid config definition. It must be in the format"
                " module>variable (e.g. htc.settings_seg>label_mapping) and must refer to a valid Python script"
            )

            module = importlib.import_module(match.group(1))
            if not hasattr(module, match.group(2)):
                # In case settings is an object
                module = getattr(module, match.group(1).split(".")[-1])
            mapping = getattr(module, match.group(2))
            # Now load as usual

        if isinstance(mapping, LabelMapping):
            mapping_obj = mapping
        elif all([var in mapping for var in ("mapping_name_index", "last_valid_label_index", "mapping_index_name")]):
            # This is easier because we have all information we need in the config
            mapping_index_name = {
                int(i): n for i, n in mapping["mapping_index_name"].items()
            }  # JSON only allows strings as keys
            zero_is_invalid = mapping.get("zero_is_invalid", False)
            mapping_obj = cls(
                mapping["mapping_name_index"], mapping["last_valid_label_index"], zero_is_invalid, mapping_index_name
            )
        else:
            if "label_mapping/background" in config and config["label_mapping/background"] == 0:
                # Unfortunately, we need to manually handle the background class as the config files are sorted and abdominal_linen comes before background and also has the label 0
                label_mapping = {"background": 0}
                for label_name, label_index in mapping.items():
                    if label_name != "background":
                        label_mapping[label_name] = label_index
            else:
                label_mapping = mapping

            mapping_obj = cls(label_mapping)

        config["label_mapping"] = mapping_obj  # Cache for future use
        return mapping_obj
