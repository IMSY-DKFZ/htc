# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from typing_extensions import Self

from htc.cpp import automatic_numpy_conversion, tensor_mapping
from htc.settings import settings
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.Config import Config
from htc.utils.Task import Task
from htc.utils.type_from_string import variable_from_string

if TYPE_CHECKING:
    from htc.tivita.DataPath import DataPath


class LabelMapping:
    def __init__(
        self,
        mapping_name_index: dict[str, int],
        last_valid_label_index: int = None,
        zero_is_invalid: bool = False,
        unknown_invalid: bool = False,
        mapping_index_name: dict[int, str] = None,
        label_colors: dict[str, str] = None,
    ):
        """
        Small helper class to handle different label mappings (e.g. original labels as defined by the clinicians vs. labels used for training).

        Args:
            mapping_name_index: Mapping of label identifiers (name of the organs) to indices (integer values).
            last_valid_label_index: The index of the last valid label in the mapping. This is useful to distinguish between valid and invalid pixels. If None, every label index smaller than settings.label_index_thresh will be considered valid.
            zero_is_invalid: If True, 0 will be treated as invalid label (additional labels may be treated as invalid via last_valid_label_index).
            unknown_invalid: If True, all labels which are not part of this mapping are considered invalid (except of raising an error).
            mapping_index_name: Mapping of label indices to names (used to construct a mapping based on a saved config).
            label_colors: Mapping of label names to color values (e.g. '#ffffff'). If None, settings.label_colors will be used.
        """
        self.mapping_name_index = mapping_name_index
        self.label_colors = label_colors if label_colors is not None else settings.label_colors
        self.zero_is_invalid = zero_is_invalid
        self.unknown_invalid = unknown_invalid

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
                assert type(i) == int, "The values in the mapping must be indices (e.g. label index values)"

                if (
                    i not in label_indices
                ):  # In case there is more than one mapping to the same label_index, take the first (background)
                    label_names.append(name)
                    label_indices.append(i)

            assert len(label_names) == len(label_indices), "Each label id must have a corresponding label name"
            self.mapping_index_name = {i: name for name, i in zip(label_names, label_indices, strict=True)}

        # Make sure the mapping is sorted
        self.mapping_index_name = dict(sorted(self.mapping_index_name.items()))  # Sort by key
        self.mapping_name_index = dict(
            sorted(self.mapping_name_index.items(), key=lambda item: item[1])
        )  # Sort by value

        assert set(self.mapping_name_index.values()) == set(
            self.mapping_index_name.keys()
        ), "Both mappings must have the same ids (it is only allowed to have more names for the same id)"

    def __repr__(self):
        labels = ", ".join([f"{l}={self.name_to_index(l)}" for l in self.label_names()])
        return f"LabelMapping({labels})"

    def __len__(self):
        """Returns: The number of unique valid label indices in this mapping. This is identical to the number of classes used during training. Please note that it is possible to have more names than ids since multiple names can map to the same id."""
        return len(self.label_indices())

    def __contains__(self, name_or_index: str | int | torch.Tensor) -> bool:
        """Returns: True when the given name or index is part of this mapping (valid or invalid)."""
        if type(name_or_index) == torch.Tensor:
            name_or_index = name_or_index.item()

        if type(name_or_index) == str:
            return name_or_index in self.mapping_name_index
        elif type(name_or_index) == int:
            return name_or_index in self.mapping_index_name
        else:
            raise ValueError(f"Invalid type {type(name_or_index)}")

    def __eq__(self, other: Self) -> bool:
        return (
            self.mapping_name_index == other.mapping_name_index
            and self.last_valid_label_index == other.last_valid_label_index
            and self.zero_is_invalid == other.zero_is_invalid
            and self.unknown_invalid == other.unknown_invalid
            and self.mapping_index_name == other.mapping_index_name
        )

    def name_to_index(self, name: str) -> int:
        """
        Maps a label name to the corresponding label index.

        Args:
            name: Name of the label.

        Returns: The index associated with the label name.
        """
        if name in self.mapping_name_index:
            return self.mapping_name_index[name]
        else:
            if self.unknown_invalid:
                # One above the last known index
                return list(self.mapping_index_name.keys())[-1] + 1
            else:
                raise ValueError(f"Cannot find the label {name} in the mapping {self.mapping_name_index}")

    def index_to_name(self, i: int | torch.Tensor, all_names: bool = False) -> str | list[str]:
        """
        Maps a label_index to its name(s).

        Args:
            i: The label_index.
            all_names: If True, all names are returned if one id maps to multiple names. Otherwise, only the first name is returned.

        Returns: The (first) name for the label (all_names=False) or list of label names (all_names=True).
        """
        if type(i) == torch.Tensor:
            i = i.item()

        if i in self.mapping_index_name:
            if all_names:
                return [name for name, label_index in self.mapping_name_index.items() if label_index == i]
            else:
                return self.mapping_index_name[i]
        else:
            if self.unknown_invalid:
                return "unknown"
            else:
                raise ValueError(f"Cannot find the label index {i} in the mapping {self.mapping_index_name}")

    def name_to_color(self, name: str) -> str:
        """
        Map label names to colors.

        >>> m = LabelMapping({"a": 0, "a_second_name": 0}, last_valid_label_index=0, label_colors={"a": "#FFFFFF"})
        >>> m.name_to_color("a")
        '#FFFFFF'

        If there is more than one name for a label id, then they all map to the same color if no specific mapping is defined for the second name:
        >>> m.name_to_color("a_second_name")
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

    def index_to_color(self, i: int | torch.Tensor) -> str:
        return self.name_to_color(self.index_to_name(i))

    def is_index_valid(self, i: int | torch.Tensor | np.ndarray) -> bool | torch.Tensor | np.ndarray:
        """Returns: True when the given label index (or tensor with indices) is valid according to this mapping."""
        if self.zero_is_invalid:
            return (0 < i) & (i <= self.last_valid_label_index)
        else:
            return i <= self.last_valid_label_index

    def is_name_valid(self, name: str) -> bool:
        """Returns: True when the given label name is valid according to this mapping."""
        return self.is_index_valid(self.name_to_index(name))

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
    def map_tensor(self, tensor: torch.Tensor | np.ndarray, old_mapping: Self) -> torch.Tensor | np.ndarray:
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

    def rename(self, rename_mapping: dict[str, str]) -> None:
        """
        Rename existing label names to new label names.

        Args:
            rename_mapping: Mapping with key being what label should be renamed and value being the new label name.
        """
        self.mapping_name_index = {
            rename_mapping.get(label_name, label_name): label_index
            for label_name, label_index in self.mapping_name_index.items()
        }
        self.label_colors = {
            rename_mapping.get(label_name, label_name): color for label_name, color in self.label_colors.items()
        }
        self.mapping_index_name = {
            label_index: rename_mapping.get(label_name, label_name)
            for label_index, label_name in self.mapping_index_name.items()
        }

    def append(self, name: str, invalid: bool = False) -> None:
        """
        Append a new label name to the mapping.

        For valid labels, the new label index will be last_valid_label_index + 1. For invalid labels, the new label index will be the first unused index.

        Args:
            name: The name of the new label.
            invalid: If True, the new label will be treated as invalid.
        """

        def get_unused_index() -> int:
            # Find the first free index
            # 256 because we usually store segmentation maps as uint8
            for i in range(self.last_valid_label_index + 1, 256):
                if i not in self.mapping_index_name:
                    return i

            raise ValueError("No unused label index found (reached maximum of 255)")

        if name not in self.mapping_name_index:
            if invalid:
                unused_index = get_unused_index()
                self.mapping_name_index[name] = unused_index
                self.mapping_index_name[unused_index] = name
            else:
                self.mapping_name_index[name] = self.last_valid_label_index + 1
                self.last_valid_label_index += 1

                if self.last_valid_label_index in self.mapping_index_name:
                    # self.last_valid_label_index + 1 is already used by an invalid label -> assign a new index to the invalid label
                    unused_index = get_unused_index()
                    self.mapping_index_name[unused_index] = self.mapping_index_name[self.last_valid_label_index]
                    self.mapping_name_index[self.mapping_index_name[self.last_valid_label_index]] = unused_index

                self.mapping_index_name[self.last_valid_label_index] = name

    def to_json(self) -> dict:
        """Returns: All class attributes as dictionary so that the object can be reconstructed again from the dict."""
        return {
            "mapping_name_index": self.mapping_name_index,
            "last_valid_label_index": self.last_valid_label_index,
            "zero_is_invalid": self.zero_is_invalid,
            "unknown_invalid": self.unknown_invalid,
            "mapping_index_name": self.mapping_index_name,
        }

    @classmethod
    def from_path(cls, path: "DataPath") -> Self:
        """
        Constructs a label mapping based on the default labels of the dataset accessed via the path object.

        These are the labels as defined by the clinicians.

        Args:
            path: Data path to the image.
        """
        label_colors = path.dataset_settings["label_colors"] if "label_colors" in path.dataset_settings else None
        return cls(
            path.dataset_settings["label_mapping"],
            path.dataset_settings["last_valid_label_index"],
            label_colors=label_colors,
        )

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> Self:
        """
        Similar to from_path() but using the dataset_settings.json from the data directory directly.

        Args:
            data_dir: Path to the data directory which must contain a dataset_settings.json file.
        """
        dsettings = DatasetSettings(data_dir)
        return cls(dsettings["label_mapping"], dsettings["last_valid_label_index"])

    @classmethod
    def from_config(cls, config: Config, task: Task = None, image_label_entry_index: int = 0) -> Self:
        """
        Constructs a label mapping as defined in the config file. For example, `config['label_mapping']` can be defined as:

        * a LabelMapping instance.
        * a config definition string in the format module>variable (e.g. `htc.settings_seg>label_mapping`). module must be importable and variable must exist in the module.
        * a dict from a JSON file (as saved via `to_class_dict()`).
        * a dict with label_name:label_index definitions (like `settings_seg.label_mapping`) in which case `settings.label_index_thresh` will be used to determine invalid labels.

        Args:
            config: The config object.
            task: The task for which the mapping should be constructed. For segmentation tasks, the mapping must be defined in `config['label_mapping']` and for classification tasks it must be defined in `config['input/image_labels'][image_label_entry_index]['image_label_mapping']`. If None, the task will be determined from the config.
            image_label_entry_index: The index of the config['input/image_labels'] list in the config file (used only for classification tasks).
        """
        if task is None:
            task = Task.from_config(config)

        if task == Task.SEGMENTATION:
            assert "label_mapping" in config, "There is no label mapping in the config file"
            mapping = config["label_mapping"]
        elif task == Task.CLASSIFICATION:
            assert "input/image_labels" in config, "There must be image labels defined for classification tasks"
            mapping = config["input/image_labels"][image_label_entry_index]["image_label_mapping"]
        else:
            raise ValueError(f"Invalid task: {task}")

        if type(mapping) == str:
            mapping = variable_from_string(mapping)

        if isinstance(mapping, LabelMapping):
            mapping_obj = mapping
        elif all(var in mapping for var in ("mapping_name_index", "last_valid_label_index", "mapping_index_name")):
            mapping_name_index = {}
            for name, index in mapping["mapping_name_index"].items():
                if name == "true":
                    name = True
                if name == "false":
                    name = False
                mapping_name_index[name] = index

            # This is easier because we have all information we need in the config
            mapping_index_name = {
                int(i): n for i, n in mapping["mapping_index_name"].items()
            }  # JSON only allows strings as keys
            zero_is_invalid = mapping.get("zero_is_invalid", False)
            unknown_invalid = mapping.get("unknown_invalid", False)
            mapping_obj = cls(
                mapping_name_index,
                mapping["last_valid_label_index"],
                zero_is_invalid,
                unknown_invalid,
                mapping_index_name,
            )
        else:
            if "label_mapping/background" in config and config["label_mapping/background"] == 0:
                # Unfortunately, we need to manually handle the background class as the config files are sorted and abdominal_linen comes before background and also has the label 0
                label_mapping = {"background": 0}
                label_mapping.update({
                    label_name: label_index for label_name, label_index in mapping.items() if label_name != "background"
                })
            else:
                label_mapping = mapping

            mapping_obj = cls(label_mapping)

        # Cache for future use
        if task == Task.SEGMENTATION:
            config["label_mapping"] = mapping_obj
        elif task == Task.CLASSIFICATION:
            config["input/image_labels"][image_label_entry_index]["image_label_mapping"] = mapping_obj
        else:
            raise ValueError(f"Invalid task: {task}")

        return mapping_obj
