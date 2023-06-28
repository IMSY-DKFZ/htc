# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from htc.models.common.transforms import HTCTransformation, ToType
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping


class HTCDataset(ABC, Dataset):
    def __init__(self, paths: list[DataPath], train: bool, config: Config = None, fold_name: str = None):
        """
        Base class for all datasets used in this repository. It contains basic functionality like reading segmentation masks or handling augmentations.

        Args:
            paths: List of images which should be loaded.
            train: Whether the images should be used during training (basically controls whether the augmentations should be applied or not).
            config: The configuration object specifying all the details for the data loading.
            fold_name: The name of the current training fold. This is necessary when training statistics are needed like label counts (e.g. for class weight methods).
        """
        if config is None:
            config = Config({})

        self.train = train
        self.paths = paths
        self.config = config
        self.fold_name = fold_name
        self.image_names = [p.image_name() for p in self.paths]
        self.features_dtype = (
            torch.float16 if self.config["trainer_kwargs/precision"] in [16, "16-mixed"] else torch.float32
        )
        self.n_channels_loading = self.config["input/n_channels"]  # Value before any channel selection

        # Data transformations
        if self.train and self.config["input/transforms_cpu"]:
            self.transforms = HTCTransformation.parse_transforms(
                self.config["input/transforms_cpu"],
                initial_dtype=self.features_dtype,
                config=self.config,
                fold_name=self.fold_name,
            )
        elif not self.train and self.config["input/test_time_transforms_cpu"]:
            self.transforms = HTCTransformation.parse_transforms(
                self.config["input/test_time_transforms_cpu"],
                initial_dtype=self.features_dtype,
                config=self.config,
                fold_name=self.fold_name,
            )
        else:
            self.transforms = HTCTransformation.parse_transforms(initial_dtype=self.features_dtype)

        if self.config["input/channel_selection"] and self.n_channels_loading == 100:
            assert (
                type(self.config["input/channel_selection"]) == list
                and len(self.config["input/channel_selection"]) == 2
            )
            start_channel, end_channel = self.config["input/channel_selection"]
            assert start_channel < end_channel
            self.config["input/n_channels"] = end_channel - start_channel

        # Allow to pass "500 images" as epoch size
        # We need this logic here to convert the epoch_size as early as possible so that everything which relies on this number can work as expected
        if type(self.config["input/epoch_size"]) == str and self.n_image_elements() is not None:
            match = re.search(r"(\d+)\s+images?", self.config["input/epoch_size"])
            assert match is not None, 'If epoch_size is a string, it must have the format "N images"'

            self.config["input/epoch_size_original"] = self.config["input/epoch_size"]
            self.config["input/epoch_size"] = self.n_image_elements() * int(match.group(1))

    @abstractmethod
    def __len__(self) -> int:
        pass

    def n_image_elements(self) -> int:
        """
        Computes the number of samples which a dataset extracts from one image (ideally, without considering invalid pixels).

        For example, in case of the pixel model, this would be the number of pixels in an image since each pixel constitutes a sample.
        """
        pass

    def apply_transforms(
        self, sample: Union[dict[str, torch.Tensor], torch.Tensor]
    ) -> Union[dict[str, torch.Tensor], torch.Tensor]:
        assert (
            len(self.transforms) >= 1 and type(self.transforms[0]) == ToType
        ), "There must always be the ToType transformation"

        was_tensor = False
        if type(sample) == torch.Tensor:
            # We also allow tensors to be passed directly to this function but the transforms always expect a sample dict
            sample = {"features": sample}
            was_tensor = True

        if len(self.transforms) == 1:
            # Shortcut if we only have the ToType transformation (for performance reasons)
            sample = self.transforms[0](sample)
        else:
            # Add batch dimension (transformations always work on batches, not samples)
            batch = {}
            for key, value in sample.items():
                if type(value) == torch.Tensor:
                    batch[key] = value.unsqueeze(dim=0)
                else:
                    batch[key] = [value]

            batch = HTCTransformation.apply_valid_transforms(batch, self.transforms)

            # Remove batch dimension
            for key, value in batch.items():
                if key == "transforms_applied":
                    continue

                if type(value) == torch.Tensor:
                    sample[key] = value.squeeze(dim=0)
                else:
                    sample[key] = value[0]

            assert "transforms_applied" not in sample

        if was_tensor:
            sample = sample["features"]

        return sample

    def label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: label values and corresponding counts used for class weighting.
        """
        assert self.fold_name is not None, "The fold name must be provided if label counts are needed"

        specs = DataSpecification.from_config(self.config)
        mapping = LabelMapping.from_config(self.config)
        image_names = [p.image_name() for p in specs.fold_paths(self.fold_name, "^train")]

        # We use the median tables to calculate the label count information
        df = median_table(image_names=image_names, label_mapping=mapping)
        df = df.groupby("label_index_mapped", as_index=False)["n_pixels"].sum()
        df.sort_values(by="label_index_mapped", inplace=True)

        return torch.from_numpy(df["label_index_mapped"].values), torch.from_numpy(df["n_pixels"].values)

    def read_labels(self, path: DataPath) -> Union[dict[str, torch.Tensor], None]:
        """
        Read the labels for the data path, compute the valid pixels and apply the label mapping.

        Args:
            path: Data path to the image.

        Returns: Dictionary with the matrices for the labels and the valid pixels (True if a pixel is valid) or None if the path does not have labels. If more than one annotation_name is requested (config["input/annotation_name"]), all annotations are included in the sample with labels_annotation_name as naming schema.
        """
        if self.config["input/no_labels"]:
            return None

        seg_data = path.read_segmentation(self.config["input/annotation_name"])
        if seg_data is None:
            return None

        if type(seg_data) != dict:
            seg_data = {None: seg_data}

        sample = {}
        for annotation_name, seg in seg_data.items():
            # Map the label ids in the segmentation file to the label ids used in the current experiment
            if "label_mapping" in self.config:
                if self.config["label_mapping"] is None:
                    # This does not change the labels
                    label_mapping = LabelMapping.from_path(path)
                elif isinstance(self.config["label_mapping"], LabelMapping):
                    label_mapping = self.config["label_mapping"]
                else:
                    label_mapping = LabelMapping.from_config(self.config)
            else:
                # Default is not to change the labels
                label_mapping = LabelMapping.from_path(path)

            original_mapping = LabelMapping.from_path(path)
            seg = torch.from_numpy(seg)
            labels = label_mapping.map_tensor(seg, original_mapping).type(torch.int64)
            valid_pixels = label_mapping.is_index_valid(labels)

            if annotation_name is None:
                sample["labels"] = labels
                sample["valid_pixels"] = valid_pixels
            elif self.config["input/merge_annotations"] == "union":
                if "labels" not in sample:
                    sample["labels"] = torch.ones_like(labels) * settings.label_index_thresh
                if "valid_pixels" not in sample:
                    sample["valid_pixels"] = torch.zeros_like(labels, dtype=torch.bool)

                sample["labels"][valid_pixels] = labels[valid_pixels]
                sample["valid_pixels"][valid_pixels] = True
            else:
                sample[f"labels_{annotation_name}"] = labels
                sample[f"valid_pixels_{annotation_name}"] = valid_pixels

        assert len(sample) > 0, "No labels found"
        return sample

    def read_experiment(self, path: DataPath) -> dict[str, torch.Tensor]:
        """
        Reads the experiment data of one image.

        Args:
            path: Path to the experiment folder.

        Returns: Dictionary with tensors of the features, labels, etc. ready to be used in the network.
        """
        if self.config["input/no_features"]:
            data = None
        elif self.config["input/preprocessing"]:
            data = self._load_preprocessed(path.image_name(), self.config["input/preprocessing"])
        else:
            assert self.n_channels_loading != 0, (
                "At least one channel is necessary (please use input/no_features if you do not want to load any"
                f" features at all): {self.n_channels_loading = }"
            )

            if self.n_channels_loading == 3:  # RGB
                data = path.read_rgb_reconstructed() / 255
            else:  # HSI
                normalization = 1 if self.config["input/normalization"] == "L1" else None
                data = path.read_cube(normalization)

        sample = self.read_labels(path)
        if sample is None:
            sample = {}

        if data is not None:
            sample["features"] = torch.from_numpy(data)

        if folder_names := self.config["input/preprocessing_additional"]:
            for name in folder_names:
                data_additional = self._load_preprocessed(path.image_name(), name)
                sample[f"data_{name}"] = torch.from_numpy(data_additional)

        if self.config["input/channel_selection"] and "features" in sample:
            start_channel, end_channel = self.config["input/channel_selection"]
            sample["features"] = sample["features"][:, :, start_channel:end_channel]

        if self.config["input/n_channels"]:
            if not self.config["input/no_features"]:
                assert sample["features"].shape[-1] == self.config["input/n_channels"], (
                    f'Number of feature channels ({sample["features"].shape = }) does not correspond to the number of'
                    f' channels in the config {self.config["input/n_channels"] = }'
                )
            else:
                assert "features" not in sample

        for name, tensor in sample.items():
            if name.startswith(("labels", "valid_pixels")):
                # May be CHW
                spatial_shape = tensor.shape[-2:]
            else:
                # May be HWC
                spatial_shape = tensor.shape[:2]

            assert spatial_shape == path.dataset_settings["spatial_shape"], (
                f"All tensors must agree in the spatial dimension but the tensor {name} has only a shape of"
                f" {tensor.shape}"
            )

        sample["image_name"] = path.image_name()

        return sample

    def get_sample_weights(self, paths: list[DataPath]) -> torch.Tensor:
        # Create a weighting for the different datasets (e.g. show semantic images more often)
        dataset_names = list(self.config["input/dataset_sampling"].keys())
        dataset_indices = torch.empty(len(paths), dtype=torch.int64)
        for i, path in enumerate(paths):
            dataset_indices[i] = dataset_names.index(path.dataset_settings["dataset_name"])

        # We first make a weight based on the imbalance of the datasets itself so that images from both datasets have an equal chance to be selected
        weights = torch.ones(len(dataset_names))
        for i in range(len(dataset_names)):
            n_samples = (dataset_indices == i).sum()
            if n_samples > 0:
                weights[i] = 1 / n_samples
            else:
                settings.log.warning(
                    f"Could not find any sample from the dataset {dataset_names[i]}. The weighting will likely not"
                    " having any effect"
                )
        weights = weights / weights.sum()

        # Then this normalized weight is adapted according to the user settings
        dataset_sampling_sum = sum(list(self.config["input/dataset_sampling"].values()))
        for i, name in enumerate(dataset_names):
            weights[i] = weights[i] * self.config["input/dataset_sampling"][name] / dataset_sampling_sum
        weights = weights / weights.sum()

        assert not torch.isnan(weights).any(), f"Some dataset sampling weights contain NAN values: {weights}"

        sample_weights = weights[dataset_indices]
        return sample_weights

    def _load_preprocessed(self, image_name: str, folder_name: str) -> Union[np.ndarray, None]:
        # Load a preprocessed HSI cube
        files_dir = settings.intermediates_dir_all / "preprocessing" / folder_name

        extensions = [
            (".blosc", decompress_file),
            (".npy", np.load),
            (".npz", lambda p: np.load(p, allow_pickle=True)["data"]),
        ]
        data = None
        for ext, load_func in extensions:
            file_path = files_dir / f"{image_name}{ext}"
            if file_path.exists():
                data = load_func(file_path)
                break

        assert data is not None, (
            f"Could not ind the image {image_name}. This probably means that you have not registered the dataset where"
            " this image is from, i.e. you need to set the corresponding environment variable. The following"
            f" intermediate directories are registered: {settings.intermediates_dir_all} and the following file"
            f" extensions were tried: {[e[0] for e in extensions]}"
        )

        if self.features_dtype != torch.float16:
            data_dtype = next(iter(data.values())).dtype if type(data) == dict else data.dtype
            if data_dtype == np.float16:
                settings.log_once.warning(
                    f"You have set the precision to {self.features_dtype} but the preprocessed data has a precision of"
                    f" {data_dtype} which means that you will not work with the full precision of the original data"
                )

        if folder_name == "parameter_images":
            names = self.config.get("input/parameter_names", ["StO2", "NIR", "TWI", "OHI"])
            assert len(names) > 0, "At least the name of one parameter is required"
            assert all(n in data for n in names), "Not all parameter names are stored in the preprocessed file"

            # Store all parameters in one array
            array = np.empty((*data[names[0]].shape, len(names)), data[names[0]].dtype)
            for i, name in enumerate(names):
                array[..., i] = data[name]

            data = array

        return data

    def _possible_annotation_names(self) -> list[str]:
        """
        Unique list of all possible annotation names which may encounter from the paths.

        If 'input/annotation_name' is set to 'all', an iteration over all paths to check for possible names is required, otherwise, the value of 'input/annotation_name' is used directly.

        Returns: List of annotation names.
        """
        requested_names = self.config["input/annotation_name"]
        assert requested_names is not None

        if requested_names == "all":
            # We need to check what are possible annotation names
            names = set()
            for p in self.paths:
                names.update(p.meta("annotation_name"))

            return list(names)
        else:
            # Specific list of annotation names requested
            if type(requested_names) == str:
                requested_names = requested_names.split("&")
            assert type(requested_names) == list

            return requested_names

    @classmethod
    def example_dataset(cls, config: Config = None):
        """
        Example dataset based on the validation images in the semantic dataset.

        Args:
            config: If not None, a custom configuration can be used. Do not forget to set the label mapping.

        Returns: Dataset instance (e.g. DatasetImage).
        """
        paths = DataSpecification("pigs_semantic-only_5foldsV2.json").paths()
        if config is None:
            config = Config({"label_mapping": settings_seg.label_mapping})
        return cls(paths, train=False, config=config)
