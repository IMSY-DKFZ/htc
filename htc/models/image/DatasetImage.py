# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper
from htc.utils.SLICWrapper import SLICWrapper


class DatasetImage(HTCDataset):
    """
    This is the basic dataset for reading images via an index-based way. The index corresponds to the path in the paths list.

    If no config is provided, the default is used which reads HSI images, its labels and a mask with valid pixels without applying augmentations.

    >>> from htc.tivita.DataPath import DataPath
    >>> paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
    >>> sample = DatasetImage(paths, train=False)[0]
    >>> list(sample.keys())
    ['labels', 'valid_pixels', 'features', 'image_name', 'image_name_annotations', 'image_index']
    >>> sample["image_name"]
    'P043#2019_12_20_12_38_35'
    >>> sample["image_index"]
    0
    >>> sample["features"].shape
    torch.Size([480, 640, 100])
    >>> sample["labels"].shape
    torch.Size([480, 640])

    You can also read the RGB images by providing a config file:

    >>> from htc.utils.Config import Config
    >>> sample = DatasetImage(paths, train=False, config=Config({"input/n_channels": 3}))[0]
    >>> sample["features"].shape
    torch.Size([480, 640, 3])

    Similar for reading the parameter images (STO2, NIR, TWI and OHI) € [0;1]:

    >>> sample = DatasetImage(
    ...     paths, train=False, config=Config({"input/n_channels": 4, "input/preprocessing": "parameter_images"})
    ... )[0]
    >>> sample["features"].shape
    torch.Size([480, 640, 4])

    In case you are not interested in any features and only want the labels, simply do

    >>> sample = DatasetImage(paths, train=False, config=Config({"input/no_features": True}))[0]
    >>> list(sample.keys())
    ['labels', 'valid_pixels', 'image_name', 'image_name_annotations', 'image_index']
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_domains = self.config.get("input/target_domain", [])
        if len(self.target_domains) > 0:
            specs = DataSpecification.from_config(self.config)

            with specs.activated_test_set():
                self.domain_mapper = DomainMapper.from_config(self.config)

    def __getitem__(self, index: int, start_pointers: dict[str, int] = None) -> dict[str, torch.Tensor]:
        sample = self.read_experiment(self.paths[index], start_pointers=start_pointers)
        sample["image_index"] = index

        if self.config["input/superpixels"] and not self.config["input/no_features"]:
            if self.config["input/n_channels"] != 3:
                # We always calculate the superpixels on the RGB image since we only want to compare the features and not the shape

                # Load the RGB image via the dataset so that potential transformations may be applied
                config_rgb = Config({"input/n_channels": 3})
                if "label_mapping" in self.config:
                    config_rgb["label_mapping"] = self.config["label_mapping"]
                if "input/test_time_transforms_cpu" in self.config:
                    config_rgb["input/test_time_transforms_cpu"] = self.config["input/test_time_transforms_cpu"]
                sample_rgb = DatasetImage([self.paths[index]], train=False, config=config_rgb)[0]

                sample["features_rgb"] = sample_rgb["features"]
                spx_features_name = "features_rgb"
            else:
                # We already have the RGB data so we can directly use it for the superpixels
                spx_features_name = "features"

        # We need to apply the transformations before we compute the superpixels because the superpixel mask cannot be transformed
        # The main problem is that the border values get mirrored leading to duplicate superpixel indices or missing indices
        sample = self.apply_transforms(sample)  # e.g. features.shape = [480, 640, 100]

        if self.config["input/superpixels"] and not self.config["input/no_features"]:
            fast_slic = SLICWrapper(**self.config["input/superpixels"])
            sample["spxs"] = fast_slic.apply_slic(sample[spx_features_name])

            if spx_features_name == "features_rgb":
                # We only needed the rgb features to calculate the superpixels
                del sample["features_rgb"]

        for domain in self.target_domains:
            sample[domain] = self.domain_mapper[domain].domain_index(sample["image_name"])

        return sample

    def __len__(self) -> int:
        return len(self.paths)
