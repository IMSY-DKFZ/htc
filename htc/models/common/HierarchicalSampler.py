# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from htc.cpp import hierarchical_bootstrapping, hierarchical_bootstrapping_labels
from htc.models.common.utils import adjust_epoch_size
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper
from htc.utils.LabelMapping import LabelMapping


class HierarchicalSampler(Sampler[int]):
    def __init__(self, paths: list[DataPath], config: Config, batch_size: int = None):
        """
        Sampler which generates random batches and takes the hierarchical structure of the data into account, especially the domain. It is designed for the camera problem and ensures that in each batch every camera (domain) is present. The remaining batch size is filled by sampling multiple subjects with replacement (and one image per subject). The sampler can be passed to the Dataloader (sampler argument).

        You must specify `input/target_domain` in your config (e.g. `config["input/target_domain"] = ["camera_index"]`) to define the domain used in the hierarchical sampling (if more than one domain is specified, only the first will be used).

        Args:
            paths: List of paths from which samples are generated.
            config: The configuration of the current run.
            batch_size: The batch size, i.e. the number of images per iteration. If `None`, `dataloader_kwargs/batch_size` from the config will be used which is recommended for image-based datasets. For other stream-based datasets (e.g. patch dataset), the situation is more difficult since the batch size does not correspond simply to the number of images. In this case, you can specify the value manually, e.g. to `dataloader_kwargs/num_workers` to make sure that the workers operate on the images given by this sampler.
        """
        self.config = config
        self.batch_size = batch_size or config["dataloader_kwargs/batch_size"]

        target_domain = self.config["input/target_domain"]
        assert (
            target_domain is not None and len(target_domain) >= 1
        ), "At least one target domain must be specified in the config (input/target_domain)"
        if len(target_domain) > 1:
            settings.log.info(
                f"More than one target domain specified. Only the first one ({target_domain[0]}) will be used for"
                " hierarchical sampling"
            )

        domain_mapper = DomainMapper(DataSpecification.from_config(self.config), target_domain=target_domain[0])
        subject_mapper = DomainMapper(DataSpecification.from_config(self.config), target_domain="subject_index")

        if self.config["input/hierarchical_sampling"] == "labels":
            self.label_mapping = {}
            mapping = LabelMapping.from_config(self.config)
        else:
            assert self.config.get(
                "input/hierarchical_sampling", True
            ), "At the moment, only True or labels can be used for `input/hierarchical_sampling`"
            self.label_mapping = None

        self.domain_mapping = {}
        for image_index, path in enumerate(paths):
            domain_index = domain_mapper.domain_index(path.image_name())
            subject_index = subject_mapper.domain_index(path.image_name())

            if domain_index not in self.domain_mapping:
                self.domain_mapping[domain_index] = {}

            if subject_index not in self.domain_mapping[domain_index]:
                self.domain_mapping[domain_index][subject_index] = []

            self.domain_mapping[domain_index][subject_index].append(image_index)

            if self.config["input/hierarchical_sampling"] == "labels":
                for label in path.annotated_labels():
                    label_index = mapping.name_to_index(label)
                    if not mapping.is_index_valid(label_index):
                        continue

                    if label_index not in self.label_mapping:
                        self.label_mapping[label_index] = {}

                    if subject_index not in self.label_mapping[label_index]:
                        self.label_mapping[label_index][subject_index] = []

                    self.label_mapping[label_index][subject_index].append(image_index)

        assert self.config["dataloader_kwargs/batch_size"] >= len(self.domain_mapping), (
            f'The batch size ({self.config["dataloader_kwargs/batch_size"]}) must be >= the number of domains'
            f" ({len(self.domain_mapping)}) in the training set"
        )

    def __iter__(self) -> Iterator[int]:
        n_subjects = math.ceil(self.batch_size / len(self.domain_mapping))
        n_batches = len(self) // self.batch_size

        if self.label_mapping is None:
            sample_indices = hierarchical_bootstrapping(
                self.domain_mapping, n_subjects=n_subjects, n_images=1, n_bootstraps=n_batches
            )
        else:
            sample_indices = hierarchical_bootstrapping_labels(
                self.domain_mapping, self.label_mapping, n_labels=n_subjects, n_bootstraps=n_batches
            )

        assert (
            sample_indices.size(1) >= self.batch_size
        ), f"Number of batch indices ({sample_indices.size(1)}) does not much the batch size"
        for row in range(sample_indices.size(0)):
            # The order of images in a batch should never matter, so we make a random selection per batch
            selection = torch.randperm(sample_indices.size(1))[: self.batch_size]
            yield from sample_indices[row, selection].tolist()

    def __len__(self) -> int:
        # We adjust the epoch_size here and not in the init since the epoch_size might not be converted from str to int yet (e.g. for patch dataset)
        adjust_epoch_size(self.config)
        return self.config["input/epoch_size"]
