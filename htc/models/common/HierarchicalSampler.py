# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from htc.cpp import hierarchical_bootstrapping
from htc.models.common.utils import adjust_epoch_size
from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper


class HierarchicalSampler(Sampler[int]):
    def __init__(self, paths: list[DataPath], config: Config):
        """
        Sampler which generates random batches and takes the hierarchical structure of the data into account. It is designed for the camera problem and ensures that in each batch every camera is present. The remaining batch size is filled by sampling multiple subjects with replacement (and one image per subject). The sampler can be passed to the Dataloader (sampler argument).

        Args:
            paths: List of paths from which samples are generated.
            config: The configuration of the current run.
        """
        self.config = config

        cam_mapper = DomainMapper(DataSpecification.from_config(self.config), target_domain="camera_index")
        subject_mapper = DomainMapper(DataSpecification.from_config(self.config), target_domain="subject_index")

        self.mapping = {}
        for image_index, path in enumerate(paths):
            camera_index = cam_mapper.domain_index(path.image_name())
            subject_index = subject_mapper.domain_index(path.image_name())

            if camera_index not in self.mapping:
                self.mapping[camera_index] = {}

            if subject_index not in self.mapping[camera_index]:
                self.mapping[camera_index][subject_index] = []

            self.mapping[camera_index][subject_index].append(image_index)

        assert self.config["dataloader_kwargs/batch_size"] >= len(self.mapping), (
            f'The batch size ({self.config["dataloader_kwargs/batch_size"]}) must be >= the number of cameras'
            f" ({len(self.mapping)}) in the training set"
        )

    def __iter__(self) -> Iterator[int]:
        n_subjects = math.ceil(self.config["dataloader_kwargs/batch_size"] / len(self.mapping))
        n_batches = len(self) // self.config["dataloader_kwargs/batch_size"]
        sample_indices = hierarchical_bootstrapping(
            self.mapping, n_subjects=n_subjects, n_images=1, n_bootstraps=n_batches
        )

        if self.config["dataloader_kwargs/batch_size"] % len(self.mapping) != 0:
            for row in range(sample_indices.size(0)):
                selection = torch.randperm(sample_indices.size(1))[: self.config["dataloader_kwargs/batch_size"]]
                assert (
                    selection.size(0) == self.config["dataloader_kwargs/batch_size"]
                ), f"Number of batch indices ({selection.size(0)}) does not much the batch size"
                yield from sample_indices[row, selection].tolist()
        else:
            assert (
                sample_indices.size(1) == self.config["dataloader_kwargs/batch_size"]
            ), f"Number of batch indices ({sample_indices.size(1)}) does not much the batch size"
            for row in range(sample_indices.size(0)):
                yield from sample_indices[row, :].tolist()

    def __len__(self) -> int:
        # We adjust the epoch_size here and not in the init since the epoch_size might not be converted from str to int yet (e.g. for patch dataset)
        adjust_epoch_size(self.config)
        return self.config["input/epoch_size"]
