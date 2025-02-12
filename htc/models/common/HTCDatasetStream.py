# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import warnings
from abc import abstractmethod
from collections.abc import Iterator
from itertools import cycle

import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler, WeightedRandomSampler

from htc.models.common.class_weights import calculate_class_weights
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.SharedMemoryDatasetMixin import SharedMemoryDatasetMixin
from htc.utils.Config import Config


class HTCDatasetStream(SharedMemoryDatasetMixin, HTCDataset, IterableDataset):
    def __init__(self, *args, single_pass: bool = False, **kwargs):
        """
        Base class for all stream-based datasets (used in non-image models) where each worker contributes equally to a batch (for randomness). Datasets of this class can only be used in conjunction with the StreamDataLoader.

        Args:
            *args: Arguments passed to the parent class.
            sampler: If not None, path indices will be used from this sampler. The order defined by the sampler is preserved by the workers. If None, paths are sampled randomly.
            single_pass: If True, iterates only once over this dataset (instead of endlessly).
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.single_pass = single_pass
        self.buffer_index = None
        self.image_index = None

        # Each worker is responsible for one part of the batch of this size
        self.batch_part_size = self.batch_size // self.num_workers

        if self.config["input/oversampling"]:
            # Class weights for data sampling (not weights for the loss function)
            self.class_weights = calculate_class_weights(
                Config({"model/class_weight_method": "1∕m"}), *self.label_counts()
            )

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Named tensors and all their associated APIs are an experimental feature and subject to change",
        )

    def __iter__(self) -> Iterator[dict[str, int | bool]]:
        assert len(self.shared_dict) > 0, "Shared dictionary is not initialized. Did you forget to call init_shared()?"
        assert self.batch_part_size > 0, (
            f"batch_part_size must not be {self.batch_part_size}. Incompatible batch size"
            f" ({self.config['dataloader_kwargs/batch_size']}) or number of workers"
            f" ({self.num_workers})"
        )

        worker_base_index = self._get_worker_index() * self.batch_part_size
        i = 0
        self.buffer_index = 0
        self.image_index = worker_base_index + i
        expected_keys = set(self.shared_dict.keys())

        for sample in self.iter_samples():  # Each worker iterates over its own samples
            n_samples = None
            self.image_index = worker_base_index + i

            if i == 0:
                sample_keys = set(sample.keys())
                assert expected_keys == sample_keys, (
                    "Every key defined in the shared dictionary must also be set in the sample but the following keys"
                    f" are missing in the sample: {expected_keys - sample_keys}"
                )

            for key, tensor in sample.items():
                assert i == (i % self.batch_part_size), "The index must never go beyond the batch part size"

                if isinstance(tensor, torch.Tensor) and len(tensor.names) > 0 and tensor.names[0] == "B":
                    # Subclasses (e.g. pixel dataset) may directly return batch parts instead of individual samples
                    n_samples_current = tensor.size(0)
                    self.shared_dict[key][
                        self.buffer_index, self.image_index : self.image_index + n_samples_current
                    ] = tensor
                else:
                    # A subclass may already filled the shared memory buffer. In that case, there is nothing more to do
                    if tensor != "pointer":
                        self.shared_dict[key][self.buffer_index, self.image_index] = tensor
                    n_samples_current = 1

                if n_samples is not None:
                    assert n_samples == n_samples_current, "The number of samples must be the same for all keys"
                else:
                    n_samples = n_samples_current

            del sample
            i += n_samples
            assert i <= self.batch_part_size, "The number of samples must never increase over the batch part size"

            if i == self.batch_part_size:
                assert self.buffer_index < self.buffer_size, "Invalid buffer location"
                yield {
                    "buffer_index": self.buffer_index,
                    "start_index": worker_base_index,
                    "end_index": worker_base_index + i,
                    "partly_filled": False,
                }

                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                i = 0

        if i > 0:
            # Last batch is not fully filled
            yield {
                "buffer_index": self.buffer_index,
                "start_index": worker_base_index,
                "end_index": worker_base_index + i,
                "partly_filled": True,
            }

    def __len__(self) -> int:
        raise NotImplementedError(
            "The length of the dataset is not available because the number of samples extracted per image can be"
            " undefined (e.g. number of patches per image)"
        )

    @property
    def buffer_size(self) -> int:
        return self.prefetch_factor + 1  # Shared ring buffer

    @abstractmethod
    def iter_samples(self) -> Iterator[dict[str, torch.Tensor]]:
        """This method must be implemented by all child classes and yields one sample at a time (e.g. one patch at a time) or complete batch parts (e.g. pixel dataset)."""

    def _iter_paths(self) -> Iterator[tuple[int, int]]:
        worker_index = self._get_worker_index()

        if self.sampler is None:
            path_indices = np.array_split(
                self.path_indices_worker.numpy(), self.num_workers
            )  # Split the path indices across all workers

            if self.config["input/dataset_sampling"]:
                path_indices_weights = [
                    self.get_sample_weights([self.paths[idx] for idx in split]).numpy() for split in path_indices
                ]
            else:
                path_indices_weights = [np.ones_like(split) for split in path_indices]

            # generate samples of data paths depending upon the weights assigned to datasets in the config
            indices = np.random.choice(
                path_indices[worker_index],
                len(path_indices[worker_index]),
                replace=False,
                p=path_indices_weights[worker_index] / np.sum(path_indices_weights[worker_index]),
            )
            if self.single_pass:
                # normalize the weights so the probability sums to 1
                for path_index in indices:
                    yield worker_index, path_index
            else:
                for path_index in cycle(indices):
                    yield worker_index, path_index
        else:
            i = 0
            while True:
                # Split indices across workers, e.g. with 3 workers:
                # w1: 0, 3, 6
                # w2: 1, 4, 7
                # w3. 2, 5, 8
                sampler_index = worker_index + i * self.num_workers
                if sampler_index >= len(self.path_indices_worker):
                    break

                path_index = self.path_indices_worker[sampler_index]
                yield worker_index, path_index.item()
                i += 1

    def _sample_pixel_indices(self, labels: torch.Tensor, n_samples: int = None) -> Sampler[int]:
        if n_samples is None:
            n_samples = len(labels)

        if self.config["input/oversampling"]:
            sample_weights = self.class_weights[labels]
            return WeightedRandomSampler(sample_weights, num_samples=n_samples)
        else:
            if n_samples == len(labels):
                return RandomSampler(labels)
            else:
                return RandomSampler(labels, num_samples=n_samples, replacement=True)
