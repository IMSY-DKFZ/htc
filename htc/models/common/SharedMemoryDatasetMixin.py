# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import inspect
from abc import abstractmethod
from collections.abc import Iterable
from typing import Union

import torch
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.sampler import Sampler

from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class SharedMemoryDatasetMixin:
    def __init__(self, *args, sampler: Union[Sampler, Iterable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.shared_dict = {}

        # Each worker operates on its own set of paths and we use a shared memory tensor which stores a random set of path indices
        if self.sampler is None:
            size = max(len(self.paths), self.config["dataloader_kwargs/num_workers"])
        else:
            size = len(self.sampler)
        self.path_indices_worker = torch.empty(size, dtype=torch.int64).share_memory_()

        # Shared memory settings
        prefetch_factor_default = (
            inspect.signature(DataLoader).parameters["prefetch_factor"].default
        )  # Default for prefetch factor is (usually) 2 in PyTorch
        self.prefetch_factor = self.config.get("dataloader_kwargs/prefetch_factor", prefetch_factor_default)

        # We always shuffle the paths in the beginning in case this class is used without the StreamDataLoader
        # For training, this means that paths are shuffled twice in the beginning (since the StreamDataLoader always shuffles paths in the beginning of each iteration)
        self.shuffle_paths()

    def __del__(self) -> None:
        # We should unpin the tensor memory when objects of this class get destructed
        # Event though this is not RAII in Python (https://en.wikibooks.org/wiki/Python_Programming/Context_Managers#Not_RAII)
        try:
            if torch.cuda is not None:
                cudart = torch.cuda.cudart()

                for key, tensor in self.shared_dict.items():
                    if type(tensor) == torch.Tensor and tensor.is_pinned():
                        code = cudart.cudaHostUnregister(tensor.data_ptr())
                        assert not tensor.is_pinned(), f"Cannot unpin the tensor {key}: {code = }"
        except RuntimeError:
            # We cannot free the memory if we are in a forked process as we otherwise may get the following error:
            # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
            pass

    @property
    @abstractmethod
    def buffer_size(self) -> int:
        """Total size of the buffer which should be allocated."""
        pass

    def init_shared(self) -> None:
        """
        Initializes shared memory by adding a tensor for the worker id and all tensors which the child class needs.

        This method must be called if iteration of this dataset is required.
        """
        if len(self.shared_dict) == 0:
            self._add_tensor_shared("worker_index", torch.int64)
            self._add_shared_resources()

    def shuffle_paths(self):
        assert len(self.paths) > 0, "At least one data path is required"

        if self.sampler is None:
            indices = torch.randperm(len(self.paths))
            if len(self.paths) < self.config["dataloader_kwargs/num_workers"]:
                # In extreme cases (e.g. in the dataset size experiment) it is possible that less paths than workers are available
                # In this case, we just repeat the indices for the paths we have so that every worker has one image to work one
                indices = torch.tensor(
                    [indices[i % len(indices)] for i in range(self.config["dataloader_kwargs/num_workers"])],
                    dtype=indices.dtype,
                )

            # Write new indices to the shared memory
            self.path_indices_worker[:] = indices
        else:
            sampler_indices = list(self.sampler)
            assert (
                len(sampler_indices) == len(self.sampler) == len(self.path_indices_worker)
            ), "Number of sampled indices must match the length of the sampler across epochs"

            self.path_indices_worker[:] = torch.tensor(sampler_indices)

    @abstractmethod
    def _add_shared_resources(self) -> None:
        """Each child class should implement this method to initialize the tensors which are needed in the batch."""
        pass

    def _add_image_index_shared(self) -> None:
        tensor = torch.empty(self.buffer_size, self.config["dataloader_kwargs/batch_size"], dtype=torch.int64)
        tensor = tensor.share_memory_()

        # The image_index tensor is only shared but not pinned since it is a CPU-only tensor

        self.shared_dict["image_index"] = tensor

    def _add_tensor_shared(self, name: str, dtype: torch.dtype, *sizes) -> None:
        tensor = torch.empty(self.buffer_size, self.config["dataloader_kwargs/batch_size"], *sizes, dtype=dtype)
        tensor = tensor.share_memory_()

        # Pin tensor
        cudart = torch.cuda.cudart()
        flags = 0  # cudaHostRegisterDefault: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8d5c17670f16ac4fc8fcb4181cb490c
        res = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), flags)
        assert res.value == 0 and res.name == "success", f"Cannot pin memory tensor {name}"
        assert tensor.is_pinned() and tensor.is_shared(), "Each tensor should be shared and pinned"

        self.shared_dict[name] = tensor

    def _get_worker_index(self) -> int:
        worker_info = get_worker_info()
        worker_index = worker_info.id if worker_info is not None else 0

        return worker_index

    @classmethod
    def batched_iteration(cls, paths: list[DataPath], config: Config) -> StreamDataLoader:
        """
        Helper method to iterate once over a list of paths. See DatasetImageBatch and DatasetImageStream classes for examples.

        Args:
            paths: Data paths to denoting the images to iterate over.
            config: Configuration options for the dataloader like batch size or number of workers.

        Returns: StreamDataLoader object which iterates in a batched-way over the data paths.
        """
        sampler = list(range(len(paths)))
        dataset = cls(paths, train=False, config=config, sampler=sampler)
        return StreamDataLoader(dataset, config)
