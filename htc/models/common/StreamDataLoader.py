# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import math
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from htc.cpp import nunique
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.torch_helpers import cpu_only_tensor
from htc.models.common.utils import adjust_epoch_size
from htc.utils.Config import Config


class StreamDataLoader:
    def __init__(self, dataset: HTCDataset, config: Config = None, **dataloader_kwargs):
        """
        This dataloader should be used with the stream datasets. It ensures that each worker contributes to a single batch (to increase the randomness).

        https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

        Args:
            dataset: The stream or batch dataset to iterate over.
            config: The configuration object. If None, the config from the dataset will be used.
            dataloader_kwargs: Additional keyword arguments passed to the DataLoader.
        """
        self.dataset = dataset
        self.config = config if config is not None else self.dataset.config

        try:
            dataset_length = len(self.dataset)
        except NotImplementedError:
            dataset_length = None

        if self.dataset.sampler is None or dataset_length is None:
            # Number of batches are solely defined by the epoch size
            adjust_epoch_size(self.config)

            assert self.config["input/epoch_size"] % self.config["dataloader_kwargs/batch_size"] == 0, (
                "The epoch size must be divisible by the batch size"
            )
            self.n_batches = self.config["input/epoch_size"] // self.config["dataloader_kwargs/batch_size"]
        else:
            # If there is a sampler, we want to make sure that we return all images defined by the sampler. The last batch may be smaller in this case
            self.n_batches = math.ceil(dataset_length / self.config["dataloader_kwargs/batch_size"])

        # With index-based datasets, every worker operates on their own batch
        self.single_mode = not isinstance(self.dataset, IterableDataset)
        if not self.single_mode:
            assert self.config["dataloader_kwargs/batch_size"] % self.config["dataloader_kwargs/num_workers"] == 0, (
                f"The batch size ({self.config['dataloader_kwargs/batch_size']}) must be divisible by the number of"
                f" workers ({self.config['dataloader_kwargs/num_workers']})"
            )

        self.dataset.init_shared()
        self.keys = [k for k in self.dataset.shared_dict.keys() if k != "worker_index"]

        # Setup our own dataloader
        forbidden_kwargs = [
            "batch_size",  # We always handle batches explicitly
            "persistent_workers",  # We do not want to re-create the shared buffers in every epoch
            "sampler",  # We either use the batch sampler or no sampler at all
        ]
        assert all(k not in dataloader_kwargs for k in forbidden_kwargs), (
            f"The following keyword arguments are not allow to be set: {forbidden_kwargs}"
        )

        loader_kwargs = copy.deepcopy(self.config["dataloader_kwargs"])
        loader_kwargs |= dataloader_kwargs

        # Removed in case they were in the config
        for k in forbidden_kwargs:
            loader_kwargs.pop(k, None)

        sampler = list(range(self.n_batches)) if self.single_mode else None
        self.dataloader = DataLoader(
            self.dataset, sampler=sampler, persistent_workers=True, batch_size=None, **loader_kwargs
        )

    def __iter__(self) -> Iterator[dict]:
        self.dataset.shuffle_paths()  # Shuffle the data paths in the beginning (for randomness across the images)
        loader_it = iter(self.dataloader)
        batches_provided = 0

        while True:
            # We need to wait until the GPU finished with the previous batch so that the workers can safely overwrite the
            # corresponding entry in the buffer. If we did not make this synchronization, it could happen, because all CUDA
            # operations are executed asynchronously, that one of the workers alters the tensor before the data is copied
            # to the GPU so that the GPU works with wrong data. This could e.g. lead to the following error:
            # /pytorch/aten/src/THCUNN/ClassNLLCriterion.cu:108: cunn_ClassNLLCriterion_updateOutput_kernel: block: [0,0,0], thread: [9,0,0] Assertion `t >= 0 && t < n_classes` failed.
            # ...
            # RuntimeError: CUDA error: device-side assert triggered
            torch.cuda.synchronize()

            if self.single_mode:
                yield self._next_batch_single(loader_it)
            else:
                yield self._next_batch_collaborative(loader_it)

            batches_provided += 1
            if batches_provided >= self.n_batches:
                return

    def __len__(self) -> int:
        """
        Returns: The number of batches which will be returned by this dataloader.
        """
        return self.n_batches

    def _next_batch_single(self, loader_it: Iterator) -> dict[str, torch.Tensor]:
        """Every worker works on their own batch."""

        ret = next(loader_it)
        buffer_index = ret["buffer_index"]
        batch_size = ret["batch_size"] if "batch_size" in ret else None

        worker_indices = self.dataset.shared_dict["worker_index"][buffer_index]
        if batch_size is not None:
            worker_indices = worker_indices[:batch_size]
        assert nunique(worker_indices) == 1, (
            f"Only one worker should have contributed to the batch ({worker_indices.unique() = })"
        )

        batch = {}
        for key in self.keys:
            tensor = self.dataset.shared_dict[key][buffer_index]
            if batch_size is not None:
                tensor = tensor[:batch_size]

            if key == "image_index":
                batch[key] = cpu_only_tensor(tensor.clone())
            else:
                batch[key] = tensor.cuda(non_blocking=True)

        if not self.dataset.train and "image_name" not in batch:
            assert "image_index" in batch
            batch["image_name"] = [self.dataset.paths[idx].image_name() for idx in batch["image_index"]]
            batch["image_name_annotations"] = [
                self.dataset.paths[idx].image_name_annotations() for idx in batch["image_index"]
            ]

        return batch

    def _next_batch_collaborative(self, loader_it: Iterator) -> dict[str, torch.Tensor]:
        """All workers operate collaboratively on one batch."""

        # Collect the batch parts from the workers
        results = {}
        for i in range(self.dataloader.num_workers):
            try:
                results[i] = next(loader_it)
            except StopIteration:
                pass

        is_batch_part = any(r["partly_filled"] for r in results.values()) or len(results) < self.dataloader.num_workers
        if is_batch_part:
            used_indices = []
            for r in results.values():
                used_indices.append(torch.arange(r["start_index"], r["end_index"], dtype=torch.int64))
            used_indices = torch.cat(used_indices)

        buffer_index = next(iter(results.values()))[
            "buffer_index"
        ]  # Index of the current buffer (same for all workers)
        assert all(r["buffer_index"] == buffer_index for r in results.values()), (
            "Each worker must return the same buffer index"
        )

        worker_indices = self.dataset.shared_dict["worker_index"][
            buffer_index
        ]  # Workers which contributed to the batch
        if is_batch_part:
            worker_indices = worker_indices[used_indices]
        assert worker_indices.unique().tolist() == list(results.keys()), (
            f"Every worker should contribute to the batch ({worker_indices.unique() = }, {list(results.keys()) = })"
        )

        batch = {}
        for key in self.keys:
            tensor = self.dataset.shared_dict[key][buffer_index]
            if is_batch_part:
                tensor = tensor[used_indices]

            if key == "image_index":
                batch[key] = cpu_only_tensor(tensor.clone())
            else:
                batch[key] = tensor.cuda(non_blocking=True)

        if not self.dataset.train and "image_name" not in batch:
            assert "image_index" in batch
            batch["image_name"] = [self.dataset.paths[idx].image_name() for idx in batch["image_index"]]
            batch["image_name_annotations"] = [
                self.dataset.paths[idx].image_name_annotations() for idx in batch["image_index"]
            ]

        return batch
