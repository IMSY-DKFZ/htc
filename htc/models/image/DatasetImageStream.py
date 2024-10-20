# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator

import torch

from htc.models.common.HTCDatasetStream import HTCDatasetStream
from htc.models.image.DatasetImage import DatasetImage


class DatasetImageStream(DatasetImage, HTCDatasetStream):
    """
    This class is similar to DatasetImage but is supposed to be used together with the StreamDataLoader. It uses a ring buffer with shared and pinned memory. This makes this dataset significantly faster than DatasetImage.

    Compared to DatasetImageBatch, worker of this class contribute collaboratively on a batch. This means that a batch is only finished when every worker finished loading its images (similar to all the other stream-based datasets). The memory consumption of the shared buffer is independent of the number of workers but the batch size must be divisible by the number of workers.

    It is also possible to use this dataset for a batched iteration over all paths:

    >>> from htc.tivita.DataPath import DataPath
    >>> from htc.utils.Config import Config
    >>> paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
    >>> config = Config({
    ...     "input/n_channels": 100,
    ...     "dataloader_kwargs/num_workers": 1,
    ...     "dataloader_kwargs/batch_size": 1,
    ... })
    >>> dataloader = DatasetImageStream.batched_iteration(paths, config)
    >>> for batch in dataloader:
    ...     print(batch["features"].shape)
    ...     print(batch["labels"].shape)
    torch.Size([1, 480, 640, 100])
    torch.Size([1, 480, 640])
    """

    def iter_samples(self) -> Iterator[dict[str, torch.Tensor]]:
        for worker_index, path_index in self._iter_paths():
            start_pointers = self._get_start_pointers(self.buffer_index, self.image_index)
            sample = self.__getitem__(path_index, start_pointers=start_pointers)
            del sample["image_name"]
            del sample["image_name_annotations"]

            if "features" in sample:
                if isinstance(sample["features"], torch.Tensor):
                    sample["features"] = sample["features"].refine_names("H", "W", "C")
                else:
                    # HTCDatasetStream needs to know that features was already filled by a pointer
                    sample["features"] = "pointer"
            if "labels" in sample:
                sample["labels"] = sample["labels"].refine_names("H", "W")
            if "valid_pixels" in sample:
                sample["valid_pixels"] = sample["valid_pixels"].refine_names("H", "W")
            sample["worker_index"] = worker_index
            sample["image_index"] = path_index

            yield sample

    def n_image_elements(self) -> int:
        return 1

    def __len__(self) -> int:
        if self.sampler is not None:
            return len(self.sampler)
        else:
            return self.config["input/epoch_size"]
