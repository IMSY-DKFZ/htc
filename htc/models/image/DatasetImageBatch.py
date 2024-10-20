# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.common.SharedMemoryDatasetMixin import SharedMemoryDatasetMixin
from htc.models.image.DatasetImage import DatasetImage


class DatasetImageBatch(SharedMemoryDatasetMixin, DatasetImage):
    def __init__(self, *args, **kwargs):
        """
        This class is similar to DatasetImage but is supposed to be used together with the StreamDataLoader. It uses a fixed buffer with shared and pinned memory. This makes this dataset significantly faster than DatasetImage.

        Compared to DatasetImageStream, one worker of this class is responsible for loading one batch at a time. There is no dependence between the workers but every worker has its own fixed buffer which leads to high memory consumption when many workers are used.

        It is also possible to use this dataset for a batched iteration over all paths:

        >>> from htc.tivita.DataPath import DataPath
        >>> from htc.utils.Config import Config
        >>> paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        >>> config = Config({
        ...     "input/n_channels": 100,
        ...     "dataloader_kwargs/num_workers": 1,
        ...     "dataloader_kwargs/batch_size": 1,
        ... })
        >>> dataloader = DatasetImageBatch.batched_iteration(paths, config)
        >>> for batch in dataloader:
        ...     print(batch["features"].shape)
        ...     print(batch["labels"].shape)
        torch.Size([1, 480, 640, 100])
        torch.Size([1, 480, 640])
        """
        super().__init__(*args, **kwargs)

        self.worker_buffer_size = self.prefetch_factor + 1
        self.worker_buffer_index = 0

    def __getitem__(self, batch_index: int) -> dict[str, int]:
        worker_index = self._get_worker_index()
        buffer_index = worker_index * self.worker_buffer_size + self.worker_buffer_index
        assert buffer_index < self.buffer_size, "Invalid buffer location"

        # Load all the images which should be part of this batch
        for i in range(self.config["dataloader_kwargs/batch_size"]):
            sampler_index = batch_index * self.config["dataloader_kwargs/batch_size"] + i
            if sampler_index >= len(self.path_indices_worker):
                return {
                    "buffer_index": buffer_index,
                    "batch_size": i,
                }

            start_pointers = self._get_start_pointers(buffer_index, i)
            image_index = self.path_indices_worker[sampler_index]
            sample = super().__getitem__(image_index, start_pointers=start_pointers)
            sample["worker_index"] = worker_index
            del sample["image_name"]
            del sample["image_name_annotations"]

            for key, tensor in sample.items():
                if key in self.pointer_keys and type(tensor) == int:
                    # Only blosc preprocessed files can be directly loaded into the pinned memory buffer, so not every pointer is used
                    assert tensor == start_pointers[key], f"The pointer for the key {key} does not match"
                else:
                    self.shared_dict[key][buffer_index, i] = tensor
            del sample

        self.worker_buffer_index = (self.worker_buffer_index + 1) % self.worker_buffer_size

        return {"buffer_index": buffer_index}

    def __len__(self) -> int:
        return len(self.sampler)

    @property
    def buffer_size(self) -> int:
        return self.worker_buffer_size * self.config["dataloader_kwargs/num_workers"]
