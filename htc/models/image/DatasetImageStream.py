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
    >>> paths = [DataPath.from_image_name('P043#2019_12_20_12_38_35')]
    >>> config = Config({
    ...     'input/n_channels': 100,
    ...     'dataloader_kwargs/num_workers': 1,
    ...     'dataloader_kwargs/batch_size': 1,
    ... })
    >>> dataloader = DatasetImageStream.batched_iteration(paths, config)
    >>> for sample in dataloader:
    ...     print(sample['features'].shape)
    ...     print(sample['labels'].shape)
    torch.Size([1, 480, 640, 100])
    torch.Size([1, 480, 640])
    """

    def iter_samples(self) -> Iterator[dict[str, torch.Tensor]]:
        for worker_index, path in self._iter_paths():
            image_name = path.image_name()
            sample = self.from_image_name(image_name)
            del sample["image_name"]

            if "features" in sample:
                sample["features"] = sample["features"].refine_names("H", "W", "C")
            if "labels" in sample:
                sample["labels"] = sample["labels"].refine_names("H", "W")
            if "valid_pixels" in sample:
                sample["valid_pixels"] = sample["valid_pixels"].refine_names("H", "W")
            sample["worker_index"] = worker_index
            sample["image_index"]: self.image_names.index(path.image_name())

            yield sample

    def _add_shared_resources(self) -> None:
        self._add_image_index_shared()
        spatial_shape = self.paths[0].dataset_settings["spatial_shape"]

        if not self.config["input/no_features"]:
            self._add_tensor_shared("features", self.features_dtype, *spatial_shape, self.config["input/n_channels"])
        if not self.config["input/no_labels"]:
            if self.config["input/annotation_name"] and not self.config["input/merge_annotations"]:
                for name in self._possible_annotation_names():
                    self._add_tensor_shared(f"labels_{name}", torch.int64, *spatial_shape)
                    self._add_tensor_shared(f"valid_pixels_{name}", torch.bool, *spatial_shape)
            else:
                self._add_tensor_shared("labels", torch.int64, *spatial_shape)
                self._add_tensor_shared("valid_pixels", torch.bool, *spatial_shape)

        if self.config["input/specs_threshold"]:
            self._add_tensor_shared("specs", torch.int64, *spatial_shape)
        if self.config["input/superpixels"]:
            self._add_tensor_shared("spxs", torch.int64, *spatial_shape)

        for domain in self.target_domains:
            self._add_tensor_shared(domain, torch.int64)

    def n_image_elements(self) -> int:
        return 1

    def __len__(self) -> int:
        if self.sampler is not None:
            return len(self.sampler)
        else:
            return self.config["input/epoch_size"]
