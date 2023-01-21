# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator

import numpy as np
import torch

from htc.models.common.HTCDatasetStream import HTCDatasetStream


class DatasetPixelStream(HTCDatasetStream):
    """
    This dataset reads random pixels from the specified images.

    >>> from htc.models.common.StreamDataLoader import StreamDataLoader
    >>> from htc.utils.Config import Config
    >>> from htc.tivita.DataPath import DataPath
    >>> config = Config({
    ...     'input/epoch_size': 10,
    ...     'input/n_channels': 100,
    ...     'dataloader_kwargs/num_workers': 1,
    ...     'dataloader_kwargs/batch_size': 5
    ... })
    >>> paths = [DataPath.from_image_name('P043#2019_12_20_12_38_35')]
    >>> dataset = DatasetPixelStream(paths, train=False, config=config)
    >>> dataloader = StreamDataLoader(dataset, config)
    >>> for sample in dataloader:
    ...     print(sample['features'].shape)
    ...     print(sample['labels'].shape)
    ...     break
    torch.Size([5, 100])
    torch.Size([5])
    """

    def iter_samples(self) -> Iterator[dict[str, torch.Tensor]]:
        for worker_index, path in self._iter_paths():
            sample = self.read_experiment(path)

            # # in case there are no valid pixels in this image.
            # # this can happen if the label mapping has been modified for a data specification file, in such a way that there are some images left without any valid pixels
            # if sample['valid_pixels'].sum() == 0:
            #     settings.log.warning(f'Received a batch with no valid pixels (image info: {path.image_name()})')
            #     continue

            sample = self.apply_transforms(sample)

            features = sample["features"].reshape(-1, sample["features"].shape[2])
            labels = sample["labels"].reshape(-1)

            pixel_sampling = self.config.get("input/pixel_sampling", "proportional")
            if pixel_sampling == "proportional":
                # Sample according to the number of valid pixels
                n_image_pixels = None
            elif pixel_sampling == "uniform":
                # Sample as many pixels as the image contains to avoid performance issues
                n_image_pixels = features.size(0)
            else:
                raise ValueError(f"Invalid pixel sampling value {pixel_sampling}")

            # Relevant pixels for the classification
            valid_pixels = sample["valid_pixels"].reshape(-1)

            if not self.config["input/all_pixels"]:
                # Only return valid pixels
                features = features[valid_pixels, :]
                labels = labels[valid_pixels]

            if self.config["input/specs_threshold"]:
                specs = sample["specs"].reshape(-1)
                if not self.config["input/all_pixels"]:
                    specs = specs[valid_pixels]

            # if there are no valid_pixels then skip this path
            if labels.size(0) == 0:
                continue

            indices = torch.tensor(list(self._sample_pixel_indices(labels, n_samples=n_image_pixels)))
            index_split = torch.split(indices, self.batch_part_size)[
                :-1
            ]  # The last split is ignored since it only contains the residual and its smaller
            worker_index_tensor = torch.ones(self.batch_part_size, dtype=torch.int64) * worker_index
            image_name_tensor = torch.ones(self.batch_part_size, dtype=torch.int64) * self.image_names.index(
                path.image_name()
            )
            batches_per_image = 0

            for split in index_split:
                sample = {
                    "features": features[split, :].refine_names("B", "C"),
                    "labels": labels[split].refine_names("B"),
                    "worker_index": worker_index_tensor.refine_names("B"),
                    "image_index": image_name_tensor.refine_names("B"),
                }

                if self.config["input/all_pixels"]:
                    sample["valid_pixels"] = valid_pixels[split].refine_names("B")

                if self.config["input/specs_threshold"]:
                    sample["specs"] = specs[split].refine_names("B")

                batches_per_image += 1

                yield sample

                if self.config.get("input/batches_per_image", np.inf) < batches_per_image:
                    break

    def n_image_elements(self) -> int:
        return self.paths[0].dataset_settings.pixels_image()

    def _add_shared_resources(self) -> None:
        self._add_image_index_shared()

        if not self.config["input/no_features"]:
            self._add_tensor_shared("features", self.features_dtype, self.config["input/n_channels"])
        if not self.config["input/no_labels"]:
            if self.config["input/annotation_name"] and not self.config["input/merge_annotations"]:
                for name in self._possible_annotation_names():
                    self._add_tensor_shared(f"labels_{name}", torch.int64)
                    if self.config["input/all_pixels"]:
                        self._add_tensor_shared(f"valid_pixels_{name}", torch.bool)
            else:
                self._add_tensor_shared("labels", torch.int64)
                if self.config["input/all_pixels"]:
                    self._add_tensor_shared("valid_pixels", torch.bool)

        if self.config["input/specs_threshold"]:
            self._add_tensor_shared("specs", torch.int64)
