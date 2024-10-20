# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import math
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler

from htc.models.common.HTCDatasetStream import HTCDatasetStream
from htc.utils.LabelMapping import LabelMapping


class DatasetPatchStream(HTCDatasetStream):
    def __init__(self, *args, **kwargs):
        """
        This dataset generates random patches from images which are mainly not from the background.

        >>> from htc.models.common.StreamDataLoader import StreamDataLoader
        >>> from htc.tivita.DataPath import DataPath
        >>> from htc.utils.Config import Config
        >>> from htc.settings import settings
        >>> config = Config({
        ...     "input/epoch_size": 10,
        ...     "input/patch_size": [32, 32],
        ...     "input/n_channels": 100,
        ...     "dataloader_kwargs/num_workers": 1,
        ...     "dataloader_kwargs/batch_size": 5,
        ...     "label_mapping": "htc.settings_seg>label_mapping",
        ... })
        >>> paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        >>> dataset = DatasetPatchStream(paths, train=False, config=config)
        >>> dataloader = StreamDataLoader(dataset, config)
        >>> for sample in dataloader:
        ...     print(sample["features"].shape)
        ...     print(sample["labels"].shape)
        ...     break
        torch.Size([5, 32, 32, 100])
        torch.Size([5, 32, 32])

        It is also possible to iterate over the samples produced by this dataset directly (without the StreamDataLoader) yielding all patches generated per image:

        >>> config = Config({
        ...     "input/patch_size": [32, 32],
        ...     "input/n_channels": 100,
        ...     "dataloader_kwargs/num_workers": 1,
        ...     "dataloader_kwargs/batch_size": 1,
        ...     "label_mapping": "htc.settings_seg>label_mapping",
        ... })
        >>> dataset = DatasetPatchStream(paths, train=False, config=config, single_pass=True)
        >>> patches = [p for p in dataset.iter_samples()]
        >>> len(patches) > 0  # The exact number depends on the number of valid pixels
        True
        >>> patches[0]["features"].shape
        torch.Size([32, 32, 100])
        """
        super().__init__(*args, **kwargs)

        assert len(self.config["input/patch_size"]) == 2, "Patch size must be two-dimensional"
        assert self.config["input/patch_size"][0] == self.config["input/patch_size"][1], "Patch size must be quadratic"
        assert self.config["input/patch_size"][0] >= 32, "At least a patch size of 32 is required"

        self.patch_size = self.config["input/patch_size"][0]
        self.patch_size_half = self.patch_size // 2
        self.label_mapping = LabelMapping.from_config(self.config)

    def iter_samples(self, include_position: bool = False) -> Iterator[dict[str, torch.Tensor]]:
        for worker_index, path_index in self._iter_paths():
            path = self.paths[path_index]
            sample_img = self.read_experiment(path)
            sample_img = self.apply_transforms(sample_img)

            relevant_pixels = sample_img["valid_pixels"].clone()
            if self.config.get("input/background_undersampling", True) and "background" in self.label_mapping:
                # We don't want patches with too many background pixels
                relevant_pixels[sample_img["labels"] == self.label_mapping.name_to_index("background")] = False

            # Find the relevant pixel positions
            relevant_rows, relevant_cols = relevant_pixels.nonzero(as_tuple=True)
            assert len(relevant_rows) > 0, (
                f"Cannot extract even a single patch from the image {sample_img['image_name']}. Either because the"
                " image contains only background pixels or relevant pixels are only in a distance of"
                f" {self.patch_size_half} from the border (which cannot be used for patch sampling)"
            )

            # See config.schema for explanation for the different strategies
            if self.config.get("input/patch_sampling", "all_valid") == "all_valid":
                sampling = self._sample_all_valid
            else:
                sampling = self._sample_random

            for center_row, center_col in sampling(sample_img, relevant_rows, relevant_cols):
                labels = sample_img["labels"][
                    center_row - self.patch_size_half : center_row + self.patch_size_half,
                    center_col - self.patch_size_half : center_col + self.patch_size_half,
                ]
                assert labels.shape == (self.patch_size, self.patch_size), (
                    f"Each extracted patch must have a size of {self.patch_size} but the current patch has only a size"
                    f" of {labels.shape}"
                )

                valid_pixels = sample_img["valid_pixels"][
                    center_row - self.patch_size_half : center_row + self.patch_size_half,
                    center_col - self.patch_size_half : center_col + self.patch_size_half,
                ]
                assert valid_pixels.any(), f"At least one valid pixel must remain (image={path.image_name()})"

                features = sample_img["features"][
                    center_row - self.patch_size_half : center_row + self.patch_size_half,
                    center_col - self.patch_size_half : center_col + self.patch_size_half,
                ]

                sample = {
                    "features": features.refine_names("H", "W", "C"),
                    "labels": labels.refine_names("H", "W"),
                    "valid_pixels": valid_pixels.refine_names("H", "W"),
                    "image_index": path_index,
                    "worker_index": worker_index,
                }

                if include_position:
                    sample["center_row"] = center_row
                    sample["center_col"] = center_col

                yield sample

    def n_image_elements(self) -> int:
        return math.ceil(
            np.prod(self.paths[0].dataset_settings["spatial_shape"]) / np.prod(self.config["input/patch_size"])
        )

    def _sample_all_valid(
        self, sample_img: dict[str, torch.Tensor], relevant_rows: torch.Tensor, relevant_cols: torch.Tensor
    ) -> Iterator[tuple[int, int]]:
        image_height, image_width = sample_img["features"].shape[:2]

        while True:
            if len(relevant_rows) == 0:
                assert len(relevant_cols) == 0
                break

            patch_idx = next(iter(RandomSampler(relevant_rows)))

            # If the patch lands outside the image borders, we just shift it back inside the image
            center_row = relevant_rows[patch_idx]
            if center_row - self.patch_size_half < 0:
                center_row = self.patch_size_half
            if center_row + self.patch_size_half > image_height:
                center_row = image_height - self.patch_size_half

            center_col = relevant_cols[patch_idx]
            if center_col - self.patch_size_half < 0:
                center_col = self.patch_size_half
            if center_col + self.patch_size_half > image_width:
                center_col = image_width - self.patch_size_half

            # We do not want to sample the same pixels again
            used_rows = (center_row - self.patch_size_half <= relevant_rows) & (
                relevant_rows < center_row + self.patch_size_half
            )
            used_cols = (center_col - self.patch_size_half <= relevant_cols) & (
                relevant_cols < center_col + self.patch_size_half
            )

            remaining = ~(used_rows & used_cols)
            relevant_rows = relevant_rows[remaining]
            relevant_cols = relevant_cols[remaining]

            yield center_row, center_col

    def _sample_random(
        self, sample_img: dict[str, torch.Tensor], relevant_rows: torch.Tensor, relevant_cols: torch.Tensor
    ) -> Iterator[tuple[int, int]]:
        patch_sampling = self.config.get("input/patch_sampling", "all_valid")
        if patch_sampling == "proportional":
            # The number of sampled patches is proportional to the total number of relevant pixels
            n_patches = int(np.ceil(len(relevant_rows) / self.patch_size**2))
        elif patch_sampling == "uniform":
            # Sample as many patches as a grid tiling would provide
            n_patches = self.n_image_elements()
        else:
            raise ValueError(f"Invalid patch sampling value {patch_sampling}")

        image_height, image_width = sample_img["features"].shape[:2]

        i = 0
        for patch_idx in RandomSampler(relevant_rows):
            if i >= n_patches or i >= len(relevant_rows):
                break

            # If the patch lands outside the image borders, we just shift it back inside the image
            center_row = relevant_rows[patch_idx]
            if center_row - self.patch_size_half < 0:
                center_row = self.patch_size_half
            if center_row + self.patch_size_half > image_height:
                center_row = image_height - self.patch_size_half

            center_col = relevant_cols[patch_idx]
            if center_col - self.patch_size_half < 0:
                center_col = self.patch_size_half
            if center_col + self.patch_size_half > image_width:
                center_col = image_width - self.patch_size_half

            i += 1
            yield center_row, center_col

    def _add_shared_resources(self) -> None:
        self._add_image_index_shared()

        if not self.config["input/no_features"]:
            self._add_tensor_shared(
                "features", self.features_dtype, *self.config["input/patch_size"], self.config["input/n_channels"]
            )
        if not self.config["input/no_labels"]:
            if self.config["input/annotation_name"] and not self.config["input/merge_annotations"]:
                for name in self._possible_annotation_names():
                    self._add_tensor_shared(f"labels_{name}", torch.int64, *self.config["input/patch_size"])
                    self._add_tensor_shared(f"valid_pixels_{name}", torch.bool, *self.config["input/patch_size"])
            else:
                self._add_tensor_shared("labels", torch.int64, *self.config["input/patch_size"])
                self._add_tensor_shared("valid_pixels", torch.bool, *self.config["input/patch_size"])
