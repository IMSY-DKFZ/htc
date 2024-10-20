# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from htc.cpp import spxs_predictions
from htc.models.common.HTCDatasetStream import HTCDatasetStream
from htc.models.common.utils import get_n_classes
from htc.models.image.DatasetImage import DatasetImage


class DatasetSuperpixelStream(HTCDatasetStream):
    def iter_samples(self) -> Iterator[dict[str, torch.Tensor]]:
        for worker_index, path_index in self._iter_paths():
            # Explicitly load via DatasetImage since we need superpixels
            path = self.paths[path_index]
            sample_img = DatasetImage([path], train=self.train, config=self.config)[0]

            # Calculate the mode value per superpixel, i.e. find the label which occurs most often in the superpixel
            _, spxs_label_counts = spxs_predictions(
                sample_img["spxs"],
                sample_img["labels"],
                sample_img["valid_pixels"],
                n_classes=get_n_classes(self.config),
            )
            # spx_label_counts.shape = (n_superpixels, n_labels) counts how often each label occurred in the superpixel
            spxs_labels = spxs_label_counts.argmax(dim=1)

            for spx_id in self._sample_pixel_indices(
                spxs_labels
            ):  # Iterate over the superpixels (oversampling may be done via superpixel labels)
                # Part of image which belongs to the superpixel
                img_spx_mask = sample_img["spxs"] == spx_id
                spx_indices_rows, spx_indices_cols = torch.nonzero(img_spx_mask, as_tuple=True)
                min_row, max_row = spx_indices_rows.min(), spx_indices_rows.max()
                min_col, max_col = spx_indices_cols.min(), spx_indices_cols.max()

                # Extract the relevant images for the superpixel
                spx_features = sample_img[
                    "features"
                ][
                    min_row : max_row + 1, min_col : max_col + 1, :
                ].clone()  # The clone here is important as we later set some of the features to 0 (and this would also affect the original image without a clone)
                spx_valid_pixels = sample_img["valid_pixels"][min_row : max_row + 1, min_col : max_col + 1].clone()

                # Everything which does not belong to the superpixel will be set to zero
                spx_mask = img_spx_mask[min_row : max_row + 1, min_col : max_col + 1]
                spx_features[~spx_mask, :] = 0
                spx_valid_pixels[~spx_mask] = False

                if not torch.any(spx_valid_pixels):
                    # No valid pixels in this superpixel --> skip it
                    continue

                # Weak label represents the label distribution of the superpixel
                spx_weak_label = spxs_label_counts[spx_id] / spxs_label_counts[spx_id].sum()

                # Resized superpixel values as features
                features = spx_features.permute(2, 0, 1).unsqueeze(dim=0)
                features = F.interpolate(
                    features.float(), size=self.config["input/resize_shape"], mode="bilinear", align_corners=False
                )
                features = features.squeeze(dim=0).half()

                yield {
                    "features": features.refine_names("C", "H", "W"),
                    "weak_labels": spx_weak_label.refine_names("N"),
                    "image_index": path_index,
                    "worker_index": worker_index,
                }

    def n_image_elements(self) -> int:
        return self.config["input/superpixels/n_segments"]

    def _add_shared_resources(self) -> None:
        self._add_image_index_shared()
        self._add_tensor_shared(
            "features", self.features_dtype, self.config["input/n_channels"], *self.config["input/resize_shape"]
        )
        self._add_tensor_shared("weak_labels", torch.float32, get_n_classes(self.config))
