# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F

from htc.models.image.DatasetImage import DatasetImage


class DatasetSuperpixelImage(DatasetImage):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_img = super().__getitem__(index)

        features = []
        spxs_sizes = []
        spxs_indices_rows = []
        spxs_indices_cols = []
        for spx_id in sample_img["spxs"].unique():
            img_spx_mask = sample_img["spxs"] == spx_id

            # Extract superpixel in image
            spx_indices_rows, spx_indices_cols = torch.nonzero(img_spx_mask, as_tuple=True)
            assert spx_indices_rows.shape[0] == spx_indices_cols.shape[0], (
                "row and col index must be available for pixels"
            )
            spxs_sizes.append(spx_indices_rows.shape[0])

            spxs_indices_rows.append(spx_indices_rows)
            spxs_indices_cols.append(spx_indices_cols)
            min_row, max_row = spx_indices_rows.min(), spx_indices_rows.max()
            min_col, max_col = spx_indices_cols.min(), spx_indices_cols.max()

            # Calculate resized feature vector
            x_image = sample_img["features"][min_row : max_row + 1, min_col : max_col + 1, :].clone()
            spx_mask = img_spx_mask[min_row : max_row + 1, min_col : max_col + 1]
            x_image[~spx_mask, :] = 0
            x_image = x_image.permute(2, 0, 1).unsqueeze(dim=0)
            x_image = F.interpolate(
                x_image.float(), size=self.config["input/resize_shape"], mode="bilinear", align_corners=False
            ).squeeze(dim=0)

            features.append(x_image)

        # Only include what is absolutely necessary in the sample
        sample = {
            "image_size": torch.tensor(sample_img["features"].shape[:2]),
            "image_name": sample_img["image_name"],
            "image_name_annotations": sample_img["image_name_annotations"],
            "features": torch.stack(features),
            "spxs_sizes": torch.tensor(spxs_sizes),
            # We already concatenate the ids since we make only full image assignments later
            "spxs_indices_rows": torch.cat(spxs_indices_rows),
            "spxs_indices_cols": torch.cat(spxs_indices_cols),
        }

        if "labels" in sample_img:
            sample["labels"] = sample_img["labels"]
            sample["valid_pixels"] = sample_img["valid_pixels"]

        return sample
