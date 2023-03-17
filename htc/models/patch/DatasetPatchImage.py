# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from htc.models.common.torch_helpers import pad_tensors
from htc.models.image.DatasetImage import DatasetImage


class DatasetPatchImage(DatasetImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(self.config["input/patch_size"]) == 2, "Only 2D patches are supported"
        assert (
            self.config["input/patch_size"][0] == self.config["input/patch_size"][1]
        ), "Only square patches are supported"
        self.patch_size = self.config["input/patch_size"][0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)

        n_channels = sample["features"].shape[-1]

        # Expand image so that the patch blocks fit easily
        features = pad_tensors([sample["features"]], size_multiple=self.patch_size)[0]
        sample["image_size"] = torch.tensor(sample["features"].shape[:2])
        sample["image_size_expanded"] = torch.tensor(features.shape[:2])

        # Split the image into patches
        patch_features = features.unfold(0, self.patch_size, self.patch_size).unfold(
            1, self.patch_size, self.patch_size
        )
        patch_features = patch_features.reshape(-1, n_channels, self.patch_size, self.patch_size)  # [300, 100, 32, 32]
        patch_features = patch_features.permute(0, 2, 3, 1)  # [300, 32, 32, 100]
        sample["features"] = patch_features

        return sample

    def reshape_img(self, tensor: torch.Tensor, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reshapes patches of an image (e.g. predictions) back to the original image. In case the image was expanded in the first place, the original image size is restored.

        Args:
            tensor: Patches of an image (n_patches, n_channels, patch_size, patch_size) or (n_patches, patch_size, patch_size).
            sample: The original sample from this dataset which contains information about the image dimension.

        Returns: Full image based on the patches (n_height, n_width, n_channels) or (n_height, n_width).
        """
        if len(sample["image_size_expanded"].shape) == 1:
            image_size = sample["image_size"]
            image_size_expanded = sample["image_size_expanded"]
        else:
            image_size = sample["image_size"][0]
            image_size_expanded = sample["image_size_expanded"][0]

        image_size = tuple(image_size)
        image_size_expanded = tuple(image_size_expanded)

        original_type = tensor.dtype

        # Reshape block of patches back to the original image size
        # tensor.shape = [300, 32, 32, 100]

        if len(tensor.shape) == 4:
            # Channel must be the second dimension for correct folding
            tensor = tensor.permute(0, 3, 1, 2)  # [300, 100, 32, 32]

        tensor = tensor.reshape(tensor.shape[0], -1).unsqueeze(dim=0).permute(0, 2, 1)  # [1, 102400, 300]
        tensor = nn.Fold(
            output_size=image_size_expanded,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )(
            tensor.type(torch.float32)
        )  # [1, 100, 480, 640]
        tensor = tensor.type(original_type).squeeze()

        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)  # [480, 640, 100]

        if image_size != image_size_expanded:
            # Remove the expanded parts
            tensor = tensor[: image_size[0], : image_size[1]]

        return tensor
