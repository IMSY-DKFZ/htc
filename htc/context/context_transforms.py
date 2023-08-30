# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from typing import Union

import torch
import torch.nn.functional as F
from kornia.augmentation import random_generator
from kornia.geometry.bbox import bbox_generator, bbox_to_mask

from htc.models.common.transforms import HTCTransformation
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class ContextTransformation(HTCTransformation):
    # Global cache of the cloth sample since it is always the same and we only want to load it once
    _filling_sample = None

    def __init__(
        self,
        fill_value: str,
        target_label: Union[int, str] = None,
        p: float = 1,
        config: Config = None,
        **kwargs,
    ):
        self.fill_value = fill_value
        self.target_label = target_label
        self.p = p
        self.config = config
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"

    def _apply_selection(self, selection: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.p != 1:
            # torch.rand gives numbers in [0;1[
            apply_transform = torch.rand(selection.size(0)) < self.p

            # Unselect the images in the batch where we don't want to apply the transformation
            selection[~apply_transform, ...] = False

        # Mark the pixels which will be altered as invalid
        batch["valid_pixels"][selection] = False
        batch["labels"][selection] = settings.label_index_thresh

        if "features" in batch:
            if self.fill_value == "0":
                # Set the feature of a given label to 0
                batch["features"][selection] = 0
            elif self.fill_value == "cloth":
                # Set the features of the batch to the features of the filling sample
                batch["features"][selection] = self._load_filling_sample(batch).expand_as(batch["features"])[selection]
            elif self.fill_value == "random_uniform":
                # Set the features to L1-normalized random numbers of a uniform distribution
                rand_tensor = torch.rand(
                    (selection.count_nonzero(), batch["features"].size(-1)),
                    dtype=batch["features"].dtype,
                    device=batch["features"].device,
                )
                rand_tensor = rand_tensor / torch.linalg.norm(rand_tensor, ord=1, dim=-1, keepdim=True)
                rand_tensor.nan_to_num_()
                batch["features"][selection] = rand_tensor
            else:
                raise ValueError(f"Invalid filling value: {self.fill_value}")

        return batch

    def _load_filling_sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if ContextTransformation._filling_sample is None:
            # Lazy load the filling sample (only once and with the same dtype and device as the batch)
            if self.fill_value == "cloth":
                assert self.config is not None, "Config is required to fill with a data sample"

                config_loading = copy.copy(self.config)
                config_loading["input/no_labels"] = True
                config_loading["input/test_time_transforms_cpu"] = None

                # DataPath.from_image_name("ref#2020_07_23_hyperspectral_MIC_organ_database#2020_02_21_04_18_06"), DataPath.from_image_name("ref#2020_07_23_hyperspectral_MIC_organ_database#2020_02_20_18_32_31")]
                paths = [
                    DataPath.from_image_name("ref#2020_07_23_hyperspectral_MIC_organ_database#2020_02_20_18_29_29")
                ]
                ContextTransformation._filling_sample = DatasetImage(paths, train=False, config=config_loading)[0]
                ContextTransformation._filling_sample = ContextTransformation._filling_sample["features"].to(
                    device=batch["features"].device, dtype=batch["features"].dtype
                )

        return ContextTransformation._filling_sample


class OrganRemoval(ContextTransformation):
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.target_label is not None, "target_label must not be None for this transformation"

        if self.target_label == "random":
            selection = []
            for b in range(len(batch["labels"])):
                unique_labels = batch["labels"][b][batch["valid_pixels"][b]].unique()
                random_label = unique_labels[torch.randint(low=0, high=len(unique_labels), size=(1,))]

                selection.append(batch["labels"][b] == random_label)
            selection = torch.stack(selection)
        else:
            selection = batch["labels"] == self.target_label

        return self._apply_selection(selection, batch)

    def __repr__(self) -> str:
        return f"OrganRemoval(fill_value={self.fill_value}, target_label={self.target_label}, p={self.p})"

    def is_applied(self, batch: dict[str, torch.Tensor]) -> bool:
        return self.target_label not in batch["labels"]


class OrganIsolation(ContextTransformation):
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.target_label is not None, "target_label must not be None for this transformation"

        if self.target_label == "random":
            selection = []
            for b in range(len(batch["labels"])):
                unique_labels = batch["labels"][b][batch["valid_pixels"][b]].unique()
                random_label = unique_labels[torch.randint(low=0, high=len(unique_labels), size=(1,))]

                selection.append(batch["labels"][b] != random_label)
            selection = torch.stack(selection)
        else:
            selection = batch["labels"] != self.target_label

        return self._apply_selection(selection, batch)

    def __repr__(self) -> str:
        return f"OrganIsolation(fill_value={self.fill_value}, target_label={self.target_label}, p={self.p})"

    def is_applied(self, batch: dict[str, torch.Tensor]) -> bool:
        return torch.all(batch["labels"][batch["valid_pixels"]] == self.target_label)


class ValidPixelsOnly(ContextTransformation):
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = ~batch["valid_pixels"]
        return self._apply_selection(selection, batch)

    def __repr__(self) -> str:
        return f"ValidPixelsOnly(fill_value={self.fill_value}, p={self.p})"

    def is_applied(self, batch: dict[str, torch.Tensor]) -> bool:
        return torch.all(batch["labels"][~batch["valid_pixels"]] == settings.label_index_thresh)


class HideAndSeek(ContextTransformation):
    def __init__(
        self,
        proportion: Union[float, tuple[float, float]],
        patch_size: list[tuple[int, int]],
        **kwargs,
    ):
        """
        Hide-and-Seek transformation for semantic segmentation. This transformation basically generates a grid for the image and randomly blacks out certain parts.

        Args:
            proportion: The proportion of grid cells which should be blacked out. Either a float to remove a fixed proportion in every image where this transformation will be applied or a range [min;max[ to sample a proportional value randomly from a uniform distribution for each image.
            patch_size: List of patch sizes (height, width) of the grid cells. This implicitly defines the grid size. For each image, a patch size will be randomly selected.
        """
        super().__init__(**kwargs)
        self.proportion = proportion
        self.patch_size = patch_size

        if type(self.proportion) == float:
            assert 0 <= self.proportion <= 1, f"The proportion value must be in the range [0;1], not {self.proportion}"
        else:
            assert all(
                0 <= p <= 1 for p in self.proportion
            ), f"Every proportion value must be in the range [0;1], not {self.proportion}"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = self._grid_hiding(batch)
        return self._apply_selection(selection, batch)

    def __repr__(self) -> str:
        return (
            f"HideAndSeek(fill_value={self.fill_value}, p={self.p}, proportion={self.proportion},"
            f" patch_size={self.patch_size})"
        )

    def _grid_hiding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, height, width = batch["features"].shape[:3]

        # We need to loop over the batch size since we want to have a different grid per image and a different number of removed patches per image
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros((1, height, width), dtype=torch.bool, device=batch["features"].device)

            # Select a random patch size for this image
            selected_patch_size = torch.randint(0, len(self.patch_size), (1,))
            patch_height = self.patch_size[selected_patch_size][0]
            patch_width = self.patch_size[selected_patch_size][1]

            # BHW
            mask = mask.permute(1, 2, 0)  # BHW --> HWB
            patch_features = (
                # Unfold H dim
                mask.unfold(dimension=0, size=patch_height, step=patch_height)
                # Unfold W dim
                .unfold(dimension=1, size=patch_width, step=patch_width)
                # BLhw (L = number of patches; h, w = patch dimensions)
                .reshape(1, -1, patch_height, patch_width)
            )

            # Select a random number of patches to hide
            n_patches = patch_features.size(1)
            if type(self.proportion) == float:
                n_remove = int(self.proportion * n_patches)
            else:
                proportion = torch.empty(1, device=batch["features"].device).uniform_(
                    self.proportion[0], self.proportion[1]
                )
                n_remove = int(proportion * n_patches)

            hide_patches = torch.randint(0, patch_features.size(1), (n_remove,), device=batch["features"].device)
            patch_features[0, hide_patches] = True

            # B*Lhw --> BLX --> BXL (X = h*w)
            mask = patch_features.reshape(1, -1, patch_height * patch_width).permute(0, 2, 1)

            # Fold back to image size
            mask = F.fold(
                mask.type(torch.float32),
                output_size=(height, width),
                kernel_size=(patch_height, patch_width),
                stride=(patch_height, patch_width),
            )
            mask = mask.type(torch.bool).squeeze(dim=1).squeeze(dim=0)  # BCHW --> HW

            masks.append(mask)

        return torch.stack(masks)


class RectangleMixin:
    def __init__(
        self,
        scale: Union[torch.Tensor, tuple[float, float]] = (0.02, 0.33),
        ratio: Union[torch.Tensor, tuple[float, float]] = (0.3, 3.3),
        **kwargs,
    ):
        """
        Mixin which can be added to a class to get random rectangles for a transformation (via `rectangle_selection()`).

        This is based on: `kornia.augmentation.RandomErasing`.

        Args:
            scale: Same parameter and defaults as `kornia.augmentation.RandomErasing`.
            ratio: Same parameter and defaults as `kornia.augmentation.RandomErasing`.
        """
        super().__init__(**kwargs)
        self._param_generator = random_generator.RectangleEraseGenerator(scale, ratio)

    def rectangle_selection(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, height, width = batch["features"].shape[:3]

        # Similar to kornia.augmentation.RandomErasing
        params = self._param_generator((batch_size, height, width))

        # We can either move the bboxes to GPU and do the mask calculation on the GPU or
        # we do the mask calculation on the CPU and then move the result to the CPU.
        # Moving only the final mask to the GPU is the fastest option but produces high CPU load
        # which is why here we use a compromise which is around 5 % slower but with less CPU load
        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"]).to(
            batch["features"].device
        )
        mask = bbox_to_mask(bboxes, width, height)
        return mask == 1.0


class RandomRectangleErasing(RectangleMixin, ContextTransformation):
    def __init__(self, **kwargs):
        """
        Randomly removes a rectangle in the image.

        This transformation is very similar to `kornia.augmentation.RandomErasing` but allows to replace the rectangle area with random numbers or with cloth instead of only fixed values.

        Note: Additionally, this class marks the area as invalid. This also happens implicitly with Kornia since the valid_pixels are also replaced with 0, i.e. False.
        """
        super().__init__(**kwargs)

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = self.rectangle_selection(batch)
        return self._apply_selection(selection, batch)

    def __repr__(self) -> str:
        return f"RandomRectangleErasing(fill_value={self.fill_value}, p={self.p})"


class SuperpixelMixin:
    def __init__(self, proportion: Union[float, tuple[float, float]], p: float = 1, **kwargs):
        """
        Mixin which can be added to a class to get a random superpixel selection (via `superpixel_selection()`).

        Note: The superpixel transformations (e.g. `SuperpixelOrganTransplantation`) must be applied on the GPU (like any other transformation which operates on the batch-level) and the affine transformations come before the superpixel transformations (to have more variation in superpixels). The affine transformations cannot transform a superpixel mask perfectly because of the border expansion (leading to duplicate or missing superpixel indices). However, for the superpixel transformations this is not an issue because there is already a high amount of randomness in the transformations itself and this glitch just introduces a bit more randomness.

        Args:
            proportion: Proportion of superpixels which should be removed in the image. Either a float to remove a fixed proportion in every image where this transformation will be applied or a range [min;max[ to sample a proportional value randomly from a uniform distribution for each image.
            p: The probability of applying this transformation to an image
        """
        super().__init__(**kwargs)
        self.proportion = proportion
        self.p = p
        if type(self.proportion) == float:
            assert 0 <= self.proportion <= 1, f"The proportion value must be in the range [0;1], not {self.proportion}"
        else:
            assert all(
                0 <= p <= 1 for p in self.proportion
            ), f"Every proportion value must be in the range [0;1], not {self.proportion}"
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"

    def superpixel_selection(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert "spxs" in batch, (
            "This transformation can only be used if the batch already contains superpixels. Please specify the"
            " appropriate input/superpixels keys in your config"
        )

        batch_size = batch["features"].shape[0]
        mask = []
        for b in range(batch_size):
            spx_mask = batch["spxs"][b]
            image_mask = torch.zeros(spx_mask.shape, dtype=torch.bool, device=batch["features"].device)

            # Randomly select a number of superpixels which should be removed
            n_superpixels = spx_mask.max() + 1
            if type(self.proportion) == float:
                n_remove = int(self.proportion * n_superpixels)
            else:
                proportion = torch.empty(1, device=batch["features"].device).uniform_(
                    self.proportion[0], self.proportion[1]
                )
                n_remove = int(proportion * n_superpixels)

            random_spxs = torch.randint(low=0, high=n_superpixels, size=(n_remove,), device=batch["features"].device)
            for spx_index in random_spxs:
                image_mask.masked_fill_(spx_mask == spx_index, True)

            mask.append(image_mask)

        return torch.stack(mask)


class RandomSuperpixelErasing(SuperpixelMixin, ContextTransformation):
    def __init__(self, **kwargs):
        """
        Randomly removes superpixels in an image.
        """
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"RandomSuperpixelErasing(fill_value={self.fill_value}, proportion={self.proportion}, p={self.p})"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = self.superpixel_selection(batch)
        return self._apply_selection(mask, batch)


class SelectionTransplantationMixin:
    def __init__(self, **kwargs):
        """
        Mixin which can be added to a class to transplant organs between images in a batch based on a selection mask.
        """
        super().__init__(**kwargs)

    def transplant_selection(self, selection: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert selection.size(0) == batch["features"].size(0)

        batch_size = selection.size(0)
        n_transform_samples = int(self.p * batch_size)

        features_last = batch["features"][-1]
        labels_last = batch["labels"][-1]
        valid_pixels_last = batch["valid_pixels"][-1]
        if n_transform_samples == batch_size:
            # Same reason as for the OrganTransplantation transformation
            features_last = features_last.clone()
            labels_last = labels_last.clone()
            valid_pixels_last = valid_pixels_last.clone()

        for donor in torch.arange(n_transform_samples):
            acceptor = (donor - 1) % batch_size
            s = selection[donor]
            if donor == batch_size - 1:
                batch["features"][acceptor, s, :] = features_last[s, :]
                batch["labels"][acceptor, s] = labels_last[s]
                batch["valid_pixels"][acceptor, s] = valid_pixels_last[s]
            else:
                batch["features"][acceptor, s, :] = batch["features"][donor, s, :]
                batch["labels"][acceptor, s] = batch["labels"][donor, s]
                batch["valid_pixels"][acceptor, s] = batch["valid_pixels"][donor, s]

        return batch


class SuperpixelOrganTransplantation(SelectionTransplantationMixin, SuperpixelMixin, HTCTransformation):
    def __init__(self, **kwargs):
        """
        Transformation which moves organs between images in a batch. This is very similar to the RectangleOrganTransplantation but moves superpixels instead of rectangular regions.
        """
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"SuperpixelOrganTransplantation(proportion={self.proportion}, p={self.p})"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = self.superpixel_selection(batch)
        return self.transplant_selection(selection, batch)


class RectangleOrganTransplantation(SelectionTransplantationMixin, RectangleMixin, HTCTransformation):
    def __init__(self, p: float = 1, **kwargs):
        """
        Transformation which moves rectangle areas between images in a batch.

        This is similar to OrganTransplantation but with rectangular boundaries.

        Args:
            p: The proportion of images inside the batch where this augmentation should be applied to.
        """
        super().__init__(**kwargs)
        self.p = p
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"

    def __repr__(self) -> str:
        return f"RectangleOrganTransplantation(p={self.p})"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = self.rectangle_selection(batch)
        return self.transplant_selection(selection, batch)


class OrganTransplantation(HTCTransformation):
    def __init__(self, p: float = 1, annotation_names: list[str] = None, paths: list[DataPath] = None, **kwargs):
        """
        Transformation which moves organs between images in a batch.

        Note: Please make sure that kornia transformations (shift, rotate, scale etc.) are performed before this transformation and not afterwards!

        This transformation can handle additional inputs like data_L1 or data_parameter_images. The corresponding features are transplanted in the same way as the original features. Additional label masks like labels_polygon#annotator1 are transplanted as well (similar to labels) but the transplanted region is only selected based on the main labels. Region keys are also supported (e.g. regions_sam) but the indices are changed for the transplanted region to ensure unique region indices inside one image.

        Args:
            p: The proportion of images inside the batch where this augmentation should be applied to. It would be good to select a p value that is a multiple of 1/batch_size, otherwise, it will only be approximated.
            annotation_names: List of annotation names to restrict donor images. Only images which have any of those annotation names will be used as donors. If None, all images can be donors.
            paths: List of training data paths (usually automatically passed to the constructor). Must be set if annotation_names is not None.
        """
        self.p = p
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"
        self.annotation_names = annotation_names
        self.paths = paths
        if self.annotation_names is not None:
            assert (
                self.paths is not None and len(self.paths) > 0
            ), "If annotation_names is not None, paths must be set and not empty"

    def __repr__(self) -> str:
        return f"OrganTransplantation(p={self.p}, annotation_names={self.annotation_names})"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["features"].shape[0]
        n_transform_samples = int(self.p * batch_size)

        if self.annotation_names is not None:
            # Only use images with one of the specified annotation names as donor (every image can be an acceptor)
            donor_indices = []
            for i, image_index in enumerate(batch["image_index"]):
                names = self.paths[image_index].annotation_names()
                if any(n in self.annotation_names for n in names):
                    donor_indices.append(i)

            donor_indices = donor_indices[:n_transform_samples]
        else:
            donor_indices = list(range(n_transform_samples))

        if len(donor_indices) == 0:
            return batch
        else:
            return self._apply_transform(batch, donor_indices)

    def _apply_transform(self, batch: dict[str, torch.Tensor], donor_indices: list[int]) -> dict[str, torch.Tensor]:
        batch_size = batch["features"].shape[0]
        features_keys = [
            k for k in ["features", "features_rgb", "data_L1", "data_parameter_images"] if k in batch.keys()
        ]
        regions_keys = [k for k in batch.keys() if k.startswith("regions")]

        # The main label mask (labels and valid_pixels) are used for the transplantation. For the additional labels, we just transplant the same region
        additional_label_keys = [k for k in batch.keys() if k.startswith(("labels_", "valid_pixels_"))]

        features_last = {k: batch[k][-1] for k in features_keys}
        labels_last = batch["labels"][-1]
        valid_pixels_last = batch["valid_pixels"][-1]
        additional_labels = {k: batch[k][-1] for k in additional_label_keys}
        regions = {k: batch[k][-1] for k in regions_keys}
        if batch_size - 1 in donor_indices:
            # If all images in the batch are affected, then the last image already received a transplanted organ before being a donor itself. Hence, we need to make sure we have a copy of the last image in this case
            features_last = {k: v.clone() for k, v in features_last.items()}
            labels_last = labels_last.clone()
            valid_pixels_last = valid_pixels_last.clone()
            additional_labels = {k: v.clone() for k, v in additional_labels.items()}
            regions = {k: v.clone() for k, v in regions.items()}

        # Randomly select an organ from the donor sample:
        for donor in donor_indices:
            acceptor = (donor - 1) % batch_size

            if donor == batch_size - 1:
                donor_features = features_last
                donor_labels = labels_last
                donor_valid_pixels = valid_pixels_last
                donor_additional_labels = additional_labels
                donor_regions = regions
            else:
                donor_features = {k: batch[k][donor] for k in features_keys}
                donor_labels = batch["labels"][donor]
                donor_valid_pixels = batch["valid_pixels"][donor]
                donor_additional_labels = {k: batch[k][donor] for k in additional_label_keys}
                donor_regions = {k: batch[k][donor] for k in regions_keys}

            valid_donor_labels = donor_labels[donor_valid_pixels].unique()
            selected_label = valid_donor_labels[torch.randperm(len(valid_donor_labels))[0]]

            # Apply selection to organ acceptor
            selection = donor_labels == selected_label
            assert selection.sum() > 0

            batch["labels"][acceptor, selection] = donor_labels[selection]
            batch["valid_pixels"][acceptor, selection] = donor_valid_pixels[selection]
            for k in features_keys:
                batch[k][acceptor, selection, :] = donor_features[k][selection, :]
            for k in additional_label_keys:
                batch[k][acceptor, selection] = donor_additional_labels[k][selection]
            for k in regions_keys:
                # Region indices are only unique within one image, i.e. it is possible that a donor region contains the same index which also exists in the acceptor image. Hence, we need to make sure that different regions inside an image still have different indices
                batch[k][acceptor, selection] = donor_regions[k][selection] + batch[k][acceptor].max() + 1

        return batch


class RandomJigsaw(HTCTransformation):
    def __init__(self, patch_size: list[tuple[int, int]], p: float = 1, **kwargs):
        """
        Creates random jigsaws of 1 or more images in a batch.

        Note: If the augmentation is applied to more than one image, the patches are also exchanged between images (and not just shuffled inside one image).

        Args:
            patch_size: List of patch sizes (height, width) specifying the size of the grid in the jigsaw. Each size must be divisible by the image dimensions (height, width). For each batch, a patch size will be randomly selected (all images in the batch must use the same grid since patches are also transferred between images in the batch).
            p: The proportion of images inside the batch where this augmentation should be applied to.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.p = p
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"

    def __repr__(self) -> str:
        return f"RandomJigsaw(patch_size={self.patch_size}, p={self.p})"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Select a random patch size for this image
        selected_patch_size = torch.randint(0, len(self.patch_size), (1,))
        patch_height = self.patch_size[selected_patch_size][0]
        patch_width = self.patch_size[selected_patch_size][1]

        batch_size = batch["features"].size(0)
        n_images = int(self.p * batch_size)
        if n_images > 0:
            features = batch["features"][:n_images]
            labels = batch["labels"][:n_images].unsqueeze(dim=-1)
            valid_pixels = batch["valid_pixels"][:n_images].unsqueeze(dim=-1)

            features, permutation = self._permute_tensor(features, patch_height, patch_width, return_permutation=True)
            labels = self._permute_tensor(labels, patch_height, patch_width, permutation).squeeze(dim=-1)
            valid_pixels = self._permute_tensor(valid_pixels, patch_height, patch_width, permutation).squeeze(dim=-1)

            batch["features"][:n_images] = features
            batch["labels"][:n_images] = labels
            batch["valid_pixels"][:n_images] = valid_pixels

        return batch

    def _permute_tensor(
        self,
        tensor: torch.Tensor,
        patch_height: int,
        patch_width: int,
        permutation: torch.Tensor = None,
        return_permutation: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size, height, width, channels = tensor.shape
        original_type = tensor.dtype

        # BHWC
        tensor = tensor.permute(1, 2, 0, 3)  # BHWC --> HWBC
        patch_features = (
            # Unfold H dim
            tensor.unfold(dimension=0, size=patch_height, step=patch_height)
            # Unfold W dim
            .unfold(dimension=1, size=patch_width, step=patch_width)
            # B*LChw (L = number of patches; h, w = patch dimensions)
            .reshape(-1, channels, patch_height, patch_width)
        )

        # Permute patches across all images in the batch
        if permutation is None:
            permutation = torch.randperm(patch_features.size(0), device=tensor.device)
        patch_features = patch_features[permutation, :, :, :]

        # B*LChw --> BLX --> BXL (X = B*h*w)
        tensor = patch_features.reshape(batch_size, -1, channels * patch_height * patch_width).permute(0, 2, 1)

        # Fold back to image size
        tensor = F.fold(
            tensor.type(torch.float32),
            output_size=(height, width),
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )
        tensor = tensor.type(original_type).permute(0, 2, 3, 1)  # BCHW --> BHWC

        if return_permutation:
            return tensor, permutation
        else:
            return tensor
