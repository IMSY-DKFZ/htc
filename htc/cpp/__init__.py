# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import functools
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import torch  # The ordering here is important! torch must come before htc._cpp

import htc._cpp


def automatic_numpy_conversion(func: Callable) -> Callable:
    # It should not matter whether the C++ functions are used with np.ndarray or torch.Tensor

    @functools.wraps(func)
    def _automatic_numpy_conversion(*args, **kwargs):
        # Convert numpy arguments to torch tensors
        conversion_happened = False
        new_args = []
        for arg in args:
            if type(arg) == np.ndarray:
                arg = torch.from_numpy(arg)
                conversion_happened = True

            new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if type(value) == np.ndarray:
                value = torch.from_numpy(value)
                conversion_happened = True

            new_kwargs[key] = value

        # Call the actual function
        if conversion_happened:
            # Return value should probably be a numpy array (because at least one argument was a numpy array)
            res = func(*new_args, **new_kwargs)
            if type(res) == tuple:
                return tuple(r.numpy() for r in res)
            elif type(res) == list:
                return [r.numpy() for r in res]
            elif type(res) == dict:
                return {k: v.numpy() for k, v in res.items()}
            else:
                return res.numpy()
        else:
            return func(*new_args, **new_kwargs)

    return _automatic_numpy_conversion


# We are wrapping each C++ function call in a Python function for documentation and additional checks


@automatic_numpy_conversion
def spxs_predictions(
    spxs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    n_classes: int = None,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """
    Calculates a prediction based on the superpixel labels by taking the mode of all labels inside the superpixel and then assigning this label to each pixel in the superpixel.

    Args:
        spxs: Image with the superpixel ids.
        labels: Image with the labels.
        mask: Mask image denoting which pixels should be considered (True = pixel will be used).
        n_classes: The number of classes which determines the output shape (defaults to the number of labels in the segmentation task).

    Returns: Tuple with the predictions (same shape as image) and a counts matrix (shape = (n_superpixels, n_classes)) which lists in detail the number of pixels per class which occurred in each superpixel.
    """
    from htc.settings_seg import settings_seg

    if n_classes is None:
        n_classes = len(settings_seg.labels)

    assert spxs.shape == labels.shape and spxs.shape == mask.shape, "All input tensors must have the same shape"
    assert len(spxs.shape) == 2, "Only 2D images are allowed"

    predictions, spx_label_counts = htc._cpp.spxs_predictions(spxs, labels, mask, n_classes)
    assert predictions.shape == spxs.shape, "Predicted image has the wrong shape"
    assert spx_label_counts.shape == (spxs.max() + 1, n_classes), "Invalid shape for the counts matrix"

    return predictions, spx_label_counts


@automatic_numpy_conversion
def segmentation_mask(
    label_image: torch.Tensor | np.ndarray, color_mapping: dict[tuple[int, int, int], int]
) -> torch.Tensor | np.ndarray:
    """
    Calculates the segmentation mask based on a color image and a defined color mapping. Every color in the label_image must get a new value in color_mapping defined.

    Args:
        label_image: The original (color) label image.
        color_mapping: The dictionary to map color values to labels defined in the format {(r, g, b): label}.

    Returns: The segmentation image.
    """
    assert label_image.shape[-1] == 3, "The label image must have a color dimension"
    assert label_image.dtype == torch.uint8, "The label image must be a RGB color image"

    colors_image = np.unique(label_image.reshape(-1, label_image.shape[2]), axis=0)
    colors_image_set = {tuple(r.tolist()) for r in colors_image}
    colors_mapping_set = set(color_mapping.keys())
    assert colors_image_set == colors_mapping_set, (
        "There must be a color mapping for every color in the image. However, the following colors of the image are"
        f" not part of the mapping: {colors_image_set - colors_mapping_set}"
    )

    seg = htc._cpp.segmentation_mask(label_image, color_mapping)
    assert label_image.shape[:2] == seg.shape, "Invalid shape for the segmentation_mask"

    return seg


@automatic_numpy_conversion
def tensor_mapping(tensor: torch.Tensor | np.ndarray, mapping: dict[int, int]) -> torch.Tensor | np.ndarray:
    """
    General function to efficiently remap values of a tensor in-place.

    >>> tensor = torch.tensor([1, 2], dtype=torch.int64)
    >>> mapping = {1: 10}
    >>> tensor_mapping(tensor, mapping)
    tensor([10,  2])
    >>> tensor
    tensor([10,  2])

    Args:
        tensor: The input tensor.
        mapping: The remapping dictionary.

    Returns: Same as the input tensor with the remapping applied in-place.
    """
    assert len(mapping) > 0, "Empty mapping"
    assert all(
        type(k) == type(v) for k, v in mapping.items()
    ), "All keys and values of the mapping must have the same type"

    if tensor.ndim == 0:
        # Map scalar values directly (in-place)
        tensor.fill_(mapping.get(tensor.item(), tensor.item()))
        return tensor
    else:
        first_value = next(iter(mapping.values()))

        if isinstance(first_value, int):
            assert not tensor.is_floating_point(), f"The tensor must have an integer type ({tensor.dtype = })"
            return htc._cpp.tensor_mapping_integer(tensor, mapping)
        elif isinstance(first_value, float):
            assert tensor.is_floating_point(), f"The tensor must have an floating type ({tensor.dtype = })"
            return htc._cpp.tensor_mapping_floating(tensor, mapping)
        else:
            raise ValueError(f"Invalid type: {type(first_value)}")


@automatic_numpy_conversion
def kfold_combinations(
    pig_indices: list, pig_labels: dict[int, list], min_labels: int, n_groups: int = 5
) -> list[list[int]]:
    """
    Finds and evaluates all possible fold combinations. See evaluation/KFoldSelection.ipynb for an example.

    Args:
        pig_indices: pig ids to be distributed across the folds.
        pig_labels: List of labels for each pig id.
        min_labels: Minimal number of labels in the validation set across all folds. If a fold does not meet this requirement, it is discarded.
        n_groups: The k in kfolds.

    Returns: List of fold combinations which meet the requirements.
    """
    assert len(pig_indices) / 3 == n_groups, "Only group sizes of 3 are supported at the moment"
    assert all(all(v <= 255 for v in t) for t in pig_labels.values()), "Label ids must be <= 255"

    return htc._cpp.kfold_combinations(pig_indices, pig_labels, min_labels, n_groups)


@automatic_numpy_conversion
def nunique(inp: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    Counts the unique elements along dimension dim.

    >>> tensor = torch.tensor([[1, 2], [1, 3]])
    >>> nunique(tensor, dim=0)
    tensor([1, 2])
    >>> nunique(tensor, dim=1)
    tensor([2, 2])

    Args:
        inp: The input tensor.
        dim: The dimension along which the number of different elements should be counted. If None, The number of unique elements are calculated based on the flattened input.

    Returns: The output tensor with the counts (always of type torch.int64).
    """
    if dim is None:
        inp = inp.flatten()
        dim = 0

    assert 0 <= dim <= len(inp.shape) - 1, f"dim {dim} is out of range for the input tensor of shape {inp.shape}"
    assert inp.nelement() > 0, "Tensor is empty"

    return htc._cpp.nunique(inp, dim)


@automatic_numpy_conversion
def map_label_image(label_image: torch.Tensor, label_mapping: "LabelMapping") -> torch.Tensor:  # noqa: F821
    """
    Creates a colored segmentation mask based on a label image and a correspond color mapping (defined as part of a label mapping).

    >>> from htc.utils.LabelMapping import LabelMapping
    >>> mapping = LabelMapping({"a": 0, "b": 1}, last_valid_label_index=1, label_colors={"a": "#FFFFFF", "b": "#000000"})
    >>> map_label_image(torch.tensor([[0, 1, 1]]), mapping)
    tensor([[[1., 1., 1., 1.],
             [0., 0., 0., 1.],
             [0., 0., 0., 1.]]])

    Args:
        label_image: Two-dimensional image with integer labels.
        label_mapping: Label mapping with a defined color mapping.

    Returns: RGBA segmentation image.
    """
    assert len(label_image.shape) == 2, "The label image must be two-dimensional"

    label_color_mapping = {
        i: mpl.colors.to_rgba(label_mapping.index_to_color(i)) for i in label_image.unique().tolist()
    }

    return htc._cpp.map_label_image(label_image, label_color_mapping)


def hierarchical_bootstrapping(
    mapping: dict[int, dict[int, list[int]]], n_subjects: int, n_images: int, n_bootstraps: int = 1000
) -> torch.Tensor:
    """
    Creates bootstrap samples based on a three-level hierarchy (domain_name, subject_name, image_name) while always selecting all domains in every bootstrap and the specified number of subjects and images (both with resampling).

    Note: This function is not deterministic but you can set a seed.

    >>> from lightning import seed_everything
    >>> seed_everything(0)  # doctest: +ELLIPSIS
    [...]
    >>> mapping = {
    ...     0: {0: [10]},  # First camera, one subject with one image
    ...     1: {1: [20, 30], 2: [40]},  # Second camera, two subjects with two and one image each
    ... }
    >>> hierarchical_bootstrapping(mapping, n_subjects=2, n_images=1, n_bootstraps=4)
    tensor([[30, 20, 10, 10],
            [30, 20, 10, 10],
            [40, 40, 10, 10],
            [30, 40, 10, 10]])

    Args:
        mapping: Domain to subjects to images mapping.
        n_subjects: Number of subjects to draw with replacement.
        n_images: Number of images to draw with replacement.
        n_bootstraps: Total number of bootstraps.

    Returns: Matrix of shape (n_bootstraps, n_domains * n_subjects * n_images) with the bootstraps. It contains the values provided for the images (final layer in the mapping).
    """
    # We are generating a random number which will be used as seed during bootstraping
    # This produces different bootstraps when the user calls this function multiple times while still allowing to set a seed
    seed = torch.randint(0, torch.iinfo(torch.int32).max, (1,), dtype=torch.int32).item()
    bootstraps = htc._cpp.hierarchical_bootstrapping(mapping, n_subjects, n_images, n_bootstraps, seed)
    assert bootstraps.shape == (n_bootstraps, len(set(mapping.keys())) * n_subjects * n_images)

    return bootstraps


def hierarchical_bootstrapping_labels(
    domain_subjects_images_mapping: dict[int, dict[int, list[int]]],
    label_images_mapping: dict[int, list[int]],
    n_labels: int,
    n_bootstraps: int = 1000,
    oversampling: bool = False,
) -> torch.Tensor:
    """
    Creates bootstrap samples based on a three-level hierarchy (domain_name, subject_name, image_name) while always selecting all domains equally often in every bootstrap. Compared to `hierarchical_bootstrapping()`, this function takes the labels into account and always selects images with the same label for each domain tuple. For each domain and label, one subject and one image is selected, i.e. selection of different subjects is preferred over selecting many images per subject.

    Using this function, a batch with a size of 4 may look like this:
    | domain | subject | image | label |
    |---|---|---|---|
    | D1 | S1 | I1 | liver |
    | D2 | S2 | I2 | liver |
    | D1 | S1 | I3 | colon |
    | D2 | S3 | I4 | colon |

    Note: This function is not deterministic but you can set a seed.

    >>> from lightning import seed_everything
    >>> seed_everything(0)  # doctest: +ELLIPSIS
    [...]
    >>> domain_mapping = {
    ...     0: {0: [10, 11]},  # First camera, one subject with two images
    ...     1: {1: [20, 30], 2: [40]},  # Second camera, two subjects with two and one image each
    ... }
    >>> label_images_mapping = {
    ...     100: [10, 11, 20],  # Images 10, 11 and 20 have label 100
    ...     200: [10, 30, 40],  # Images 10, 30 and 40 have label 200
    ... }
    >>> hierarchical_bootstrapping_labels(domain_mapping, label_images_mapping, n_labels=2, n_bootstraps=4)
    tensor([[20, 10, 30, 10],
            [20, 10, 20, 11],
            [20, 11, 20, 11],
            [30, 10, 20, 11]])

    Args:
        domain_subjects_images_mapping: Domain to subjects to images mapping.
        label_images_mapping: Label to images mapping. Every image must occur in the domain_subjects_images_mapping exactly once.
        n_labels: Number of labels to draw with replacement per domain. For example, with 3 domains and 2 labels, 6 images will be selected per bootstrap sample.
        n_bootstraps: Total number of bootstraps.
        oversampling: If True, instead selecting the labels randomly, the least currently chosen label is selected first. This is achieved by keeping an account for the already selected labels (including every label for each image) which is updated whenever selecting an image. This may still not yield a perfect balance across labels because some labels appear on nearly all images (e.g., background) but underrepresented classes are at least selected as often as possible.

    Returns: Matrix of shape (n_bootstraps, n_domains * n_labels) with the bootstraps. It contains the values provided for the images (final layer in the mappings).
    """
    n_domains = len(set(domain_subjects_images_mapping.keys()))
    images2domain = {
        img: d
        for d, subjects in domain_subjects_images_mapping.items()
        for images in subjects.values()
        for img in images
    }
    for label, images in label_images_mapping.items():
        assert (
            len({images2domain[img] for img in images}) == n_domains
        ), f"Label {label} is not present in all domains (only the images {images} have this label)"

    # We are generating a random number which will be used as seed during bootstraping
    # This produces different bootstraps when the user calls this function multiple times while still allowing to set a seed
    seed = torch.randint(0, torch.iinfo(torch.int32).max, (1,), dtype=torch.int32).item()
    bootstraps = htc._cpp.hierarchical_bootstrapping_labels(
        domain_subjects_images_mapping, label_images_mapping, n_labels, n_bootstraps, oversampling, seed
    )
    assert bootstraps.shape == (n_bootstraps, n_domains * n_labels)

    return bootstraps


@automatic_numpy_conversion
def colorchecker_automask(
    rot_image: torch.Tensor,
    cc_board: str,
    square_size: int,
    safety_margin: int,
    square_dist_vertical: int,
    square_dist_horizontal: int,
) -> dict[str, dict[str, int]]:
    """
    Automatically detect colorchecker chips. See the ColorcheckerReader class for usage of this function.
    """
    assert rot_image.ndim == 3, "The image must be three-dimensional"
    assert rot_image.dtype == torch.float32, "The image must be of type torch.float32"

    return htc._cpp.colorchecker_automask(
        rot_image, cc_board, square_size, safety_margin, square_dist_vertical, square_dist_horizontal
    )


@automatic_numpy_conversion
def colorchecker_automask_search_area(
    rot_image: torch.Tensor,
    cc_board: str,
    square_size: int,
    safety_margin: int,
    square_dist_vertical: int,
    square_dist_horizontal: int,
) -> torch.Tensor:
    """
    Visualize the search are of the automatic colorchecker detection algorithm. See the ColorcheckerReader class for usage of this function.
    """
    assert rot_image.ndim == 3, "The image must be three-dimensional"
    assert rot_image.dtype == torch.float32, "The image must be of type torch.float32"

    return htc._cpp.colorchecker_automask_search_area(
        rot_image, cc_board, square_size, safety_margin, square_dist_vertical, square_dist_horizontal
    )
