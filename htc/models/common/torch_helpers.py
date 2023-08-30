# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import math
import types
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_tensors(
    tensors: list[torch.Tensor],
    dim: Union[tuple[int, ...], int] = (0, 1),
    pad_value: float = 0.0,
    size_multiple: Union[tuple[int, ...], int] = None,
) -> list[torch.Tensor]:
    """
    Pads a list of tensors and appends new values to the right/bottom for the shape dimensions provided. This can be used to stack/cat tensors of different shape.

    >>> tensor_a = torch.ones(2, 2, dtype=torch.int32)
    >>> tensor_b = torch.ones(3, 3, dtype=torch.int32)
    >>> tensor_ab = pad_tensors([tensor_a, tensor_b])
    >>> tensor_ab[0].shape
    torch.Size([3, 3])
    >>> torch.stack(tensor_ab).shape
    torch.Size([2, 3, 3])

    Args:
        tensors: List of tensors to stack.
        dim: Dimensions which should be padded.
        pad_value: Padding value appended to the tensors.
        size_multiple: If not None, makes sure that the size of the tensors is divisible by this size, e.g. to have a size divisible by 32. Either provide a tuple with values per dimension (e.g. (height, width)) or an int which is then used for all dimensions.

    Returns: Batch tensor (first dimension is the batch dimension).
    """
    if type(dim) == int:
        dim = (dim,)

    shapes = [t.shape for t in tensors]
    assert [shapes[0] == len(s) for s in shapes], "All tensors must have the same number of shape dimensions"
    assert len(dim) <= len(
        shapes[0]
    ), f"dim provides {len(dim)} dimensions but the tensors only have {len(shapes[0])} dimensions"
    target_sizes = [max(s[d] for s in shapes) for d in dim]

    if size_multiple is not None:
        # Increase the target size up to the next number divisible by the given multiplier
        if type(size_multiple) == int:
            size_multiple = (size_multiple,) * len(dim)

        assert len(dim) == len(
            size_multiple
        ), "A size multiple must be given for each dimension or one value for all dimensions"

        for i in range(len(target_sizes)):
            multiple = size_multiple[i]  # e.g. 28
            if target_sizes[i] % multiple != 0:  # 28 % 32 = 0
                target_sizes[i] += (
                    math.ceil(target_sizes[i] / multiple) * multiple - target_sizes[i]
                )  # 28 += 1 * 32 - 28

    padded_tensors = []
    for tensor in tensors:
        # For each dimension, we must specify two padding values (left, right)
        padding = [0] * len(tensor.shape) * 2
        for d, target in zip(dim, target_sizes):
            padding[2 * d] = target - tensor.shape[d]
        padding.reverse()  # For some reason, Pytorch F.pad reads the paddings from right to left
        padding = tuple(padding)

        tensor = F.pad(tensor, padding, value=pad_value)
        padded_tensors.append(tensor)

    return padded_tensors


def scaled_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of 1D tensors of varying lengths into a single 2D tensor by linearly interpolating
    the shorter tensors to match the length of the longest tensor.

    >>> tensor_a = torch.tensor([1, 2, 3])
    >>> tensor_b = torch.tensor([1, 2, 3, 4, 5])
    >>> tensor_ab = scaled_stack([tensor_a, tensor_b])
    >>> tensor_ab.shape
    torch.Size([2, 5])
    >>> tensor_ab
    tensor([[1.0000, 1.5000, 2.0000, 2.5000, 3.0000],
            [1.0000, 2.0000, 3.0000, 4.0000, 5.0000]])

    Args:
        tensors: A list of 1D tensors to be stacked.

    Returns: A stacked tensor with shape (len(tensors), max_len), where max_len is the length of the longest tensor in the input list. The resulting tensor will always have a floating type.
    """
    assert all(t.ndim == 1 for t in tensors), "All tensors must be 1D"

    max_len = max(len(x) for x in tensors)
    for i in range(len(tensors)):
        if len(tensors[i]) < max_len:
            if not tensors[i].is_floating_point():
                tensors[i] = tensors[i].to(torch.float32)

            tensors[i] = torch.nn.functional.interpolate(
                tensors[i].unsqueeze(dim=0).unsqueeze(dim=0),
                size=max_len,
                mode="linear",
                align_corners=True,
            ).squeeze()

    return torch.stack(tensors)


def smooth_one_hot(labels: torch.Tensor, n_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    """
    Create one-hot label vectors with optional label smoothing:

        - if smoothing == 0, it's one-hot method
        - if 0 < smoothing < 1, it's smooth method

    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962

    >>> labels = torch.tensor([1, 0])
    >>> smooth_one_hot(labels, n_classes=2, smoothing=0)
    tensor([[0., 1.],
            [1., 0.]], dtype=torch.float16)
    >>> smooth_one_hot(labels, n_classes=2, smoothing=0.1)
    tensor([[0.1000, 0.8999],
            [0.8999, 0.1000]], dtype=torch.float16)

    Args:
        labels: Tensor with label indices, e.g. (batch, height, width).
        n_classes: Number of classes which determines the output shape of the smoothed label vector.
        smoothing: Smoothing value which will be equally distributed across all other classes, e.g. if smoothing=0.1 then for the label index 1 the vector [0.1, 0.9] will be returned.

    Returns: Smoothed one-hot label tensor of type torch.float16. The class dimension will be added to the end, e.g. (batch, height, width, class).
    """
    assert 0 <= smoothing < 1, "Invalid smoothing value"
    assert labels.dtype == torch.int64, "Wrong type for labels vector"

    confidence = 1.0 - smoothing
    new_shape = torch.Size((*labels.shape, n_classes))
    with torch.no_grad():
        labels_smooth = torch.empty(size=new_shape, dtype=torch.float16, device=labels.device)
        labels_smooth.fill_(smoothing / (n_classes - 1))
        labels_smooth.scatter_(dim=-1, index=labels.unsqueeze(dim=-1), value=confidence)

    return labels_smooth


def group_mean(indices: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Group and average values by the given indices.

    >>> indices = torch.tensor([0, 0, 2, 2, 2])
    >>> values = torch.tensor([1, 2, 3, 4, 5])
    >>> group_mean(indices, values)
    (tensor([0, 2]), tensor([1.5000, 4.0000]))

    Args:
        indices: Indices which define the group membership. Tensor will be flattened.
        values: Values which should be averaged. Tensor will be flattened.

    Returns: Indices and corresponding group average values.
    """
    assert indices.dtype == torch.int64, "Indices must be of type int64 (index values)"
    indices = indices.flatten()
    values = values.flatten()
    assert len(indices) == len(values), "Indices and values must have the same length"

    last_index = indices.max() + 1

    aggregated = torch.zeros(last_index, dtype=values.dtype, device=indices.device)
    aggregated.scatter_add_(0, indices, values)

    counts = torch.zeros(last_index, dtype=values.dtype, device=indices.device)
    counts.scatter_add_(0, indices, torch.ones_like(values))

    valid = counts > 0
    valid_indices = torch.arange(0, last_index, dtype=indices.dtype, device=indices.device)[valid]
    valid_aggregated = (aggregated / counts)[valid]

    return valid_indices, valid_aggregated


def move_batch_gpu(batch: dict, device: torch.device = None) -> dict:
    """
    Moves every tensor in the batch to the GPU (or any other device).

    Args:
        batch: Batch with PyTorch tensors.
        device: The device to move the batch to. Defaults to the current CUDA device.

    Returns: Dictionary with the same keys as in batch but with every tensor on the GPU.
    """
    if device is None:
        device = torch.device("cuda")

    batch_gpu = {}
    for key, value in batch.items():
        if type(value) == torch.Tensor:
            batch_gpu[key] = value.to(device)
        elif type(value) == list and all(type(v) == torch.Tensor for v in value):
            batch_gpu[key] = [t.to(device) for t in value]
        else:
            batch_gpu[key] = value

    return batch_gpu


def copy_sample(sample: dict) -> dict:
    """
    Create a copy of the sample (or batch) object. Non-tensor objects are copied as references.

    >>> sample = {"features": torch.tensor([1])}
    >>> sample_copy = copy_sample(sample)
    >>> sample["features"][0] = 10
    >>> sample
    {'features': tensor([10])}
    >>> sample_copy
    {'features': tensor([1])}

    Args:
        sample: Dictionary referencing tensor objects.

    Returns: New dictionary with the same keys and copies of the original values.
    """
    batch_copy = {}
    for key, value in sample.items():
        if type(value) == torch.Tensor:
            batch_copy[key] = value.detach().clone()
        else:
            batch_copy[key] = sample[key]

    return batch_copy


def cpu_only_tensor(tensor: torch.Tensor) -> None:
    """
    Force the tensor to stay on the CPU by making the cuda calls a null-op.

    This is useful if you want to prevent that external libraries (e.g. pytorch lightning) move your tensor to the GPU.

    >>> t = cpu_only_tensor(torch.tensor([1]))
    >>> t.cuda().device
    device(type='cpu')

    Args:
        tensor: Tensor which should stay on the CPU
    """
    assert tensor.device == torch.device("cpu"), "The tensor is not on the CPU"

    def to(self, *args, **kwargs):
        if len(args) > 0 and (type(args[0]) == str or type(args[0] == torch.device)):
            args = tuple(args[i] for i in range(len(args)) if i > 0)
        elif "device" in kwargs:
            del kwargs["device"]

        return torch.Tensor.to(self, *args, **kwargs)

    def cuda(self, *args, **kwargs):
        return self

    # Similar to https://discuss.pytorch.org/t/force-a-tensor-to-live-on-a-cpu/1408
    tensor.to = types.MethodType(to, tensor)
    tensor.cuda = types.MethodType(cuda, tensor)

    return tensor


def str_to_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """
    Converts a string type to a PyTorch data type.

    >>> str_to_dtype("float16")
    torch.float16

    Args:
        dtype: Data type as string or torch.dtype. For the latter, the dtype is just returned as is.

    Returns: PyTorch data type.
    """
    if type(dtype) == torch.dtype:
        return dtype

    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype {dtype}")


class FlexibleIdentity(nn.Identity):
    """
    Same as nn.Identity but the forward function also accepts additional arguments.
    """

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input
