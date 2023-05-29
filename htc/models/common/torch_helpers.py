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


def smooth_one_hot(labels: torch.Tensor, n_classes: int, smoothing: float = 0.0):
    """
    Create one-hot label vectors with optional label smoothing:

        - if smoothing == 0, it's one-hot method
        - if 0 < smoothing < 1, it's smooth method

    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962

    >>> labels = torch.tensor([1, 0])
    >>> smooth_one_hot(labels, n_classes=2, smoothing=0)
    tensor([[0., 1.],
            [1., 0.]])
    >>> smooth_one_hot(labels, n_classes=2, smoothing=0.1)
    tensor([[0.1000, 0.9000],
            [0.9000, 0.1000]])

    Args:
        labels: Vector with the label index values.
        n_classes: Number of classes which determines the output shape of the smoothed label vector.
        smoothing: Smoothing value which will be equally distributed across all other classes, e.g. if smoothing=0.1 then for the label index 1 the vector [0.1, 0.9] will be returned.
    """
    assert 0 <= smoothing < 1, "Invalid smoothing value"
    assert len(labels.shape) == 1, "labels must be a vector"
    assert labels.dtype == torch.int64, "Wrong type for labels vector"

    confidence = 1.0 - smoothing
    label_shape = torch.Size((labels.size(0), n_classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=labels.device)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, labels.data.unsqueeze(1), confidence)

    return true_dist


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
