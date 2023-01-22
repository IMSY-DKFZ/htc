# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Callable as _Callable

from htc.models.common.HTCModel import HTCModel as _HTCModel
from htc.models.image.ModelImage import ModelImage as _ModelImage
from htc.models.pixel.ModelPixel import ModelPixel as _ModelPixel
from htc.models.pixel.ModelPixelRGB import ModelPixelRGB as _ModelPixelRGB
from htc.models.superpixel_classification.ModelSuperpixelClassification import (
    ModelSuperpixelClassification as _ModelSuperpixelClassification,
)
from htc.utils.Config import Config as _Config


def _inherit_doc(origin: _Callable, replacements: dict[str, str]):
    def wrapper(func: _Callable):
        func.__doc__ = origin.__doc__

        for old_name, new_name in replacements.items():
            func.__doc__ = func.__doc__.replace(old_name, new_name)

        return func

    return wrapper


def image(
    run_folder: str, pretrained: bool = True, fold_name: str = None, n_classes: int = None, n_channels: int = None
) -> _ModelImage:
    """
    Pretrained image-based segmentation model. You can directly use this model to train a network on your data. The weights will be initialized with the weights from the pretrained network except the segmentation head which is initialized randomly (and may also be different in terms of number of classes).

    Load the pretrained model for the image-based segmentation network:
    >>> import torch
    >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
    >>> print("some log messages"); model = torch.hub.load("IMSY-DKFZ/htc", "image", run_folder=run_folder, trust_repo=True)  # doctest: +ELLIPSIS
    some log messages...
    >>> input_data = torch.randn(1, 100, 480, 640)  # NCHW
    >>> model(input_data).shape
    torch.Size([1, 19, 480, 640])

    It is also possible to have a different number of classes as output or a different number of channels as input:
    >>> print("some log messages"); model = torch.hub.load("IMSY-DKFZ/htc", "image", run_folder=run_folder, n_classes=3, n_channels=10, trust_repo=True)  # doctest: +ELLIPSIS
    some log messages...
    >>> input_data = torch.randn(1, 10, 480, 640)  # NCHW
    >>> model(input_data).shape
    torch.Size([1, 3, 480, 640])

    Args:
        run_folder: Name of the training run from which the weights should be loaded (e.g. to select HSI or RGB models).
        pretrained: If True, load the pretrained weights from the network corresponding to the run_folder. If False, weights are initialized randomly.
        fold_name: Name of the validation fold which defines the trained network of the run. If None, the model with the highest metric score will be used.
        n_classes: Number of classes for the network output. If None, uses the same setting as in the trained network (e.g. 18 organ classes + background for the organ segmentation networks).
        n_channels: Number of channels of the input. If None, uses the same settings as in the trained network (e.g. 100 channels).

    Returns: Model with pretrained weights.
    """
    if pretrained:
        return _ModelImage.pretrained_model("image", run_folder, fold_name, n_classes, n_channels)
    else:
        run_dir = _HTCModel.find_pretrained_run("image", run_folder)
        config = _Config(run_dir / "config.json")
        return _ModelImage(config)


@_inherit_doc(image, {"image": "patch"})
def patch(
    run_folder: str, pretrained: bool = True, fold_name: str = None, n_classes: int = None, n_channels: int = None
) -> _ModelImage:
    if pretrained:
        return _ModelImage.pretrained_model("patch", run_folder, fold_name, n_classes, n_channels)
    else:
        run_dir = _HTCModel.find_pretrained_run("patch", run_folder)
        config = _Config(run_dir / "config.json")
        return _ModelImage(config)


@_inherit_doc(
    image,
    {
        "image": "superpixel_classification",
        "1, 100, 480, 640": "2, 100, 32, 32",
        "torch.Size([1, 19, 480, 640])": "torch.Size([2, 19])",
        "1, 10, 480, 640": "2, 10, 32, 32",
        "torch.Size([1, 3, 480, 640])": "torch.Size([2, 3])",
    },
)
def superpixel_classification(
    run_folder: str, pretrained: bool = True, fold_name: str = None, n_classes: int = None, n_channels: int = None
) -> _ModelSuperpixelClassification:
    if pretrained:
        return _ModelSuperpixelClassification.pretrained_model(
            "superpixel_classification", run_folder, fold_name, n_classes, n_channels
        )
    else:
        run_dir = _HTCModel.find_pretrained_run("superpixel_classification", run_folder)
        config = _Config(run_dir / "config.json")
        return _ModelSuperpixelClassification(config)


@_inherit_doc(
    image,
    {
        """
    It is also possible to have a different number of classes as output or a different number of channels as input:
    >>> print("some log messages"); model = torch.hub.load("IMSY-DKFZ/htc", "image", run_folder=run_folder, n_classes=3, n_channels=10, trust_repo=True)  # doctest: +ELLIPSIS
    some log messages...
    >>> input_data = torch.randn(1, 10, 480, 640)  # NCHW
    >>> model(input_data).shape
    torch.Size([1, 3, 480, 640])
""": """
    For the pixel model, you can specify a different number of classes but you do not need to set the number of input channels because the underlying convolutional operations directly operate along the channel dimension. Hence, you can just supply input data with a different number of channels and it will work as well.
    >>> print("some log messages"); model = torch.hub.load("IMSY-DKFZ/htc", "pixel", run_folder=run_folder, n_classes=3, trust_repo=True)  # doctest: +ELLIPSIS
    some log messages...
    >>> input_data = torch.randn(2, 90)  # NC
    >>> model(input_data).shape
    torch.Size([2, 3])
""",
        "n_channels: Number of channels of the input. If None, uses the same settings as in the trained network (e.g. 100 channels).\n": (
            ""
        ),
        "image": "pixel",
        "torch.randn(1, 100, 480, 640)  # NCHW": "torch.randn(2, 100)  # NC",
        "model(input_data).shape": "model(input_data)['class'].shape",
        "torch.Size([1, 19, 480, 640])": "torch.Size([2, 19])",
    },
)
def pixel(run_folder: str, pretrained: bool = True, fold_name: str = None, n_classes: int = None) -> _ModelPixel:
    ModelClass = _ModelPixelRGB if "rgb" in run_folder or "parameters" in run_folder else _ModelPixel
    if pretrained:
        return ModelClass.pretrained_model("pixel", run_folder, fold_name, n_classes)
    else:
        run_dir = _HTCModel.find_pretrained_run("pixel", run_folder)
        config = _Config(run_dir / "config.json")
        return ModelClass(config)
