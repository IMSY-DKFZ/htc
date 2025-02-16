# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pickle
from collections.abc import Iterator
from typing import Self

import kornia.augmentation as K
import numpy as np
import torch

from htc.models.common.torch_helpers import str_to_dtype
from htc.models.common.utils import dtype_from_config
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.type_from_string import type_from_string


class HTCTransformation:
    def __init__(self, paths: list[DataPath] = None, **kwargs):
        """
        Base class for all transformations.

        If you have a transformation where you need access to the corresponding image (DataPath) and need to apply the transformation to each image separately, you can inherit from this class and overwrite the `transform_image()` method.

        Args:
            paths: List of paths for which this transformation is called on. This is required if the transformation is applied on batches of images and no image_name_annotations but only an image_index is available. The original image is recovered via paths[image_index].
            **kwargs: Unused.
        """
        self.paths = paths

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert "features" in batch and ("image_name_annotations" in batch or "image_index" in batch), (
            "features and image_name_annotations/image_index are required"
        )
        if "image_name_annotations" in batch:
            paths = [DataPath.from_image_name(image_name) for image_name in batch["image_name_annotations"]]
        else:
            assert self.paths is not None, (
                "self.paths must be provided if image_name_annotations is not part of the batch"
            )
            paths = [self.paths[image_index] for image_index in batch["image_index"]]

        # Default implementation to apply a random transformation for each image
        for b in range(batch["features"].size(0)):
            batch["features"][b] = self.transform_image(paths[b], batch["features"][b])

        return batch

    def transform_image(self, path: DataPath, image: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def parse_transforms(
        transform_specs: list[dict] = None, initial_dtype: torch.dtype = torch.float32, **kwargs
    ) -> list["HTCTransformation"]:
        """
        Parses a list of transformation specifications (as used in a config).

        Args:
            transform_specs: List of transformations, e.g. from a config files: config["transforms_cpu"].
            initial_dtype: There must always be a type transformation in the beginning of the transformation pipeline. In general, we want to use float32 on the GPU to ensure enough precision but likely having the data as float16 on the CPU.
            **kwargs: Additional arguments which will be passed to the transformation class (e.g. reference to the config).
        """

        def _transform_params(transform_specs: list[dict]) -> Iterator[tuple[str, dict]]:
            # If all transformations are from the same type (e.g. KorniaTransform objects), then it is more efficient to have only one transform which includes all transformations for this type. This has the advantage that we have to do the type conversions only once

            transformation_names = []
            transformation_kwargs = []

            i = 0
            while i < len(transform_specs):
                t = transform_specs[i]

                if "transformation_name" in t:
                    trans_params = {}
                    for key, value in t.items():
                        if key == "class":
                            continue

                        if key == "transformation_name":
                            transformation_names.append(value)
                        else:
                            trans_params[key] = value
                    transformation_kwargs.append(trans_params)

                    # We need all consecutive transformations for the current class name
                    has_next = i + 1 < len(transform_specs) and transform_specs[i + 1]["class"] == t["class"]
                    if not has_next:
                        # No further transformations for this class
                        assert len(transformation_names) == len(transformation_kwargs)
                        yield (
                            t["class"],
                            {
                                "transformation_names": transformation_names,
                                "transformation_kwargs": transformation_kwargs,
                            },
                        )
                        transformation_names = []
                        transformation_kwargs = []
                else:
                    # Standard transformation (just one object for the class)
                    yield t["class"], {key: value for key, value in t.items() if key != "class"}

                i += 1

        # Construct config objects
        import htc.models.common.transforms as htc_transforms

        if transform_specs is None:
            transform_specs = []

        if len(transform_specs) > 0 and transform_specs[0]["class"] == "ToType":
            # User does the typing explicitly
            transformations = []
        else:
            # We always want a type transform in the beginning (to ensure correct type on the CPU and enough precision, i.e. fp32, on the GPU)
            transformations = [ToType(initial_dtype)]

        for t_class, t_params in _transform_params(transform_specs):
            if ">" in t_class:
                cls = type_from_string(t_class)
            else:
                cls = getattr(htc_transforms, t_class)
            transformations.append(cls(**t_params, **kwargs))

        settings.log.debug(f"Used transformations:\n{transformations}")

        return transformations

    @staticmethod
    def apply_valid_transforms(
        batch: dict[str, torch.Tensor], transforms: list["HTCTransformation"]
    ) -> dict[str, torch.Tensor]:
        """
        Applies the transformations to the batch and checks whether after the transformation the batch is still valid. That is, at least one valid pixel must remain (per image). Otherwise, the original, non-transformed image will be returned.

        Note: Works only for transformations which do not alter the batch in-place.

        For multi-layer segmentations, at least one valid pixel must remain in the first layer (the first layer is supposed to contain the main annotations of the image), otherwise the transformed image is not used.
        For multiple annotations, at least one valid pixel must remain for each annotated image. If one annotation_name has only invalid pixels left, the original, non-transformed image stays in the batch.

        Args:
            batch: The batch on which the transformations should be applied. All tensors must either be in the BHWC or BHW format.
            transforms: List of transformations.

        Returns: The transformed batch or the original batch. The key 'transforms_applied' with a value of True is added to the batch to mark it as transformed (the key is always added irrespective of whether transformations were applied or not).
        """
        if len(transforms) == 1 and type(transforms[0]) == ToType:
            batch["transforms_applied"] = True
            return transforms[0](batch)

        batch_tmp = batch
        for t in transforms:
            batch_tmp = t(batch_tmp)

        label_keys = [k for k in batch_tmp.keys() if k.startswith("labels")]
        valid_keys = [k for k in batch_tmp.keys() if k.startswith("valid_pixels")]

        if len(label_keys) == 0 and len(valid_keys) == 0:
            # During prediction, we do not have labels
            batch |= batch_tmp
        else:
            # It is possible that an augmentation creates an invalid image, e.g. if only a small part was annotated and then this small part of the image gets rotated outside the field of view. In this case, we just use the original non-augmented image
            # We need to make this decision for each image in the batch
            if len(valid_keys) > 0:
                valid_pixels = [batch_tmp[k] for k in valid_keys]
            else:
                valid_pixels = [batch_tmp[k] < settings.label_index_thresh for k in label_keys]

            if all(t.ndim == 4 for t in valid_pixels):
                # For multi-layer segmentations only the first layer is relevant
                # The first layer contains the main annotations (e.g. semantic segmentations) whereas the other layers usually contain more fine-grained information (e.g. tags). The annotated region in the second layer may also be much smaller than the first layer, so we do not want to discard the image if the second layer is nearly empty
                valid_pixels = [t[..., 0] for t in valid_pixels]

            # Do we have for each image at least one valid pixel?
            # BHW -> BX -> B
            valid_samples = [t.reshape(t.size(0), -1).any(dim=-1) for t in valid_pixels]

            assert all(t.ndim == 1 for t in valid_samples), "Only the batch dimension should remain"

            # In case of multiple annotations, we only keep the transformed image if the augmentations from all annotations still yield a valid image
            valid_samples = torch.stack(valid_samples).all(dim=0)

            if valid_samples.all():
                # Complete batch remains valid, so just use it
                batch |= batch_tmp
            else:
                # Decision for each image in the batch separately
                for key in batch_tmp.keys():
                    # index keys (e.g. image_index) never get transformed so we can just keep the original values
                    if type(batch[key]) == torch.Tensor and not key.endswith("index"):
                        batch[key][valid_samples] = batch_tmp[key][valid_samples]

        batch["transforms_applied"] = True
        return batch


class Normalization(HTCTransformation):
    def __init__(self, order: int = 1, channel_dim: int = -1, **kwargs):
        """
        Simple normalization operation across the spectral channel independent for each pixel. Usually, all our networks expect L1 normalized data to account for multiplicative illumination changes.

        This transformation accepts either a tensor object as input
        >>> normalization = Normalization()
        >>> features = torch.ones(1, 2, 2, 100)
        >>> normalization(features)[0, 0, 0, :2]
        tensor([0.0100, 0.0100])

        or a dictionary with the features of the batch. In this case, only the `features` key of the batch is altered in-place:
        >>> batch = normalization({"features": features})
        >>> batch["features"][0, 0, 0, :2]
        tensor([0.0100, 0.0100])

        Args:
            order: Order of the norm. 1 = L1 norm, 2 = L2 norm, etc.
            channel_dim: Dimension of the channel axis.
            **kwargs: Unused.
        """
        self.order = order
        self.channel_dim = channel_dim

    def __call__(self, batch: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if type(batch) == dict:
            features = batch["features"]
        else:
            features = batch

        features = features / torch.linalg.norm(features, ord=self.order, dim=self.channel_dim, keepdim=True)
        features.nan_to_num_()

        if type(batch) == dict:
            batch["features"] = features
        else:
            batch = features

        return batch

    def __repr__(self) -> str:
        return f"Normalization(order={self.order}, channel_dim={self.channel_dim})"


class StandardizeHSI(HTCTransformation):
    def __init__(self, stype: str, config: Config, fold_name: str, **kwargs):
        """
        This transformation performs standardization on the HSI pixels. This can either be done globally across all values (stype="pixel") or individually per channel (stype="channel") in which case the 480*640 values per channel are standardized separately.

        You must precompute the std and mean values for your specs file with the run_standardization.py script for this transformation to work.

        Args:
            stype: Type of standardization (pixel or channel).
            config: The configuration object used for training.
            fold_name: The name of the current fold (required to load the precomputed mean and std values).
            **kwargs: Unused.
        """
        # Standardization parameters are precomputed
        specs_name = DataSpecification.from_config(config).name()
        params_path = settings.intermediates_dir_all / "data_stats" / f"{specs_name}#standardization.pkl"
        assert params_path.exists(), f"could not find the precomputed standardization parameter at {params_path}"

        params = pickle.load(params_path.open("rb"))
        assert fold_name in params, (
            f"Could not find {fold_name} in standardization file (available keys: {params.keys()})"
        )
        params = params[fold_name]

        if config["input/n_channels"] == 100:
            modality = "hsi"
        elif config["input/n_channels"] == 4:
            modality = "tpi"
        elif config["input/n_channels"] == 3:
            modality = "rgb"
        else:
            raise ValueError(f"Could not map the number of input channels {config['input/n_channels']} to any modality")

        if stype == "pixel":
            self.mean = params[f"{modality}_{stype}_mean"].item()
            self.std = params[f"{modality}_{stype}_std"].item()
        elif stype == "channel":
            self.mean = torch.from_numpy(params[f"{modality}_{stype}_mean"].astype(np.float32))
            self.std = torch.from_numpy(params[f"{modality}_{stype}_std"].astype(np.float32))
        else:
            raise ValueError(f"Unknown stype {stype}")

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch["features"] = (batch["features"] - self.mean) / self.std

        return batch

    def __repr__(self) -> str:
        if isinstance(self.mean, torch.Tensor):
            return f"StandardizeHSI(mean.shape={self.mean.shape})"
        else:
            return f"StandardizeHSI(mean={self.mean}, std={self.std})"


class StandardNormalVariate(HTCTransformation):
    """
    Standardizes each pixel separately. The 100 reflectance values will have zero mean and unit variance.
    """

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features_std = batch["features"].std(dim=-1, keepdim=True)
        features_mean = batch["features"].mean(dim=-1, keepdim=True)
        batch["features"] = (batch["features"] - features_mean) / features_std

        return batch

    def __repr__(self) -> str:
        return "StandardNormalVariate"


class ToType(HTCTransformation):
    def __init__(self, dtype: str | torch.dtype = torch.float32, **kwargs):
        """
        Transformation to convert all floating point tensors to the specified type.

        Args:
            dtype: Target type (e.g. "float32" or torch.float32).
            **kwargs: Unused.
        """
        self.dtype = str_to_dtype(dtype)

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, tensor in batch.items():
            if type(tensor) == torch.Tensor and tensor.is_floating_point():
                batch[key] = tensor.type(self.dtype)

        return batch

    def __repr__(self) -> str:
        return f"ToType(dtype={self.dtype})"

    @staticmethod
    def from_config(config: Config = None) -> Self:
        if config is not None:
            return ToType(dtype=dtype_from_config(config))
        else:
            return ToType(dtype=torch.float32)


class KorniaTransform(HTCTransformation):
    def __init__(self, transformation_names: list[str], transformation_kwargs: list[dict], **kwargs):
        assert len(transformation_names) == len(transformation_kwargs), (
            "There must be arguments for each transformation"
        )

        transforms = []
        for name, t_kwargs in zip(transformation_names, transformation_kwargs, strict=True):
            TransformationClass = getattr(K, name)

            # Kornia expects tuples instead of lists for augmentation parameters
            for k, v in t_kwargs.items():
                if type(v) == list:
                    t_kwargs[k] = tuple(v)

            transforms.append(TransformationClass(**t_kwargs))

        self.compose = K.AugmentationSequential(*transforms)

        self.keys_to_type = {
            "features": "input",
            "features_rgb": "input",
            "data_L1": "input",
            "data_parameter_images": "input",
            "labels": "mask",
            "valid_pixels": "mask",
            "spxs": "mask",
            "regions": "mask",
            "modified_pixels": "mask",
        }
        self.skip_keys = ["meta"]

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Applies the specified transformations to the batch.

        Args:
            batch: Batch with the data (the original data is not modified).

        Returns: New batch with the transformed data. The dictionary includes only the modified keys (e.g. no "image_name").
        """
        assert "features" in batch, "Need features for the transformation"
        for key, value in batch.items():
            # We only augment 2D+ images
            if key.endswith("index") or type(value) != torch.Tensor or value.ndim == 1 or key in self.skip_keys:
                continue

            for known_key, known_type in self.keys_to_type.items():
                if key == known_key:
                    break
                elif key.startswith(known_key):
                    # labels_name1 --> same as labels
                    self.keys_to_type[key] = known_type
                    break

            if key not in self.keys_to_type:
                raise ValueError(
                    f"Found the tensor {key} in the batch but there is no type associated with this key. You probably"
                    " want to add your tensor name to the self.keys_to_type attribute"
                )

        arguments = {}
        trans_types = {}
        dtypes = {}

        # On the CPU, we always need float32 but on the GPU it depends on the input (which again is very often float32)
        dtype = torch.float32 if batch["features"].device.type == "cpu" else batch["features"].dtype

        for key, trans_type in self.keys_to_type.items():
            if key in batch:
                if trans_type == "input":
                    # From BHWC to BCHW
                    arguments[key] = batch[key].permute(0, 3, 1, 2).type(dtype)
                else:
                    # kornia can only handle floats and all tensors must have the same dtype (may lead to strange errors otherwise)
                    arguments[key] = batch[key].type(dtype)
                    if arguments[key].ndim == 3:
                        # Add channel dim if not already available
                        arguments[key] = arguments[key].unsqueeze(dim=1)

                trans_types[key] = trans_type
                dtypes[key] = batch[key].dtype

        augmented = self.compose(*arguments.values(), data_keys=list(trans_types.values()))
        if type(augmented) == torch.Tensor:
            augmented = [augmented]
        augmented = dict(zip(arguments.keys(), augmented, strict=True))

        batch_transformed = {}
        for key in batch.keys():
            if key in augmented:
                if self.keys_to_type[key] == "input":
                    # From BCHW to BHWC
                    batch_transformed[key] = augmented[key].permute(0, 2, 3, 1).type(dtypes[key])
                else:
                    # Back to the original type and remove channel dim if 1
                    batch_transformed[key] = augmented[key].type(dtypes[key]).squeeze(dim=1)
            else:
                # Make sure that the stuff we don't change is still available (e.g. image name)
                batch_transformed[key] = batch[key]

        return batch_transformed

    def __repr__(self) -> str:
        return "KorniaTransform"
