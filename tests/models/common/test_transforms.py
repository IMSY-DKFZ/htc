# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from collections.abc import Callable
from functools import partial

import kornia.augmentation as K
import pytest
import torch
from kornia.geometry.transform.flips import hflip
from torch.utils.data import DataLoader

from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.common.transforms import HTCTransformation, Normalization, ToType
from htc.models.common.utils import samples_equal
from htc.models.image.DatasetImage import DatasetImage
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestHTCTransformation:
    def test_apply_valid_transforms(self) -> None:
        def trans_invalid(batch):
            return {
                "valid_pixels": torch.zeros(batch["valid_pixels"].shape, dtype=torch.bool),
                "features": torch.zeros(batch["features"].shape, dtype=torch.float32),
            }

        def trans_valid(batch):
            return {
                "valid_pixels": torch.ones(batch["valid_pixels"].shape, dtype=torch.bool),
                "features": torch.zeros(batch["features"].shape, dtype=torch.float32),
            }

        batch = {
            "valid_pixels": torch.ones(1, 10, 10, 2, dtype=torch.bool),
            "features": torch.ones(1, 10, 10, 3, dtype=torch.float32),
        }
        batch_copy = copy.deepcopy(batch)

        batch_invalid = HTCTransformation.apply_valid_transforms(batch, [trans_invalid])
        del batch_invalid["transforms_applied"]
        assert samples_equal(batch_copy, batch_invalid)

        batch_valid = HTCTransformation.apply_valid_transforms(batch, [trans_valid])
        del batch_valid["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid)

        batch = {
            "valid_pixels": torch.ones(1, 10, 10, dtype=torch.bool),
            "features": torch.ones(1, 10, 10, 3, dtype=torch.float32),
        }
        batch_copy = copy.deepcopy(batch)
        batch_invalid = HTCTransformation.apply_valid_transforms(batch, [trans_invalid])
        del batch_invalid["transforms_applied"]
        assert samples_equal(batch_copy, batch_invalid)

        batch_valid = HTCTransformation.apply_valid_transforms(batch, [trans_valid])
        del batch_valid["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid)

        def trans_one_valid(batch):
            valid_pixels = torch.ones(batch["valid_pixels"].shape, dtype=torch.bool)
            valid_pixels[1] = False
            return {
                "valid_pixels": valid_pixels,
                "features": torch.zeros(batch["features"].shape, dtype=torch.float32),
            }

        batch = {
            "valid_pixels": torch.ones(2, 10, 10, dtype=torch.bool),
            "features": torch.ones(2, 10, 10, 3, dtype=torch.float32),
        }
        batch_copy = copy.deepcopy(batch)
        batch_one_valid = HTCTransformation.apply_valid_transforms(batch, [trans_one_valid])
        del batch_one_valid["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid)
        assert torch.all(batch_one_valid["valid_pixels"])
        assert torch.all(batch_one_valid["features"][0] == 0) and torch.all(batch_one_valid["features"][1] == 1), (
            "Only the first image should change"
        )

        # Multi-layer segmentations
        def trans_invalid_layer(batch, layer: int, row: int = None, col: int = None):
            batch = {
                "valid_pixels": torch.ones(batch["valid_pixels"].shape, dtype=torch.bool),
                "features": torch.zeros(batch["features"].shape, dtype=torch.float32),
            }
            if row is None and col is None:
                batch["valid_pixels"][:, :, :, layer] = False
            else:
                batch["valid_pixels"][:, row, col, layer] = False

            return batch

        batch = {
            "valid_pixels": torch.ones(1, 10, 10, 2, dtype=torch.bool),
            "features": torch.ones(1, 10, 10, 3, dtype=torch.float32),
        }
        batch_copy = copy.deepcopy(batch)

        # There must be a valid pixel in every segmentation layer
        batch_invalid = HTCTransformation.apply_valid_transforms(batch, [partial(trans_invalid_layer, layer=0)])
        del batch_invalid["transforms_applied"]
        assert samples_equal(batch_copy, batch_invalid)

        batch_invalid_layer2 = HTCTransformation.apply_valid_transforms(batch, [partial(trans_invalid_layer, layer=1)])
        del batch_invalid_layer2["transforms_applied"]
        assert not samples_equal(batch_copy, batch_invalid_layer2)

        # Single invalid pixels are no problem
        batch_valid_0 = HTCTransformation.apply_valid_transforms(
            batch, [partial(trans_invalid_layer, layer=0, row=2, col=2)]
        )
        del batch_valid_0["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid_0)

        batch_valid_1 = HTCTransformation.apply_valid_transforms(
            batch, [partial(trans_invalid_layer, layer=1, row=2, col=2)]
        )
        del batch_valid_1["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid_1)

        # Multiple annotations
        def trans_invalid_annotation(batch, annotation_name: str, row: int = None, col: int = None):
            batch = {
                "valid_pixels_a1": torch.ones(batch["valid_pixels_a1"].shape, dtype=torch.bool),
                "features_a1": torch.zeros(batch["features_a1"].shape, dtype=torch.float32),
                "valid_pixels_a2": torch.ones(batch["valid_pixels_a2"].shape, dtype=torch.bool),
                "features_a2": torch.zeros(batch["features_a2"].shape, dtype=torch.float32),
            }
            if row is None and col is None:
                batch[f"valid_pixels_{annotation_name}"].fill_(False)
            else:
                batch[f"valid_pixels_{annotation_name}"][:, row, col] = False

            return batch

        batch = {
            "valid_pixels_a1": torch.ones(1, 10, 10, 2, dtype=torch.bool),
            "features_a1": torch.ones(1, 10, 10, 3, dtype=torch.float32),
            "valid_pixels_a2": torch.ones(1, 10, 10, 2, dtype=torch.bool),
            "features_a2": torch.ones(1, 10, 10, 3, dtype=torch.float32),
        }
        batch_copy = copy.deepcopy(batch)

        # There must be a valid pixel for each annotation
        batch_invalid = HTCTransformation.apply_valid_transforms(
            batch, [partial(trans_invalid_annotation, annotation_name="a1")]
        )
        del batch_invalid["transforms_applied"]
        assert samples_equal(batch_copy, batch_invalid)

        batch_invalid_layer2 = HTCTransformation.apply_valid_transforms(
            batch, [partial(trans_invalid_annotation, annotation_name="a2")]
        )
        del batch_invalid_layer2["transforms_applied"]
        assert samples_equal(batch_copy, batch_invalid_layer2)

        # Single invalid pixels are no problem
        batch_valid_0 = HTCTransformation.apply_valid_transforms(
            batch, [partial(trans_invalid_annotation, annotation_name="a1", row=2, col=2)]
        )
        del batch_valid_0["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid_0)

        batch_valid_1 = HTCTransformation.apply_valid_transforms(
            batch, [partial(trans_invalid_annotation, annotation_name="a2", row=2, col=2)]
        )
        del batch_valid_1["transforms_applied"]
        assert not samples_equal(batch_copy, batch_valid_1)

    def test_default_typing(self) -> None:
        explicit = HTCTransformation.parse_transforms([{"class": "ToType"}])
        assert len(explicit) == 1
        assert type(explicit[0]) == ToType

        implicit = HTCTransformation.parse_transforms([{"class": "Normalization"}])
        assert len(implicit) == 2
        assert type(implicit[0]) == ToType
        assert type(implicit[1]) == Normalization


class TestNormalization:
    def test_basics(self) -> None:
        path = DataPath.from_image_name("P058#2020_05_13_18_09_26")

        config = Config({
            "input/normalization": "L1",
            "input/transforms_cpu": [{"class": "ToType", "dtype": "float32"}],
        })
        sample1 = DatasetImage([path], train=True, config=config)[0]

        config = Config({
            "input/preprocessing": "L1",
            "trainer_kwargs/precision": 32,
            "input/transforms_cpu": [{"class": "ToType", "dtype": "float32"}],
        })
        sample2 = DatasetImage([path], train=True, config=config)[0]

        config = Config({"input/transforms_cpu": [{"class": "Normalization"}]})
        sample3 = DatasetImage([path], train=True, config=config)[0]

        # Samples are not exactly equal due to precision errors
        assert samples_equal(sample1, sample2, atol=1e-06, rtol=1e-03) and samples_equal(sample1, sample3)

    def test_dtype(self) -> None:
        t = Normalization()

        sample = {"features": torch.tensor([1, 2], dtype=torch.float32)}
        assert t(sample)["features"].dtype == torch.float32

        sample = {"features": torch.tensor([1, 2], dtype=torch.float16)}
        assert t(sample)["features"].dtype == torch.float16


class TestTransformPCA:
    def test_basics(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        path = DataPath.from_image_name("S001#2022_10_24_13_49_45")
        config = Config({
            "input/preprocessing": "L1_recalibrated",
            "input/data_spec": "sepsis-inclusion_palm_5folds_test-0.25_seed-0.json",
        })
        sample_original = DatasetImage([path], train=True, config=config, fold_name="fold_0")[0]
        config["input/transforms_cpu"] = [
            {"class": "TransformPCA", "n_components": 3},
            {"class": "ToType", "dtype": "float16"},
        ]
        sample_pca = DatasetImage([path], train=True, config=config, fold_name="fold_0")[0]

        assert torch.all(sample_original["labels"] == sample_pca["labels"])
        assert sample_original["features"].size(-1) == 100
        assert sample_pca["features"].size(-1) == 3
        assert not torch.allclose(sample_original["features"][..., :3], sample_pca["features"])


class TestStandardNormalVariate:
    def test_basics(self) -> None:
        path = DataPath.from_image_name("P058#2020_05_13_18_09_26")
        config = Config({
            "input/normalization": "L1",
            "input/transforms_cpu": [{"class": "StandardNormalVariate"}],
        })
        sample = DatasetImage([path], train=True, config=config)[0]

        assert torch.allclose(sample["features"].std(dim=-1), torch.ones(1))
        assert torch.allclose(sample["features"].mean(dim=-1), torch.zeros(1), atol=1e-05)


class TestToType:
    def test_conversion(self) -> None:
        t = ToType(dtype=torch.float16)

        sample = {
            "features": torch.tensor([1, 2], dtype=torch.float32),
            "features2": torch.tensor([1, 2], dtype=torch.float16),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        }
        sample_t = t(sample)
        assert sample_t["features"].dtype == torch.float16
        assert sample_t["features2"].dtype == torch.float16
        assert sample_t["labels"].dtype == torch.int64


@pytest.mark.filterwarnings(
    r"ignore:Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0\.:UserWarning"
)
class TestKorniaTransform:
    def test_labels_interpolation(self) -> None:
        config = Config({"input/n_channels": 3})
        dataset = DatasetImage([DataPath.from_image_name("P058#2020_05_13_18_09_26")], train=False, config=config)
        sample = dataset[0]

        img = sample["features"].permute(2, 0, 1).unsqueeze(dim=0)
        labels = sample["labels"].unsqueeze(dim=0).unsqueeze(dim=0)
        labels_unique = labels.unique()

        aug = K.AugmentationSequential(
            K.RandomAffine((20, 20), padding_mode="reflection", p=1), data_keys=["input", "mask"]
        )
        img_transformed, labels_transformed = aug(img, labels.float())

        assert not torch.all(img_transformed == img)
        assert len(labels_unique) == len(labels_transformed.unique())
        assert torch.all(labels_unique == labels_transformed.int().unique())

    @pytest.mark.parametrize(
        "config_kornia",
        [
            # Rotation
            Config({
                "input/transforms_cpu": [
                    {
                        "class": "KorniaTransform",
                        "p": 1,
                        "transformation_name": "RandomAffine",
                        "degrees": [45, 45],
                        "padding_mode": "reflection",
                    }
                ]
            }),
            # Scale
            Config({
                "input/transforms_cpu": [
                    {
                        "class": "KorniaTransform",
                        "p": 1,
                        "transformation_name": "RandomAffine",
                        "degrees": 0,
                        "scale": [1.1, 1.1],
                        "padding_mode": "reflection",
                    }
                ]
            }),
            # Horizontal flip
            Config({
                "input/transforms_cpu": [
                    {
                        "class": "KorniaTransform",
                        "p": 1,
                        "transformation_name": "RandomHorizontalFlip",
                    }
                ]
            }),
            # Vertical flip
            Config({
                "input/transforms_cpu": [
                    {
                        "class": "KorniaTransform",
                        "p": 1,
                        "transformation_name": "RandomVerticalFlip",
                    }
                ]
            }),
            # Elastic transformation
            Config({
                "input/transforms_cpu": [
                    {
                        "class": "KorniaTransform",
                        "p": 1,
                        "transformation_name": "RandomElasticTransform",
                        "padding_mode": "reflection",
                        "alpha": [0.7, 0.7],
                        "sigma": [16, 16],
                    }
                ]
            }),
        ],
    )
    def test_affine_applied(self, config_kornia: Config) -> None:
        for dtype in [torch.float16, torch.float32]:
            path = DataPath.from_image_name("P058#2020_05_13_18_09_26")
            config = Config({"input/n_channels": 3, "input/transforms_cpu": [{"class": "ToType", "dtype": dtype}]})
            dataset = DatasetImage([path], train=True, config=config)
            sample = dataset[0]

            config_kornia_type = copy.copy(config_kornia)
            config_kornia_type["input/n_channels"] = 3
            config_kornia_type["input/transforms_cpu"].insert(0, {"class": "ToType", "dtype": dtype})
            sample_kornia = DatasetImage([path], train=True, config=config_kornia_type)[0]
            assert sample["features"].dtype == sample_kornia["features"].dtype == dtype
            assert not torch.all(sample_kornia["features"] == sample["features"])
            assert not torch.all(sample_kornia["labels"] == sample["labels"])
            assert torch.all(sample_kornia["labels"].unique() == sample["labels"].unique())

    @pytest.mark.serial
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_gpu(self, dtype: torch.dtype) -> None:
        config = Config({
            "input/n_channels": 3,
            "input/transforms_cpu": [
                {
                    "class": "ToType",
                    "dtype": dtype,
                },
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                },
            ],
        })
        dataset = DatasetImage([DataPath.from_image_name("P058#2020_05_13_18_09_26")], train=False, config=config)
        dataloader = DataLoader(dataset, sampler=[0, 0], batch_size=2)

        batch_original = next(iter(dataloader))
        batch = move_batch_gpu(batch_original)

        aug_kornia = HTCTransformation.parse_transforms(config["input/transforms_cpu"], config=config)
        batch_transformed = HTCTransformation.apply_valid_transforms(batch, aug_kornia)

        assert batch_transformed["features"].shape == batch["features"].shape
        assert batch_transformed["labels"].shape == batch["labels"].shape
        assert batch_transformed["valid_pixels"].shape == batch["valid_pixels"].shape

        assert batch_transformed["features"].dtype == batch["features"].dtype == dtype
        assert batch_transformed["labels"].dtype == batch["labels"].dtype
        assert batch_transformed["valid_pixels"].dtype == batch["valid_pixels"].dtype

        assert batch_transformed["features"].size(0) == 2
        assert batch_transformed["labels"].size(0) == 2
        torch.all(batch_transformed["labels"].cpu() == hflip(batch_original["labels"]))

    def test_multi_layered_segmentations(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        config = Config({
            "input/n_channels": 3,
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
            "label_mapping": None,
        })
        dataset = DatasetImage(
            [DataPath.from_image_name("sepsis#JG955#t_7-post_admission_day_2_visit_1#leg_upper#2019_09_18_09_21_28")],
            train=False,
            config=config,
        )
        dataloader = DataLoader(dataset, sampler=[0], batch_size=1)

        batch = next(iter(dataloader))
        aug_kornia = HTCTransformation.parse_transforms(config["input/transforms_cpu"], config=config)
        batch_transformed = HTCTransformation.apply_valid_transforms(batch, aug_kornia)
        assert batch_transformed["labels"].shape == batch["labels"].shape
        assert batch_transformed["valid_pixels"].shape == batch["valid_pixels"].shape

        # Same transformation applied for all channels
        features = torch.ones(1, 10, 10, 2)
        features[0, 2, 2, :] = 0
        labels = torch.ones(1, 2, 10, 10, dtype=torch.int64)
        labels[0, :, 2, 2] = 0

        batch = {
            "features": features,
            "labels": labels,
        }
        batch_transformed = HTCTransformation.apply_valid_transforms(batch, aug_kornia)

        assert torch.all(batch_transformed["features"].permute(0, 3, 1, 2) == batch_transformed["labels"])
        assert torch.all(batch_transformed["labels"][0, 0, :, :] == batch_transformed["labels"][0, 1, :, :])

        labels_transformed = torch.ones(1, 2, 10, 10, dtype=torch.int64)
        labels_transformed[0, :, 2, -3] = 0
        assert torch.all(batch_transformed["labels"] == labels_transformed)

    def test_multiple_annotations(self) -> None:
        config = Config({
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })

        aug = HTCTransformation.parse_transforms(config["input/transforms_cpu"])
        x = torch.ones(1, 10, 10, 3)
        x[0, 3:, 1:4, :] = 0
        x_flip = hflip(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        y1 = torch.ones(1, 10, 10, dtype=torch.int64)
        y1[0, 3:, 1:4] = 0
        y1_flip = hflip(y1)
        y2 = torch.ones(1, 10, 10, dtype=torch.int64)
        y2[0, 3:, 2:4] = 0
        y2_flip = hflip(y2)

        batch = {
            "features": x,
            "labels_a1": y1,
            "labels_a2": y2,
        }
        batch_transformed = HTCTransformation.apply_valid_transforms(batch, aug)
        assert torch.all(batch_transformed["features"] == x_flip)
        assert torch.all(batch_transformed["labels_a1"] == y1_flip)
        assert torch.all(batch_transformed["labels_a2"] == y2_flip)
        assert not torch.all(batch_transformed["labels_a1"] == batch_transformed["labels_a2"])

    def test_unknown_key(self) -> None:
        config = Config({
            "input/n_channels": 3,
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })

        aug_kornia = HTCTransformation.parse_transforms(config["input/transforms_cpu"])
        with pytest.raises(ValueError, match="no type associated"):
            batch = {"features": torch.zeros(1, 1, 1, 1), "unknown": torch.zeros(1, 1, 1)}
            HTCTransformation.apply_valid_transforms(batch, aug_kornia)

    def test_features_flip(self) -> None:
        config = Config({
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })

        aug = HTCTransformation.parse_transforms(config["input/transforms_cpu"])
        x = torch.ones(1, 10, 10, 3)
        x[0, 3:, 1:4, :] = 0
        x_flip = hflip(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        batch = {"features": x, "image_name": ["img"]}
        batch_transformed = HTCTransformation.apply_valid_transforms(batch, aug)
        assert torch.all(batch_transformed["features"] == x_flip)
