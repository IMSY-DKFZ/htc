# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io
from collections.abc import Callable

import numpy as np
import torch
from kornia.geometry.transform.flips import hflip
from lightning import seed_everything
from torch.utils.data import DataLoader

from htc.models.common.transforms import KorniaTransform, Normalization, ToType
from htc.models.common.utils import samples_equal
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping


class TestDatasetImage:
    def test_spxs_loading(self) -> None:
        # Check whether the dataset can load the superpixels for the example image properly
        config = Config({"input/superpixels/n_segments": 1000, "input/superpixels/compactness": 10})
        dataset = DatasetImage.example_dataset(config)
        sample = dataset[10]

        assert "spxs" in sample
        assert sample["spxs"].shape == sample["features"].shape[:2] == sample["labels"].shape
        assert sample["spxs"].dtype == torch.int64

    def test_reproducible_ordering(self) -> None:
        def load_samples() -> list[int]:
            seed_everything(settings.default_seed, workers=True)
            dataset = DatasetImage.example_dataset()
            dataloader = DataLoader(dataset, persistent_workers=True, batch_size=2, num_workers=4)

            img_indices = []
            i = 0
            loader_it = iter(dataloader)
            for batch in loader_it:
                img_indices += batch["image_index"].tolist()
                i += 1
                if i >= 5:
                    break

            loader_it = iter(dataloader)
            for batch in loader_it:
                img_indices += batch["image_index"].tolist()
                i += 1
                if i >= 8:
                    break

            return img_indices

        img_indices = load_samples()
        img_indices2 = load_samples()
        assert img_indices == img_indices2, (
            "The same indices should be returned when calling the dataloader multiple times"
        )

    def test_l1_preprocessing(self) -> None:
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        sample_raw = DatasetImage(
            [path], train=False, config=Config({"trainer_kwargs/precision": "16-mixed", "input/normalization": "L1"})
        )[0]
        sample_preprocessed = DatasetImage(
            [path], train=False, config=Config({"trainer_kwargs/precision": "16-mixed", "input/preprocessing": "L1"})
        )[0]
        preprocessing_folder = path.intermediates_dir / "preprocessing" / "L1"
        sample_preprocessed2 = DatasetImage(
            [path],
            train=False,
            config=Config({"trainer_kwargs/precision": "16-mixed", "input/preprocessing": str(preprocessing_folder)}),
        )[0]

        assert samples_equal(sample_raw, sample_preprocessed)
        assert samples_equal(sample_raw, sample_preprocessed2)
        assert sample_raw["features"].dtype == torch.float16

    def test_label_mapping_dataset(self) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.semantic))
        mapping = LabelMapping.from_path(paths[0])

        # The mapping of the data path equals no mapping (i.e. raw data on disk)
        dataset = DatasetImage(paths, train=False, config=Config({"label_mapping": mapping, "input/no_features": True}))
        for sample in dataset:
            path = DataPath.from_image_name(sample["image_name_annotations"])
            seg = path.read_segmentation()
            assert np.all(sample["labels"].numpy() == seg)

    def test_label_counts(self) -> None:
        specs_json = """
        [
            {
                "fold_name": "fold_1",
                "train": {
                    "image_names": ["P044#2020_02_01_09_51_15", "P045#2020_02_05_10_57_43"]
                }
            },
            {
                "fold_name": "fold_2",
                "train": {
                    "image_names": ["P049#2020_02_11_19_09_49"]
                }
            }
        ]
        """
        specs = DataSpecification(io.StringIO(specs_json))
        config = Config({"input/data_spec": specs, "label_mapping": settings_seg.label_mapping})
        fold_name = specs.fold_names()[0]
        train_paths = specs.folds[fold_name]["train"]
        dataset = DatasetImage(train_paths, train=False, config=config, fold_name=fold_name)

        # Manually collect the information from the example images
        counts_images = {}
        for sample in dataset:
            label_indices, label_counts = sample["labels"][sample["valid_pixels"]].unique(return_counts=True)
            for l, c in zip(label_indices, label_counts, strict=True):
                l = l.item()
                c = c.item()

                if l not in counts_images:
                    counts_images[l] = 0

                counts_images[l] += c

        counts_image = dict(sorted(counts_images.items()))

        label_indices, label_counts = dataset.label_counts()
        assert list(counts_image.keys()) == label_indices.tolist()
        assert list(counts_image.values()) == label_counts.tolist()

    def test_label_counts_image(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        config = Config("htc_projects/sepsis_icu/configs/sepsis-inclusion_palm_image.json")
        spec = DataSpecification.from_config(config)
        fold_name = spec.fold_names()[0]
        paths = spec.fold_paths(fold_name, "^train")
        dataset = DatasetImage(paths, train=False, config=config, fold_name=fold_name)
        label_indices, label_counts = dataset.label_counts()

        df = median_table(
            paths=paths,
            additional_mappings={"sepsis_status": LabelMapping.from_config(config)},
        )
        df = df.groupby("sepsis_status_index", as_index=False)["image_name"].nunique()
        assert label_indices.tolist() == df["sepsis_status_index"].tolist()
        assert label_counts.tolist() == df["image_name"].tolist()

    def test_channel_selection(self) -> None:
        config = Config({
            "input/normalization": "L1",
            "input/n_channels": 100,
            "input/channel_selection": [10, 90],
        })
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        dataset = DatasetImage([path], train=False, config=config)
        sample = dataset[0]

        assert sample["features"].shape[:2] == path.dataset_settings["spatial_shape"]
        assert sample["features"].size(-1) == 80
        assert config["input/n_channels"] == 80
        assert dataset.n_channels_loading == 100

    def test_key_selection(self) -> None:
        config = Config({
            "input/normalization": "L1",
            "input/n_channels": 3,
        })
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        sample = DatasetImage([path], train=False, config=config)[0]
        assert "features" in sample and "labels" in sample and "valid_pixels" in sample

        config["input/no_labels"] = True
        sample = DatasetImage([path], train=False, config=config)[0]
        assert "features" in sample and "labels" not in sample and "valid_pixels" not in sample

        config["input/no_labels"] = False
        config["input/no_features"] = True
        sample = DatasetImage([path], train=False, config=config)[0]
        assert "features" not in sample and "labels" in sample and "valid_pixels" in sample

    def test_preprocessing_additional(self) -> None:
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        config = Config({
            "input/preprocessing": "L1",
            "input/preprocessing_additional": [{"name": "L1"}, {"name": "parameter_images"}],
            "input/n_channels": 100,
        })
        sample = DatasetImage([path], train=True, config=config)[0]

        assert "data_L1" in sample and "data_parameter_images" in sample
        assert torch.all(sample["features"] == sample["data_L1"])
        assert sample["data_parameter_images"].shape == (480, 640, 4)

        # Transformations must also be applied to the additional images
        config = Config({
            "input/preprocessing": "L1",
            "input/preprocessing_additional": [{"name": "L1"}, {"name": "parameter_images"}],
            "input/n_channels": 100,
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })
        sample_transformed = DatasetImage([path], train=True, config=config)[0]
        assert sample_transformed["data_parameter_images"].shape == (480, 640, 4)
        assert torch.all(sample_transformed["features"] == sample_transformed["data_L1"])

        for key in ["features", "data_L1", "data_parameter_images"]:
            flipped = hflip(sample[key].permute(2, 0, 1)).permute(1, 2, 0)
            assert torch.all(sample_transformed[key] == flipped)

    def test_transforms(self) -> None:
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")

        config = Config({
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })
        dataset = DatasetImage([path], train=True, config=config)
        dataset[0]
        t = dataset.transforms
        assert len(t) == 2
        assert type(t[0]) == ToType
        assert type(t[1]) == KorniaTransform

        config = Config({
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                },
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomVerticalFlip",
                    "p": 0.25,
                },
            ],
        })
        dataset = DatasetImage([path], train=True, config=config)
        t = dataset.transforms
        assert len(t) == 2
        assert type(t[0]) == ToType
        assert type(t[1]) == KorniaTransform

        config = Config({
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                },
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomVerticalFlip",
                    "p": 0.25,
                },
                {
                    "class": "Normalization",
                    "order": 1,
                },
            ],
        })
        dataset = DatasetImage([path], train=True, config=config)
        t = dataset.transforms
        assert len(t) == 3
        assert type(t[0]) == ToType and t[0].dtype == torch.float32
        assert type(t[1]) == KorniaTransform
        assert type(t[2]) == Normalization and t[2].order == 1

        # A bit contrived, but theoretically possible
        config = Config({
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                },
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomVerticalFlip",
                    "p": 0.25,
                },
                {
                    "class": "Normalization",
                    "order": 1,
                },
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                },
            ],
        })
        dataset = DatasetImage([path], train=True, config=config)
        t = dataset.transforms
        assert len(t) == 4
        assert type(t[0]) == ToType and t[0].dtype == torch.float32
        assert type(t[1]) == KorniaTransform
        assert type(t[2]) == Normalization and t[2].order == 1
        assert type(t[3]) == KorniaTransform

    def test_annotation_name(self) -> None:
        path = DataPath.from_image_name("P091#2021_04_24_12_02_50@polygon#annotator2")

        config = Config({
            "label_mapping": None,
        })
        sample = DatasetImage([path], train=False, config=config)[0]
        assert np.all(path.read_segmentation("polygon#annotator2") == sample["labels"].numpy())

        config = Config({
            "label_mapping": None,
            "input/annotation_name": "polygon#annotator1",
        })
        sample = DatasetImage([path], train=False, config=config)[0]
        assert np.all(path.read_segmentation("polygon#annotator1") == sample["labels"].numpy())

        config["input/annotation_name"] = "all"
        sample = DatasetImage([path], train=False, config=config)[0]
        assert not torch.all(sample["labels_polygon#annotator1"] == sample["labels_polygon#annotator2"])
        assert not torch.all(sample["labels_polygon#annotator1"] == sample["labels_polygon#annotator3"])
        assert not torch.all(sample["labels_polygon#annotator2"] == sample["labels_polygon#annotator3"])

        config = Config({
            "label_mapping": None,
            "input/annotation_name": "all",
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })
        sample_transformed = DatasetImage([path], train=True, config=config)[0]
        for key, value in sample.items():
            if type(value) == torch.Tensor:
                if value.ndim == 3:
                    flipped = hflip(value.permute(2, 0, 1)).permute(1, 2, 0)
                else:
                    flipped = hflip(value)
                assert torch.all(sample_transformed[key] == flipped)
            else:
                assert key in ["image_name", "image_name_annotations", "image_index"]

    def test_annotation_name_combination(self) -> None:
        path = DataPath.from_image_name("P091#2021_04_24_12_02_50")
        mapping = LabelMapping.from_path(path)

        config = Config({
            "input/annotation_name": "all",
            "input/merge_annotations": "union",
        })
        sample = DatasetImage([path], train=False, config=config)[0]
        assert "labels" in sample and "valid_pixels" in sample

        for name, seg in path.read_segmentation("all").items():
            assert f"labels_{name}" not in sample
            assert f"valid_pixels_{name}" not in sample

            assert sample["valid_pixels"].sum() > np.sum(mapping.is_index_valid(seg))

    def test_image_label(self, check_sepsis_ICU_data_accessible: Callable) -> None:
        check_sepsis_ICU_data_accessible()

        path = DataPath.from_image_name("S438#2023_10_02_19_23_03")  # No sepsis, no septic shock
        config = Config({
            "task": "classification",
            "input/no_labels": True,
            "input/image_labels": [
                {
                    "meta_attributes": ["sepsis_status"],
                    "image_label_mapping": {
                        "sepsis": 10,
                        "no_sepsis": 20,
                    },
                }
            ],
        })

        sample = DatasetImage([path], train=False, config=config)[0]
        assert sample["image_labels"] == 20
        assert sample["image_labels"].dtype == torch.int64

        config["input/image_labels"].append({
            "meta_attributes": ["septic_shock"],
            # septic_shock is a boolean column, so we don't need to specify a mapping
        })

        sample = DatasetImage([path], train=False, config=config)[0]
        assert torch.all(sample["image_labels"] == torch.tensor([20, 0], dtype=torch.int64))
