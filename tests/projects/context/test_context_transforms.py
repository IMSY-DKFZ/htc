# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy

import pytest
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader

from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.common.transforms import HTCTransformation
from htc.models.common.utils import samples_equal
from htc.models.image.DatasetImage import DatasetImage
from htc.models.superpixel_classification.DatasetSuperpixelImage import DatasetSuperpixelImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc_projects.context.context_transforms import (
    HideAndSeek,
    OrganIsolation,
    OrganRemoval,
    OrganTransplantation,
    RandomJigsaw,
    RandomMixUp,
    RandomSuperpixelErasing,
    RectangleOrganTransplantation,
    SuperpixelOrganTransplantation,
    ValidPixelsOnly,
)


class TestOrganRemoval:
    def test_class_removal(self) -> None:
        # The number of zeros in the features of the transformed sample has to be the same as
        # the number of zeros in the old sample + the number of times that the label to black out appears in the label tensor
        # Load data
        paths = [DataPath.from_image_name("P044#2020_02_01_09_51_15")]
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/n_channels": 100,
            "input/preprocessing": "L1",
        })
        dataset = DatasetImage(paths, train=False, config=config)
        dataloader = DataLoader(dataset)

        # Test-time transformation directly applied during data loading
        config_ttt = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/n_channels": 100,
            "input/preprocessing": "L1",
            "input/test_time_transforms_cpu": [
                {
                    "class": "htc_projects.context.context_transforms>OrganRemoval",
                    "fill_value": "0",
                    "target_label": 4,
                }
            ],
        })
        dataset_ttt = DatasetImage(paths, train=False, config=config_ttt)
        dataloader_ttt = DataLoader(dataset_ttt)

        # Choose a label to black out
        target_label = 4

        # Transform the data
        transform = OrganRemoval(fill_value="0", target_label=target_label)
        for sample, sample_ttt in zip(dataloader, dataloader_ttt, strict=True):
            # (n_features == 0) + n_labels_to_black_out
            features = sample["features"]
            labels = sample["labels"]

            n_features_zeros = 0
            n_labels_to_black_out = 0
            zeros_across_a_dim_original = torch.count_nonzero(features, dim=3)

            if 0 in torch.count_nonzero(features, dim=3):
                _, counts = torch.unique(zeros_across_a_dim_original, return_counts=True)
                n_features_zeros = sum(torch.count_nonzero(features, dim=3) == 0)

            if target_label in labels:
                _, n_labels_to_black_out = torch.unique(labels[labels == target_label], return_counts=True)
                n_labels_to_black_out = n_labels_to_black_out.item()

            n_zeros = n_features_zeros + n_labels_to_black_out

            # Number of blacked out (all feature weights are 0),  in the transformed sample
            assert target_label in sample["labels"]
            sample_transformed = transform(sample)
            assert target_label not in sample_transformed["labels"]

            sample_transformed_features = sample_transformed["features"]
            n_blacked_out_labels = 0
            zeros_across_a_dim = torch.count_nonzero(sample_transformed_features, dim=3)

            if 0 in zeros_across_a_dim:
                _, counts = torch.unique(zeros_across_a_dim, return_counts=True)
                n_blacked_out_labels = counts[0]
            assert n_zeros == n_blacked_out_labels

            # Manually applying the transformation must be identical to the config approach
            assert sample.keys() == sample_ttt.keys()
            for key in sample.keys():
                if type(sample[key]) == torch.Tensor:
                    assert torch.all(sample[key] == sample_ttt[key])
                    assert sample[key].data_ptr() != sample_ttt[key].data_ptr(), "Different memory objects must be used"
                else:
                    assert sample[key] == sample_ttt[key]

    def test_class_removal_valid_pixels(self) -> None:
        # The number of new invalid pixels has to be the same as the number of blacked out labels
        paths = [DataPath.from_image_name("P044#2020_02_01_09_51_15")]
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/no_features": True,
            "input/preprocessing": "L1",
        })
        dataset = DatasetImage(paths, train=False, config=config)
        dataloader = DataLoader(dataset)

        # Number of invalid pixels of the picture before ttt
        for sample in dataloader:
            _, counts = torch.unique(sample["valid_pixels"], return_counts=True)
            n_false_pixels_init = 0
            if False in _:
                n_false_pixels_init = counts[0]

        config_ttt = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/n_channels": 3,
            "input/test_time_transforms_cpu": [
                {
                    "class": "htc_projects.context.context_transforms>OrganRemoval",
                    "fill_value": "0",
                    "target_label": 4,
                }
            ],
        })
        dataset_ttt = DatasetImage(paths, train=False, config=config_ttt)
        dataloader_ttt = DataLoader(dataset_ttt)

        for sample_transformed in dataloader_ttt:
            # Number of invalid pixels after image labels have been blacked out
            _, counts = torch.unique(sample_transformed["valid_pixels"], return_counts=True)
            n_false_pixels_transformed = 0
            if False in _:
                n_false_pixels_transformed = counts[0]

            # Number of labels that have been changed
            sample_transformed_features = sample_transformed["features"]
            n_blacked_out_labels = 0
            zeros_across_a_dim = torch.count_nonzero(sample_transformed_features, dim=3)

            if 0 in zeros_across_a_dim:
                _, counts = torch.unique(zeros_across_a_dim, return_counts=True)
                n_blacked_out_labels = counts[0]

            assert n_blacked_out_labels == n_false_pixels_transformed - n_false_pixels_init

    def test_cloth_filling_image(self) -> None:
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        mapping = settings_seg.label_mapping
        config = Config({"label_mapping": mapping, "input/n_channels": 3})
        sample_original = DatasetImage([path], train=False, config=config)[0]
        assert mapping.name_to_index("background") in sample_original["labels"]

        config_ttt = Config({
            "label_mapping": mapping,
            "input/n_channels": 3,
            "input/test_time_transforms_cpu": [
                {
                    "class": "htc_projects.context.context_transforms>OrganRemoval",
                    "fill_value": "cloth",
                    "target_label": 0,
                }
            ],
        })
        dataset = DatasetImage([path], train=False, config=config_ttt)
        sample_transformed = dataset[0]
        assert mapping.name_to_index("background") not in sample_transformed["labels"]

        cloth_features = dataset.transforms[1]._filling_sample
        selection = sample_original["labels"] == 0
        assert torch.all(sample_transformed["features"][selection] == cloth_features[selection])

    def test_cloth_filling_full(self) -> None:
        t = OrganRemoval(fill_value="cloth", target_label=0, config=Config({"input/n_channels": 3}))

        sample = {
            "features": torch.ones((480, 640, 3), dtype=torch.float32),
            "labels": torch.zeros((480, 640), dtype=torch.int64),
            "valid_pixels": torch.ones((480, 640), dtype=torch.bool),
        }

        sample = t(sample)
        assert t._filling_sample is not None
        assert torch.all(sample["labels"] == settings.label_index_thresh)
        assert not torch.all(sample["valid_pixels"])
        assert torch.all(sample["features"] == t._filling_sample)


class TestOrganIsolation:
    def test_zero_filling(self) -> None:
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        mapping = settings_seg.label_mapping
        config = Config({
            "label_mapping": mapping,
            "input/n_channels": 3,
            "input/transforms_cpu": [
                {
                    "class": "htc_projects.context.context_transforms>OrganIsolation",
                    "fill_value": "0",
                    "target_label": 3,
                }
            ],
        })

        sample = DatasetImage([path], train=True, config=config)[0]
        selection = sample["labels"] == 3
        assert torch.all(sample["labels"].unique() == torch.tensor([3, 100]))
        assert not torch.any(sample["valid_pixels"][~selection])
        assert torch.all(sample["features"][~selection] == 0)

    def test_cloth_filling(self) -> None:
        t = OrganIsolation(fill_value="cloth", target_label=0, config=Config({"input/n_channels": 3}))

        sample = {
            "features": torch.ones((480, 640, 3), dtype=torch.float32),
            "labels": torch.zeros((480, 640), dtype=torch.int64),
            "valid_pixels": torch.ones((480, 640), dtype=torch.bool),
        }

        sample_backup = copy.deepcopy(sample)
        sample = t(sample)
        assert t._filling_sample is not None
        assert samples_equal(sample, sample_backup), "Target label is the only label, nothing should change"

        t = OrganIsolation(fill_value="cloth", target_label=1, config=Config({"input/n_channels": 3}))
        sample = t(sample)
        assert torch.all(sample["features"] == t._filling_sample)
        assert torch.all(sample["labels"] == settings.label_index_thresh)
        assert not torch.any(sample["valid_pixels"])

    def test_random_filling(self) -> None:
        seed_everything(0, workers=True)
        t = OrganIsolation(fill_value="random_uniform", target_label=1)

        labels = torch.ones((4, 4), dtype=torch.int64)
        labels[1, 1] = 0

        sample = {
            "features": torch.ones((4, 4, 3), dtype=torch.float32),
            "labels": labels,
            "valid_pixels": torch.ones((4, 4), dtype=torch.bool),
        }

        sample_backup = copy.deepcopy(sample)
        sample = t(sample)
        assert all(sample["features"][1, 1] != sample_backup["features"][1, 1])
        assert torch.allclose(sample["features"][1, 1].abs().sum(), torch.tensor(1, dtype=torch.float32))
        assert torch.all(sample["features"][sample["labels"] == 1] == sample_backup["features"][sample["labels"] == 1])

    def test_random(self) -> None:
        seed_everything(0, workers=True)

        path1 = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        path2 = DataPath.from_image_name("P058#2020_05_13_18_09_26")
        mapping = settings_seg.label_mapping
        config = Config({"label_mapping": mapping, "input/n_channels": 3})

        dataset = DatasetImage([path1, path2], train=True, config=config)
        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))

        t = OrganIsolation(fill_value="0", target_label="random")
        batch_transformed = t(batch)

        labels0 = batch_transformed["labels"][0].unique()
        labels1 = batch_transformed["labels"][1].unique()
        assert len(labels0) == len(labels1) == 2

        # 11 and 0 are the randomly selected labels
        assert labels0[0] == 11
        assert labels1[0] == 0

        assert batch_transformed["valid_pixels"][0][batch_transformed["labels"][0] == 11].all()
        assert not batch_transformed["valid_pixels"][0][batch_transformed["labels"][0] != 11].any()
        assert batch_transformed["valid_pixels"][1][batch_transformed["labels"][1] == 0].all()
        assert not batch_transformed["valid_pixels"][1][batch_transformed["labels"][1] != 0].any()

    def test_p(self) -> None:
        seed_everything(0, workers=True)
        t = OrganIsolation(fill_value="0", target_label=1, p=0.5)

        labels = torch.ones((2, 4, 4), dtype=torch.int64)
        labels[0, 1, 1] = 0
        labels[1, 1, 1] = 0

        sample = {
            "features": torch.ones((2, 4, 4, 3), dtype=torch.float32),
            "labels": labels,
            "valid_pixels": torch.ones((2, 4, 4), dtype=torch.bool),
        }

        sample_copy = copy.deepcopy(sample)
        sample = t(sample)

        # With the seed, the transformation gets only applied to the first image
        assert torch.all(sample["features"][0, 1, 1] == 0)
        assert torch.all(sample["features"][1] == 1)

        t = OrganIsolation(fill_value="0", target_label=1, p=0.1)
        applied = []
        for _ in range(100):
            sample = t(copy.deepcopy(sample_copy))
            applied.append(torch.all(sample["features"][:, 1, 1] == 0, dim=-1))
        n_applied = torch.cat(applied).sum()
        assert 0.05 <= n_applied / 200 <= 0.15

    def test_superpixel(self) -> None:
        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        mapping = settings_seg.label_mapping
        config_rgb = Config({
            "label_mapping": mapping,
            "input/n_channels": 3,
            "input/superpixels": {"n_segments": 1000, "compactness": 10},
            "input/resize_shape": [32, 32],
            "input/test_time_transforms_cpu": [
                {
                    "class": "htc_projects.context.context_transforms>OrganIsolation",
                    "fill_value": "0",
                    "target_label": mapping.name_to_index("colon"),
                }
            ],
        })

        sample_rgb = DatasetSuperpixelImage([path], train=False, config=config_rgb)[0]
        assert torch.all(sample_rgb["features"][0] == 0)
        assert not torch.all(sample_rgb["features"] == 0)

        config_hsi = copy.copy(config_rgb)
        config_hsi["input/n_channels"] = 100

        sample_hsi = DatasetSuperpixelImage([path], train=False, config=config_hsi)[0]
        assert torch.all(sample_hsi["features"][0] == 0)
        assert not torch.all(sample_hsi["features"] == 0)

        # Same superpixels for RGB and HSI
        assert torch.all(sample_rgb["spxs_sizes"] == sample_hsi["spxs_sizes"])
        assert torch.all(sample_rgb["spxs_indices_rows"] == sample_hsi["spxs_indices_rows"])
        assert torch.all(sample_rgb["spxs_indices_cols"] == sample_hsi["spxs_indices_cols"])


class TestValidPixelsOnly:
    def test_basics(self) -> None:
        seed_everything(0, workers=True)
        sample = {
            "features": torch.rand(4, 4),
            "labels": torch.arange(0, 16, dtype=torch.int64).reshape(4, 4),
            "valid_pixels": torch.zeros(4, 4, dtype=torch.bool),
        }

        t = ValidPixelsOnly(fill_value="0")
        sample = t(sample)
        assert t.is_applied(sample)
        assert torch.all(sample["features"] == 0)
        assert torch.all(sample["labels"] == settings.label_index_thresh)
        assert not sample["valid_pixels"].any()

        sample = {
            "features": torch.rand(4, 4),
            "labels": torch.arange(0, 16, dtype=torch.int64).reshape(4, 4),
            "valid_pixels": torch.zeros(4, 4, dtype=torch.bool),
        }

        sample["valid_pixels"][0, 0] = True
        sample = t(sample)
        assert t.is_applied(sample)
        assert torch.all(sample["features"][1:, 1:] == 0) and sample["features"][0, 0] != 0
        assert torch.all(sample["labels"][1:, 1:] == settings.label_index_thresh) and sample["labels"][0, 0] == 0
        assert not sample["valid_pixels"][1:, 1:].any() and sample["valid_pixels"][0, 0]


class TestRandomRectangleErasing:
    def test_basics(self) -> None:
        seed_everything(0, workers=True)

        path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        config = Config({
            "input/n_channels": 100,
            "input/transforms_cpu": [
                {
                    "class": "htc_projects.context.context_transforms>RandomRectangleErasing",
                    "fill_value": "0",
                }
            ],
        })

        sample = DatasetImage([path], train=True, config=config)[0]

        # Make sure we really selected a rectangle
        rows, cols = torch.nonzero(sample["labels"] == settings.label_index_thresh, as_tuple=True)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        mask = torch.zeros(sample["labels"].shape, dtype=torch.bool)
        mask[min_row : max_row + 1, min_col : max_col + 1] = True

        assert torch.all(sample["labels"][mask] == settings.label_index_thresh)
        assert torch.all(sample["labels"][~mask] < settings.label_index_thresh)
        assert not sample["valid_pixels"][mask].any()
        assert sample["valid_pixels"][~mask].all()
        assert torch.all(sample["features"][mask] == 0)
        assert torch.all(sample["features"][~mask] != 0)


class TestRandomSuperpixelErasing:
    @pytest.mark.serial
    def test_superpixel_basics(self) -> None:
        seed_everything(0, workers=True)

        path1 = DataPath.from_image_name("P044#2020_02_01_09_51_15")
        path2 = DataPath.from_image_name("P058#2020_05_13_18_09_26")
        config = Config({
            "input/n_channels": 3,
            "input/superpixels/n_segments": 1000,
            "input/superpixels/slic_zero": True,
            "input/superpixels/compactness": 10,
            "input/superpixels/sigma": 3,
            "input/transforms_gpu": [
                {
                    "class": "htc_projects.context.context_transforms>RandomSuperpixelErasing",
                    "fill_value": "0",
                    "proportion": [0.1, 0.2],
                }
            ],
        })

        dataset = DatasetImage([path1, path2], train=True, config=config)
        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))

        batch = move_batch_gpu(batch)
        aug = HTCTransformation.parse_transforms(config["input/transforms_gpu"], config=config)
        batch = HTCTransformation.apply_valid_transforms(batch, aug)

        spxs_filled_mask = batch["features"] == 0  # Fill value
        assert True in spxs_filled_mask.unique()
        assert torch.all(batch["labels"][0][spxs_filled_mask[0, :, :, 0]] == settings.label_index_thresh)
        assert torch.all(batch["labels"][1][spxs_filled_mask[1, :, :, 0]] == settings.label_index_thresh)
        assert torch.all(batch["labels"][0][~spxs_filled_mask[0, :, :, 0]] < settings.label_index_thresh)
        assert torch.all(batch["labels"][1][~spxs_filled_mask[1, :, :, 0]] < settings.label_index_thresh)
        assert not batch["valid_pixels"][spxs_filled_mask[:, :, :, 0]].any()
        assert batch["valid_pixels"][~spxs_filled_mask[:, :, :, 0]].all()

    def test_batch(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
            "spxs": torch.ones(7, 10, 10, dtype=torch.int64),
        }
        batch["spxs"][:, :, 5:] = 0
        for i in range(7):
            batch["labels"][i] = i

        t = RandomSuperpixelErasing(p=1, fill_value="0", proportion=0.5)
        batch_transformed = t(batch)

        for i in range(7):
            assert torch.all(batch_transformed["labels"][i].unique() == torch.tensor([i, settings.label_index_thresh]))


class TestSuperpixelOrganTransplantation:
    def test_basics(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
            "spxs": torch.ones(7, 10, 10, dtype=torch.int64),
        }
        batch["spxs"][:, :, 5:] = 0
        for i in range(7):
            batch["labels"][i] = i

        t = SuperpixelOrganTransplantation(p=0.2, proportion=0.5)
        batch_transformed = t(batch)

        # From sample 0 to sample 6
        for i in range(6):
            assert torch.all(batch_transformed["labels"][i] == i)

        assert torch.all(batch_transformed["labels"][6].unique() == torch.tensor([0, 6]))

    def test_all(self) -> None:
        seed_everything(1, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
            "spxs": torch.ones(7, 10, 10, dtype=torch.int64),
        }
        batch["spxs"][:, :, 5:] = 0
        for i in range(7):
            batch["labels"][i] = i

        t = SuperpixelOrganTransplantation(p=1, proportion=0.5)
        batch_transformed = t(batch)

        assert all(len(batch_transformed["labels"][i].unique()) == 2 for i in range(7))


class TestRectangleOrganTransplantation:
    def test_basics(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
        }
        for i in range(7):
            batch["labels"][i] = i

        t = RectangleOrganTransplantation(p=0.2)
        batch_transformed = t(batch)

        # From sample 0 to sample 6
        for i in range(6):
            assert torch.all(batch_transformed["labels"][i] == i)

        assert torch.all(batch_transformed["labels"][6].unique() == torch.tensor([0, 6]))

    def test_all(self) -> None:
        seed_everything(1, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
        }
        for i in range(7):
            batch["labels"][i] = i

        t = RectangleOrganTransplantation(p=1)
        batch_transformed = t(batch)

        assert all(len(batch_transformed["labels"][i].unique()) == 2 for i in range(7))


class TestOrganTransplantation:
    def test_basics(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.randint(0, 6, (700,)).reshape(7, 10, 10),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
        }
        batch_copy = copy.deepcopy(batch)

        t = OrganTransplantation(p=0.2)
        batch_transformed = t(batch)

        donor_sample = 0
        acceptor_sample = 6
        selected_label = 4
        selection = batch_copy["labels"][donor_sample, :, :] == selected_label

        # Assert that donor sample is unchanged:
        assert torch.all(
            batch_transformed["features"][donor_sample, :, :, :] == batch_copy["features"][donor_sample, :, :, :]
        )
        assert torch.all(batch_transformed["labels"][donor_sample, :, :] == batch_copy["labels"][donor_sample, :, :])
        assert torch.all(
            batch_transformed["valid_pixels"][donor_sample, :, :] == batch_copy["valid_pixels"][donor_sample, :, :]
        )

        # Assert that acceptor sample is correctly transformed:
        assert torch.all(
            batch_transformed["features"][acceptor_sample, selection, :]
            == batch_copy["features"][donor_sample, selection, :]
        )
        assert torch.all(
            batch_transformed["features"][acceptor_sample, ~selection, :]
            == batch_copy["features"][acceptor_sample, ~selection, :]
        )

    def test_all(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(7, 10, 10, dtype=torch.bool),
        }
        for i in range(7):
            batch["labels"][i] = i

        t = OrganTransplantation(p=1)
        batch_transformed = t(batch)

        for i in range(7):
            assert torch.all(batch_transformed["labels"][i] == (i + 1) % 7)

    def test_additional_input(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(7, 10, 10, 3, dtype=torch.float32),
            "labels": torch.full((7, 10, 10), fill_value=settings.label_index_thresh, dtype=torch.int64),
            "labels_additional": torch.empty(7, 10, 10, dtype=torch.int64),
            "valid_pixels_additional": torch.ones(7, 10, 10, dtype=torch.bool),
            "regions": torch.empty(7, 10, 10, dtype=torch.int64),
        }
        batch["features_rgb"] = batch["features"].clone()
        batch["data_L1"] = batch["features"].clone()
        batch["data_parameter_images"] = batch["features"].clone()

        for i in range(7):
            batch["labels"][i, 0] = i
            batch["labels_additional"][i] = i + 1
            batch["regions"][i, 0] = 0
            batch["regions"][i, 1:] = 1
        batch["valid_pixels"] = batch["labels"] < settings.label_index_thresh

        t = OrganTransplantation(p=1)
        batch_transformed = t(batch)

        for i in range(7):
            assert torch.all(batch_transformed["labels"][i, 0] == (i + 1) % 7)
            assert torch.all(batch_transformed["labels"][i, 1:] == settings.label_index_thresh)
            assert torch.all(batch_transformed["labels_additional"][i, 0] == (i + 1) % 7 + 1), (
                "Transplanted area from labels"
            )
            assert torch.all(batch_transformed["labels_additional"][i, 1:] == i + 1), "Unchanged area"
            batch["regions"][i, 0] = 2
            batch["regions"][i, 1:] = 1

        assert torch.all(batch_transformed["features_rgb"] == batch["features"])
        assert torch.all(batch_transformed["data_L1"] == batch["features"])
        assert torch.all(batch_transformed["data_parameter_images"] == batch["features"])

    def test_annotation_names(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(4, 10, 10, 3, dtype=torch.float32),
            "labels": torch.empty(4, 10, 10, dtype=torch.int64),
            "valid_pixels": torch.ones(4, 10, 10, dtype=torch.bool),
            "image_index": torch.arange(4),
        }
        for i in range(4):
            batch["labels"][i] = i

        class DataPathTest:
            def __init__(self, annotation_name: str) -> None:
                self.annotation_name = annotation_name

            def annotation_names(self) -> list[str]:
                return [self.annotation_name]

        paths = [DataPathTest("name1"), DataPathTest("name2"), DataPathTest("name2"), DataPathTest("name1")]

        t = OrganTransplantation(p=1, annotation_names=["name1"], paths=paths)
        batch_transformed = t(batch)

        # image0 to image3, image3 to image 2
        # image1 and image2 are no donors
        assert torch.all(batch_transformed["labels"][0] == 0)
        assert torch.all(batch_transformed["labels"][1] == 1)
        assert torch.all(batch_transformed["labels"][2] == (2 + 1) % 4)
        assert torch.all(batch_transformed["labels"][3] == (3 + 1) % 4)


class TestRandomJigsaw:
    def test_basics(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.ones(4, 480, 640, 3, dtype=torch.float32),
            "labels": torch.empty(4, 480, 640, dtype=torch.int64),
            "valid_pixels": torch.ones(4, 480, 640, dtype=torch.bool),
        }
        for i in range(4):
            batch["features"][i] = i
            batch["labels"][i] = i
        assert torch.all(batch["labels"].unsqueeze(dim=-1).expand(batch["features"].shape) == batch["features"])

        t = RandomJigsaw(patch_size=[(60, 80)], p=0.5)
        batch = t(batch)

        assert torch.all(batch["labels"][0].unique() == torch.tensor([0, 1]))
        assert torch.all(batch["labels"][1].unique() == torch.tensor([0, 1]))
        assert torch.all(batch["labels"][2].unique() == torch.tensor([2]))
        assert torch.all(batch["labels"][3].unique() == torch.tensor([3]))
        assert torch.all(batch["labels"].unsqueeze(dim=-1).expand(batch["features"].shape) == batch["features"])

        assert batch["features"].dtype == torch.float32
        assert batch["labels"].dtype == torch.int64
        assert batch["valid_pixels"].dtype == torch.bool


class TestHideAndSeek:
    def test_batch(self) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.rand(4, 480, 640, 3, dtype=torch.float32),
            "labels": torch.empty(4, 480, 640, dtype=torch.int64),
            "valid_pixels": torch.ones(4, 480, 640, dtype=torch.bool),
        }
        for i in range(4):
            batch["labels"][i] = i

        t = HideAndSeek(p=1, fill_value="0", patch_size=[(60, 80)], proportion=[0.2, 0.8])
        batch_transformed = t(batch)

        for i in range(4):
            assert torch.all(batch_transformed["labels"][i].unique() == torch.tensor([i, settings.label_index_thresh]))


class TestRandomMixUp:
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_basics(self, p: float) -> None:
        seed_everything(0, workers=True)
        batch = {
            "features": torch.arange(4, dtype=torch.float32).reshape(-1, 1, 1, 1).repeat(1, 10, 10, 3),
            "labels": torch.arange(4, dtype=torch.int64).reshape(-1, 1, 1).repeat(1, 10, 10),
            "valid_pixels": torch.ones(4, 10, 10, dtype=torch.bool),
            "image_labels": torch.arange(4, dtype=torch.int64),
            "meta": torch.tensor([[10, 0], [20, 1], [15, 0], [30, 1]], dtype=torch.float32),
        }
        batch["valid_pixels"][0, 0, 0] = False

        t = RandomMixUp(config=Config({"input/n_classes": 4}), p=p)
        batch_copy = copy.deepcopy(batch)
        batch = t(batch)

        assert batch["labels"].shape == (4, 10, 10, 4)
        assert batch["labels"].dtype == torch.float32
        labels_sum = batch["labels"].sum(dim=-1)
        assert torch.allclose(labels_sum, torch.ones_like(labels_sum))

        assert batch["features"].shape == (4, 10, 10, 3)
        assert batch["features"].dtype == torch.float32

        assert batch["valid_pixels"].dtype == torch.bool
        assert batch_copy["valid_pixels"].sum() == 4 * 10 * 10 - 1

        assert batch["image_labels"].shape == (4, 4)
        assert batch["image_labels"].dtype == torch.float32
        image_labels_sum = batch["labels"].sum(dim=-1)
        assert torch.allclose(image_labels_sum, torch.ones_like(image_labels_sum))

        assert batch["meta"].shape == (4, 2)
        assert batch["meta"].dtype == torch.float32
        extrema = batch["meta"].aminmax(dim=0)
        assert extrema.min[0] >= 10 and extrema.min[1] >= 0
        assert extrema.max[0] <= 30 and extrema.max[1] <= 1

        if p == 0.0:
            assert torch.allclose(batch["features"], batch_copy["features"])
            assert batch["valid_pixels"].sum() == 4 * 10 * 10 - 1
        else:
            assert not torch.allclose(batch["features"], batch_copy["features"])
            assert batch["valid_pixels"].sum() == 4 * 10 * 10 - 2, (
                "The invalid pixel of the first image now also affects another image"
            )
