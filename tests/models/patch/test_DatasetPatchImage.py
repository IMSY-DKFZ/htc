# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch

from htc.models.image.DatasetImage import DatasetImage
from htc.models.patch.DatasetPatchImage import DatasetPatchImage
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestDatasetPatchImage:
    @pytest.mark.parametrize("patch_size", [[32, 32], [64, 64], [240, 320]])
    def test_reshape(self, patch_size: list) -> None:
        config = Config({
            "input/epoch_size": 300,
            "input/patch_size": patch_size,
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 10,
        })
        paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        dataset = DatasetPatchImage(paths, train=False, config=config)
        sample = dataset[0]

        dataset_img = DatasetImage(paths, train=False, config=config)
        sample_img = dataset_img[0]

        # Reshape patch features back to the image
        features = dataset.reshape_img(sample["features"], sample)
        assert torch.all(features == sample_img["features"])

        # During training, we actually need to reshape a block of predicted labels. Here, we use some dummy labels to test this
        labels_img = sample_img["features"].argmax(dim=-1)
        labels = sample["features"].argmax(dim=-1)
        labels = dataset.reshape_img(labels, sample)
        assert torch.all(labels == labels_img)

        if patch_size == [64, 64]:
            # The last patches at the bottom should contain only zeros
            assert torch.all(sample["features"][70:, 32:, :, :].unique() == torch.tensor([0]))
            assert not torch.all(sample["features"][69:, 32:, :, :].unique() == torch.tensor([0]))
            assert not torch.all(sample["features"][70:, 31:, :, :].unique() == torch.tensor([0]))
