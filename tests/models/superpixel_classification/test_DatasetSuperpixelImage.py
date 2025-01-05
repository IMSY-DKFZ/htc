# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.image.DatasetImage import DatasetImage
from htc.models.superpixel_classification.DatasetSuperpixelImage import DatasetSuperpixelImage
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestDatasetSuperpixelImage:
    def test_sample(self) -> None:
        config = Config({
            "input/epoch_size": 1000,
            "input/n_channels": 100,
            "input/resize_shape": [32, 32],
            "input/preprocessing": "L1",
            "input/superpixels/n_segments": 1000,
            "input/superpixels/compactness": 10,
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
        })
        paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        dataset = DatasetSuperpixelImage(paths, train=False, config=config)
        sample = dataset[0]

        dataset_image = DatasetImage(paths, train=False, config=config)
        sample_image = dataset_image[0]

        n_superpixels = len(sample_image["spxs"].unique())
        assert n_superpixels == sample["features"].size(0)
        assert n_superpixels == sample["spxs_sizes"].size(0)
        assert sample["spxs_indices_rows"].shape == sample["spxs_indices_cols"].shape
        assert len(sample["spxs_indices_rows"].unique()) == sample_image["features"].shape[0]
        assert len(sample["spxs_indices_cols"].unique()) == sample_image["features"].shape[1]
        assert len(sample["spxs_indices_rows"]) == sample_image["features"].shape[0] * sample_image["features"].shape[1]
        assert len(sample["spxs_indices_cols"]) == sample_image["features"].shape[0] * sample_image["features"].shape[1]
