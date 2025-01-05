# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.superpixel_classification.DatasetSuperpixelStream import DatasetSuperpixelStream
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestDatasetSuperpixelStream:
    def test_sample(self) -> None:
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/epoch_size": 8,
            "input/n_channels": 100,
            "input/resize_shape": [32, 32],
            "input/superpixels/n_segments": 1000,
            "input/superpixels/compactness": 10,
            "dataloader_kwargs/num_workers": 2,
            "dataloader_kwargs/batch_size": 4,
        })
        paths = [
            DataPath.from_image_name("P043#2019_12_20_12_38_35"),
            DataPath.from_image_name("P059#2020_05_14_12_50_10"),
        ]
        dataset = DatasetSuperpixelStream(paths, train=False, config=config)
        dataloader = StreamDataLoader(dataset, config)

        sample = next(iter(dataloader))
        assert (
            len([
                dataset.paths[i]
                for i in sample["image_index"]
                if dataset.paths[i].image_name() == "P043#2019_12_20_12_38_35"
            ])
            == 2
        )
        assert (
            len([
                dataset.paths[i]
                for i in sample["image_index"]
                if dataset.paths[i].image_name() == "P059#2020_05_14_12_50_10"
            ])
            == 2
        )
        assert sample["weak_labels"].shape == (4, len(settings_seg.labels))
        assert sample["features"].shape == (4, 100, 32, 32)
