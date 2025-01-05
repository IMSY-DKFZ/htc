# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
from lightning import seed_everything

from htc.models.common.HTCDatasetStream import HTCDatasetStream
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.data.DataSpecification import DataSpecification
from htc.models.hyper_diva.DatasetPixelStreamAverage import DatasetPixelStreamAverage
from htc.models.patch.DatasetPatchStream import DatasetPatchStream
from htc.models.pixel.DatasetPixelStream import DatasetPixelStream
from htc.models.superpixel_classification.DatasetSuperpixelStream import DatasetSuperpixelStream
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.Config import Config


@pytest.mark.serial
@pytest.mark.slow
class TestStreamDataLoader:
    @staticmethod
    def _check_dataloader(DatasetClass: type[HTCDatasetStream], config: Config) -> int:
        paths = DataSpecification("pigs_semantic-only_5foldsV2.json").paths()

        seed_everything(settings.default_seed, workers=True)
        dataset = DatasetClass(paths, train=True, config=config)
        dataloader = StreamDataLoader(dataset, config)
        img_indices = []
        for b, batch in enumerate(dataloader):
            img_indices.append(batch["image_index"])
            for value in batch.values():
                assert len(value) == config["dataloader_kwargs/batch_size"]

        # Check that the same images get loaded when iterating a second time
        seed_everything(settings.default_seed, workers=True)
        dataset = DatasetClass(paths, train=True, config=config)
        dataloader = StreamDataLoader(dataset, config)
        for b, batch in enumerate(dataloader):
            assert torch.all(img_indices[b] == batch["image_index"])

        return b

    def test_pixel(self) -> None:
        config = Config({
            "input/epoch_size": "5 images",
            "input/n_channels": 100,
            "dataloader_kwargs/batch_size": 118800,
            "dataloader_kwargs/num_workers": 2,
        })
        paths = DataSpecification("pigs_semantic-only_5foldsV2.json").paths()
        dataset = DatasetPixelStream(paths, train=True, config=config)
        dataloader = StreamDataLoader(dataset, config)

        for b, batch in enumerate(dataloader):
            for value in batch.values():
                assert len(value) == config["dataloader_kwargs/batch_size"]

        assert config["input/epoch_size"] == 1544400, "5*307200=1536000 does not fit, but 1544400 does"
        assert b + 1 == 1544400 / 118800, "Number of batches expected"

    def test_average(self) -> None:
        config = Config({
            "input/epoch_size": "5 images",
            "input/averaging": 5,
            "input/target_domain": ["subject_index"],
            "input/data_spec": "data/pigs_semantic-only_5foldsV2.json",
            "input/n_channels": 100,
            "dataloader_kwargs/batch_size": 118800,
            "dataloader_kwargs/num_workers": 2,
        })
        paths = DataSpecification("pigs_semantic-only_5foldsV2.json").paths()
        dataset = DatasetPixelStreamAverage(paths, train=True, config=config)
        dataloader = StreamDataLoader(dataset, config)

        for b, batch in enumerate(dataloader):
            for value in batch.values():
                assert len(value) == config["dataloader_kwargs/batch_size"]

        assert config["input/epoch_size"] == 356400, (
            "5*307200=1536000 and 1536000/5=307200 (for averaging=5) does not fit but 356400 does"
        )
        assert b + 1 == 356400 / 118800, "Number of batches expected"

    def test_patch(self) -> None:
        config = Config({
            "input/epoch_size": "5 images",
            "input/patch_size": [32, 32],
            "input/n_channels": 100,
            "dataloader_kwargs/batch_size": 600,
            "dataloader_kwargs/num_workers": 2,
            "label_mapping": settings_seg.label_mapping,
        })

        last_batch_index = TestStreamDataLoader._check_dataloader(DatasetPatchStream, config)

        assert config["input/epoch_size"] == 1800, "5*300 does not fit, but 1800 does"
        assert last_batch_index + 1 == 1800 / 600, "Number of batches expected"

    def test_superpixel_classification(self) -> None:
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/epoch_size": "5 images",
            "input/resize_shape": [32, 32],
            "input/n_channels": 100,
            "input/superpixels/n_segments": 1000,
            "input/superpixels/compactness": 10,
            "dataloader_kwargs/batch_size": 600,
            "dataloader_kwargs/num_workers": 2,
        })

        last_batch_index = TestStreamDataLoader._check_dataloader(DatasetSuperpixelStream, config)

        assert config["input/epoch_size"] == 5400, "5*1000 does not fit, but 5400 does"
        assert last_batch_index + 1 == 5400 / 600, "Number of batches expected"

    def test_shuffle_paths(self) -> None:
        seed_everything(settings.default_seed, workers=True)

        config = Config({
            "input/epoch_size": "5 images",
            "input/patch_size": [32, 32],
            "input/n_channels": 100,
            "dataloader_kwargs/batch_size": 600,
            "dataloader_kwargs/num_workers": 5,
            "label_mapping": settings_seg.label_mapping,
        })
        paths = DataSpecification("pigs_semantic-only_5foldsV2.json").paths()
        paths = paths[:5]
        dataset = DatasetPatchStream(paths, train=True, config=config)
        dataloader = StreamDataLoader(dataset, config)

        assert len(dataset.path_indices_worker) == 5

        # Each worker has exactly one image to work on so the path indices must match exactly with the image indices in the batch
        for batch in dataloader:
            assert torch.all(dataset.path_indices_worker.repeat_interleave(int(600 / 5)) == batch["image_index"])

        # Is also true in the next epoch
        for batch in dataloader:
            assert torch.all(dataset.path_indices_worker.repeat_interleave(int(600 / 5)) == batch["image_index"])
