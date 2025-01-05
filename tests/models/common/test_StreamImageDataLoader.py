# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from torch.utils.data.sampler import RandomSampler

from htc.models.common.StreamImageDataLoader import StreamImageDataLoader
from htc.models.image.DatasetImage import DatasetImage
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestStreamImageDataLoader:
    def test_sampling(self) -> None:
        config = Config.from_model_name("default", "image")
        config["dataloader_kwargs/batch_size"] = 2
        config["input/epoch_size"] = 4

        dataset = DatasetImage([DataPath.from_image_name("P058#2020_05_13_18_09_26")], train=False, config=config)
        n_image_pixels = dataset.paths[0].dataset_settings.pixels_image()
        n_valid_pixels = dataset[0]["valid_pixels"].count_nonzero()
        assert n_valid_pixels < n_image_pixels, "This test only makes sense if the image contains invalid pixels"

        sampler = RandomSampler(dataset, replacement=True, num_samples=config["input/epoch_size"])
        dataloader = StreamImageDataLoader(dataset, config=config, sampler=sampler, **config["dataloader_kwargs"])

        n_received_batches = 0
        n_received_pixels = 0
        for sample in dataloader:
            n_valid_pixels_batch = sample["valid_pixels"].count_nonzero()
            assert n_valid_pixels_batch == n_valid_pixels * config["dataloader_kwargs/batch_size"]

            n_received_pixels += n_valid_pixels_batch
            n_received_batches += 1

        n_exact_batches = config["input/epoch_size"] // config["dataloader_kwargs/batch_size"]
        assert n_received_batches > n_exact_batches, (
            "There must be more batches since the image contains invalid pixels and we need more images to compensate"
            " for this"
        )
        assert n_received_pixels >= dataloader.required_pixels
