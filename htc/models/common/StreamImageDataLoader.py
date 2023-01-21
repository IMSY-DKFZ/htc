# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from torch.utils.data import DataLoader

from htc.utils.Config import Config


class InfiniteSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from self.sampler


class StreamImageDataLoader(DataLoader):
    def __init__(self, *args, config: Config, **kwargs):
        """
        This class can be used instead of the PyTorch dataloader. It will sample the number of images according to the number of valid pixels of an image. For example, if 10 images should be seen during an epoch and 10 % of the pixels are annotated in an image, then 100 images will be yielded from this dataloader.

        Note: This dataloader cannot provide a length since the number of images will be unknown in advance. This also means that no progress bar will be shown during training.

        Args:
            config: Configuration object for the training.
        """
        assert "sampler" in kwargs and "batch_sampler" not in kwargs, "Currently only explicit sampler are supported"

        self.config = config

        # We adjust the sampler (and not the dataloader iteration) because it is much faster to iterate multiple times
        # over a sampler instead of iterating multiple times over a dataloader
        sampler = InfiniteSampler(kwargs.pop("sampler"))
        super().__init__(*args, sampler=sampler, **kwargs)

        # Calculate the total number of pixes this dataloader should return
        n_image_pixels = self.dataset.paths[0].dataset_settings.pixels_image()
        self.required_pixels = self.config["input/epoch_size"] * n_image_pixels

    def __iter__(self):
        yielded_pixels = 0
        for sample in super().__iter__():
            yielded_pixels += sample["valid_pixels"].count_nonzero()
            yield sample

            if yielded_pixels >= self.required_pixels:
                return
