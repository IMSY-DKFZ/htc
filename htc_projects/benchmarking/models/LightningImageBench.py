# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import subprocess
import time
from pathlib import Path
from timeit import default_timer

import pandas as pd
from torch.utils.data import DataLoader

from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.image.DatasetImage import DatasetImage
from htc.models.image.DatasetImageBatch import DatasetImageBatch
from htc.models.image.LightningImage import LightningImage
from htc.settings import settings


class LightningImageBench(LightningImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start = None
        self.start_time = None
        self.timings = []

    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        if not kwargs["train"]:
            return LightningImage.dataset(**kwargs)
        else:
            config = kwargs["config"]

            if config["benchmarking/dataloader"] == "default":
                return DatasetImage(**kwargs)
            elif config["benchmarking/dataloader"] == "ring_buffer":
                # For benchmarking it is better to iterate only once over every image to avoid caching effects
                sampler = list(range(len(kwargs["paths"])))
                return DatasetImageBatch(sampler=sampler, **kwargs)
            else:
                raise ValueError(f"Unknown benchmarking/dataloader: {config['benchmarking/dataloader']}")

    def train_dataloader(self) -> DataLoader:
        if self.config["benchmarking/dataloader"] == "default":
            sampler = list(range(len(self.dataset_train)))
            return DataLoader(
                self.dataset_train, sampler=sampler, persistent_workers=True, **self.config["dataloader_kwargs"]
            )
        elif self.config["benchmarking/dataloader"] == "ring_buffer":
            return StreamDataLoader(self.dataset_train)
        else:
            raise ValueError(f"Unknown benchmarking/dataloader: {self.config['benchmarking/dataloader']}")

    def on_train_epoch_start(self) -> None:
        drop_cache_path = Path("/writable_proc/sys/vm/drop_caches")
        if drop_cache_path.exists():
            # Clear the system memory cache before every epoch so that every image has to be loaded from disk again
            # writable_proc instead of proc because we assume that the benchmark is run inside a Docker container
            res = subprocess.run(f"echo 3 > {drop_cache_path}", shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                settings.log.warning(f"Could not clear the system memory cache:\n{res.stderr}")

        # For absolute epoch start/end times so that we can extract the GPU util for the epoch later
        self.start_time = time.time()
        # Only for differences, not for absolute values (https://stackoverflow.com/a/72771333)
        self.start = default_timer()

    def on_train_epoch_end(self) -> None:
        end = default_timer()
        assert self.start is not None, "on_train_epoch_start not called"

        end_time = time.time()
        assert self.start_time is not None, "on_train_epoch_start not called"

        # We store the time (in seconds) for each epoch and filter out the warmup epochs later
        self.timings.append({
            "epoch_index": self.current_epoch,
            "time": end - self.start,
            "start_time": self.start_time,
            "end_time": end_time,
        })

        df = pd.DataFrame(self.timings)
        df.to_feather(Path(self.logger.save_dir) / "timings.feather")
