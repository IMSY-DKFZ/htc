# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from htc.data_processing.DatasetIteration import DatasetIteration
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import compress_file
from htc.utils.Config import Config
from htc.utils.general import clear_directory
from htc.utils.paths import ParserPreprocessing


class L1Normalization(DatasetIteration):
    def __init__(self, paths: list[DataPath], file_type: str, output_dir: Path = None, regenerate: bool = False):
        super().__init__(paths)
        self.file_type = file_type

        if output_dir is None:
            self.output_dir = settings.intermediates_dir / "preprocessing" / "L1"
        else:
            self.output_dir = output_dir / "L1"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        config = Config(
            {
                "input/normalization": "L1",
                "input/no_labels": True,
            }
        )
        self.dataset = DatasetImage(self.paths, train=False, config=config)

        if regenerate:
            clear_directory(self.output_dir)

    def compute(self, i: int) -> None:
        if not (self.output_dir / f"{self.paths[i].image_name()}.{self.file_type}").exists():
            sample = self.dataset[i]
            img = sample["features"].numpy().astype(np.float16)

            if self.file_type == "npy":
                np.save(self.output_dir / sample["image_name"], img)
            elif self.file_type == "blosc":
                compress_file(self.output_dir / f"{sample['image_name']}.blosc", img)
            else:
                raise ValueError(f"Invalid file type {self.file_type}")


if __name__ == "__main__":
    prep = ParserPreprocessing(description="Precomputes a filter for all images")
    paths = prep.get_paths()

    settings.log.info(f"Computing L1 normalized images for {len(paths)} paths...")
    L1Normalization(
        paths=paths, file_type=prep.args.file_type, output_dir=prep.args.output_path, regenerate=prep.args.regenerate
    ).run()
