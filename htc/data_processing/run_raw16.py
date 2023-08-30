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


class Raw16(DatasetIteration):
    def __init__(
        self,
        paths: list[DataPath],
        file_type: str,
        output_dir: Path = None,
        regenerate: bool = False,
        precision: str = "16",
    ):
        super().__init__(paths)
        self.file_type = file_type
        self.precision = precision

        if output_dir is None:
            self.output_dir = settings.intermediates_dir_all / "preprocessing" / f"raw{self.precision}"
        else:
            self.output_dir = output_dir / f"raw{self.precision}"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        config = Config(
            {
                "trainer_kwargs/precision": "16-mixed" if self.precision == "16" else "32",
                "input/no_labels": True,
            }
        )
        self.dataset = DatasetImage(self.paths, train=False, config=config)

        if regenerate:
            clear_directory(self.output_dir)

    def compute(self, i: int) -> None:
        if not (self.output_dir / f"{self.paths[i].image_name()}.{self.file_type}").exists():
            sample = self.dataset[i]
            img = sample["features"].numpy()
            if self.precision == "16":
                img = img.astype(np.float16)

            if self.file_type == "npy":
                np.save(self.output_dir / sample["image_name"], img)
            elif self.file_type == "blosc":
                compress_file(self.output_dir / f"{sample['image_name']}.blosc", img)
            else:
                raise ValueError(f"Invalid file type {self.file_type}")


if __name__ == "__main__":
    prep = ParserPreprocessing(description="Stores all images as float16 or float32 without further preprocessing")
    prep.parser.add_argument(
        "--precision",
        default="16",
        choices=["16", "32"],
        type=str,
        help="Target precision (16 or 32 bit)",
    )
    paths = prep.get_paths()

    Raw16(
        paths=paths,
        file_type=prep.args.file_type,
        output_dir=prep.args.output_path,
        regenerate=prep.args.regenerate,
        precision=prep.args.precision,
    ).run()
