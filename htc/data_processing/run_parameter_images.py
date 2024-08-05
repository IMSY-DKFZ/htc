# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from htc.data_processing.DatasetIteration import DatasetIteration
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import compress_file
from htc.utils.general import clear_directory
from htc.utils.paths import ParserPreprocessing


class ParameterImages(DatasetIteration):
    def __init__(
        self,
        paths: list[DataPath],
        file_type: str,
        output_dir: Path = None,
        regenerate: bool = False,
        folder_name: str = "parameter_images",
    ):
        super().__init__(paths)
        self.file_type = file_type

        if output_dir is None:
            self.output_dir = settings.intermediates_dir_all / "preprocessing" / folder_name
        else:
            self.output_dir = output_dir / folder_name
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if regenerate:
            clear_directory(self.output_dir)

    def compute(self, i: int) -> None:
        path = self.paths[i]

        if self._compute_necessary(path.image_name()):
            cube = path.read_cube()
            sto2 = path.compute_sto2(cube)
            params = {
                "StO2": sto2.data,
                "NIR": path.compute_nir(cube).data,
                "TWI": path.compute_twi(cube).data,
                "OHI": path.compute_ohi(cube).data,
                "TLI": path.compute_tli(cube).data,
                "THI": path.compute_thi(cube).data,
                "background": sto2.mask,
            }

            for data in params.values():
                assert np.all(data >= 0) and np.all(data <= 1), "all values in the parameter image must be in [0, 1]"

            if self.file_type == "npy":
                np.savez_compressed(self.output_dir / path.image_name(), **params)
            elif self.file_type == "blosc":
                compress_file(self.output_dir / f"{path.image_name()}.blosc", params)
            else:
                raise ValueError(f"Invalid file type {self.file_type}")


if __name__ == "__main__":
    prep = ParserPreprocessing(description="Calculate tissue parameter images (sto2 etc.) from HSI cubes")
    paths = prep.get_paths()

    settings.log.info(f"Computing parameter images for {len(paths)} paths...")
    ParameterImages(
        paths, file_type=prep.args.file_type, output_dir=prep.args.output_path, regenerate=prep.args.regenerate
    ).run()
