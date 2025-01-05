# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pickle

import numpy as np

from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.parallel import p_map
from htc.utils.paths import ParserPreprocessing


class RunningStats:
    def __init__(self, channels: int):
        self.sum = np.zeros(channels, np.float64)
        self.sum_squarred = np.zeros(channels, np.float64)
        self.total_elements = 0

    def add_data(self, data: np.ndarray) -> None:
        # Simple, iterative way to calculate std and mean (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Na%C3%AFve_algorithm)
        # It is not numerically stable, but with float64 it is ok
        data = data.astype(np.float64)
        dims_reduce = tuple(range(len(data.shape) - 1))
        self.sum += np.sum(data, axis=dims_reduce)
        self.sum_squarred += np.sum(data**2, axis=dims_reduce)
        self.total_elements += np.prod(data.shape[:-1])

    def channel_params(self) -> np.ndarray:
        mean = self.sum / self.total_elements
        std = np.sqrt((self.sum_squarred - self.sum**2 / self.total_elements) / self.total_elements)

        return mean, std

    def pixel_params(self) -> np.ndarray:
        # Pixel params based on the channel sums
        total_elements = self.total_elements * np.prod(self.sum.shape)
        total = np.sum(self.sum)
        sum_squarred = np.sum(self.sum_squarred)

        mean = total / total_elements
        std = np.sqrt((sum_squarred - total**2 / total_elements) / total_elements)

        return mean, std


def calc_standardization(datasets: dict[str, list[DataPath]]) -> dict[str, float]:
    rs_hsi = RunningStats(channels=100)
    rs_tpi = RunningStats(channels=4)
    rs_rgb = RunningStats(channels=3)

    for name, paths in datasets.items():
        if name.startswith("train"):
            dataset_hsi = DatasetImage(
                paths, train=False, config=Config({"input/n_channels": 100, "input/preprocessing": "L1"})
            )
            dataset_tpi = DatasetImage(
                paths, train=False, config=Config({"input/n_channels": 4, "input/preprocessing": "parameter_images"})
            )
            dataset_rgb = DatasetImage(paths, train=False, config=Config({"input/n_channels": 3}))

            for dataset, rs in [(dataset_hsi, rs_hsi), (dataset_tpi, rs_tpi), (dataset_rgb, rs_rgb)]:
                for sample in dataset:
                    rs.add_data(sample["features"].numpy())

    results = {}
    results["hsi_channel_mean"], results["hsi_channel_std"] = rs_hsi.channel_params()
    results["hsi_pixel_mean"], results["hsi_pixel_std"] = rs_hsi.pixel_params()

    results["tpi_channel_mean"], results["tpi_channel_std"] = rs_tpi.channel_params()
    results["tpi_pixel_mean"], results["tpi_pixel_std"] = rs_tpi.pixel_params()

    results["rgb_channel_mean"], results["rgb_channel_std"] = rs_rgb.channel_params()
    results["rgb_pixel_mean"], results["rgb_pixel_std"] = rs_rgb.pixel_params()

    return results


def calc_standardization_folds(specs: DataSpecification) -> dict[str, dict[str, float]]:
    fold_datasets = [specs.folds[f] for f in specs.fold_names()]
    results = p_map(calc_standardization, fold_datasets)

    return dict(zip(specs.fold_names(), results, strict=True))


if __name__ == "__main__":
    prep = ParserPreprocessing(description="Precomputes standardization statistics for each fold")
    paths = prep.get_paths()  # Must always be called
    assert prep.args.spec is not None, (
        "The --spec argument must be supplied so that the standardization parameters can be calculated per fold"
    )

    specs = DataSpecification(prep.args.spec)
    assert paths == specs.paths()
    results = calc_standardization_folds(specs)

    target_dir = settings.intermediates_dir_all / "data_stats"
    target_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(results, (target_dir / f"{specs.name()}#standardization.pkl").open("wb"))
