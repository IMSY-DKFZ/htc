# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingsBenchmarking:
    def __init__(self):
        self.colors_dark = {
            "original": "#7F7F7F",
            "blosc": "#FF7F0E",
            "fp16": "#8C564B",
            "gpu-aug": "#9467BD",
            "ring-buffer": "#2CA02C",
        }
        self.colors_light = {
            "original": "#ECECEC",
            "blosc": "#FFECDB",
            "fp16": "#EEE6E4",
            "gpu-aug": "#EFE8F5",
            "ring-buffer": "#DFF1DF",
        }

        self.networks_timestamp = os.getenv("HTC_BENCHMARKING_TIMESTAMP", "2023-09-03_22-48-13")
        self.warmup = 2  # n epochs in the beginning which should be ignored
        self.window = 5  # smoothing of the utilization values
        self.sorting = ["original", "blosc", "fp16", "gpu-aug", "ring-buffer"]

        self._results_dir = None

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_BENCHMARKING", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_BENCHMARKING is not set. Files for the benchmarking project"
                    f" will be written to {self._results_dir.find_best_location()}"
                )

        return self._results_dir

    @property
    def paper_dir(self) -> MultiPath:
        target_dir = self.results_dir / "paper"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir


settings_benchmarking = SettingsBenchmarking()
