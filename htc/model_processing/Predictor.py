# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing
from abc import abstractmethod
from pathlib import Path

import numpy as np

from htc.settings import settings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config


class Predictor:
    def __init__(
        self,
        run_dir: Path,
        use_predictions: bool = False,
        store_predictions: bool = False,
        config: Config = None,
        mode: str = "predictions",
        **kwargs,
    ):
        self.run_dir = run_dir
        self.use_predictions = use_predictions
        self.store_predictions = store_predictions
        self.mode = mode
        self.config = Config(self.run_dir / "config.json") if config is None else config
        self.config["dataloader_kwargs/num_workers"] = 1  # One worker process is usually enough for inference tasks
        self.name_path_mapping = {}

        # We usually don't need labels for the prediction
        self.config["input/no_labels"] = True

        if self.mode == "activations" or self.mode == "reconstructions":
            assert self.fold_name is not None, (
                f"The fold name has to specified when calculating {self.mode}. "
                "As, for predictions an ensemble of fold models is used but this is "
                f"not possible for {self.mode}, so they are calculated per-fold."
            )
            predictions_dir = self.run_dir / self.fold_name / self.mode
        elif self.mode == "predictions":
            predictions_dir = self.run_dir / "predictions"
        else:
            raise ValueError(f"Invalid value specified for mode parameter in Predictor class: {self.mode}")

        if self.use_predictions and predictions_dir.exists():
            files = sorted(predictions_dir.glob("*.blosc"))
            if len(files) == 0:
                files = sorted(predictions_dir.glob("*.npy"))

            self.existing_predictions = {f.stem: f for f in files}
            if len(self.existing_predictions) == 0:
                settings.log.warning(
                    f"The --use-predictions option is set but the predictions folder {predictions_dir} is empty."
                    " Predictions must be recomputed"
                )
        else:
            self.existing_predictions = {}

        for name, value in kwargs.items():
            setattr(self, name, value)

    def load_predictions(self, image_name: str) -> np.array:
        if image_name not in self.existing_predictions:
            return None

        path = self.existing_predictions[image_name]
        data = None

        if path.suffix == ".blosc":
            data = decompress_file(path)
        elif path.suffix == ".npy":
            data = np.load(path)
        else:
            raise ValueError(f"Cannot read the predictions from {path}")

        return data

    @abstractmethod
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        pass
