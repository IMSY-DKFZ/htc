# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as multiprocessing

from htc.settings import settings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


class Predictor:
    def __init__(
        self,
        run_dir: Path | list[Path],
        config: Config | str,
        use_predictions: bool = False,
        store_predictions: bool = False,
        num_workers: int = 1,
        mode: str = "predictions",
        **kwargs,
    ):
        self.run_dir = run_dir
        if isinstance(self.run_dir, list):
            self.run_dir_main = self.run_dir[0]
        else:
            self.run_dir_main = self.run_dir

        self.config = config if type(config) == Config else Config(self.run_dir_main / config)
        self.config["dataloader_kwargs/num_workers"] = num_workers
        self.use_predictions = use_predictions
        self.store_predictions = store_predictions
        self.mode = mode

        # Avoid problems if this script is applied to new data with different labels (everything which the model does not know of will be ignored)
        mapping = LabelMapping.from_config(self.config)
        mapping.unknown_invalid = True

        self.name_path_mapping = {}

        # If not explicitly stated otherwise, we usually don't need labels for the prediction
        if "input/no_labels" not in self.config:
            self.config["input/no_labels"] = True

        if self.mode == "activations" or self.mode == "reconstructions":
            assert self.fold_name is not None, (
                f"The fold name has to specified when calculating {self.mode}. "
                "As, for predictions an ensemble of fold models is used but this is "
                f"not possible for {self.mode}, so they are calculated per-fold."
            )
            predictions_dir = self.run_dir_main / self.fold_name / self.mode
        elif self.mode == "predictions":
            predictions_dir = self.run_dir_main / "predictions"
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

    def load_predictions(self, image_name: str) -> torch.Tensor:
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

        return torch.from_numpy(data).share_memory_()

    @abstractmethod
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        pass
