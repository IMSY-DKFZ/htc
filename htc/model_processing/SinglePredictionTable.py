# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
import torch

from htc.model_processing.run_tables import TableTestPredictor
from htc.models.common.HTCModel import HTCModel
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class SinglePredictionTable:
    def __init__(
        self,
        model: str = None,
        run_folder: str = None,
        path: str | Path = None,
        fold_name: str = None,
        device: str = "cuda",
        test: bool = True,
        config: Config | str = None,
    ) -> None:
        """
        This class can be used to evaluate a model against a set of images.

        This is similar to the `htc tables` command, but does not spawn producer-consumer processes and operates only on the main process. It also returns the resulting table directly and does not store it. This is useful if a custom set of paths should be validated.

        In comparison to the SinglePredictor class, this class does not return the model predictions but only the computed metrics.

        Example prediction using a single model:
        >>> from htc import DataPath
        >>> table_predictor = SinglePredictionTable(
        ...     model="image", run_folder="2023-02-08_14-48-02_organ_transplantation_0.8"
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> path1 = DataPath.from_image_name("P041#2019_12_14_12_29_18")
        >>> path2 = DataPath.from_image_name("P041#2019_12_14_13_33_30")
        >>> path3 = DataPath.from_image_name("P043#2019_12_20_10_08_40")
        >>> df = table_predictor.compute_table_paths([path1, path2, path3], ["DSC"])  # doctest: +ELLIPSIS
        [...]
        >>> len(df)
        3
        >>> list(df.columns)
        ['dataset_index', 'image_name', 'subject_name', 'timestamp', 'annotation_name', 'dice_metric', 'dice_metric_image', 'used_labels']

        The resulting table `df` contains the computed metrics for every requested image. Usually, you want to further hierarchically aggregate those metrics towards class-level scores. This can be achieved via the `MetricAggregation` class:
        >>> from htc import MetricAggregation
        >>> agg = MetricAggregation(df, config=table_predictor.config, metrics=["dice_metric"])
        >>> df_agg = agg.grouped_metrics()
        >>> list(df_agg.columns)
        ['label_index', 'dice_metric', 'label_name']

        Args:
            model: Basic model type like image or pixel. Passed directly to `HTCModel.find_pretrained_run()`.
            run_folder: Name of the training run directory. Passed directly to `HTCModel.find_pretrained_run()`.
            path: Direct path to the run directory or to a fold. Passed directly to `HTCModel.find_pretrained_run()`. If the path to a fold is given (and fold_name is None), the model for this fold will be used.
            fold_name: Name of the validation fold which defines the trained network of the run. If None and test=False, the model with the highest metric score will be used.
            device: Device which is used to compute the predictions.
            test: If True, ensembles the output from all models. If False, the output from only one model will be used (e.g., the best model or the model from the requested fold).
            config: Configuration object to use or name of the configuration file to load (relative to the run directory). If None, the default configuration file of the training run will be loaded.
        """
        self.run_dir = HTCModel.find_pretrained_run(model, run_folder, path)
        if isinstance(self.run_dir, list):
            self.run_dir_main = self.run_dir[0]
        else:
            self.run_dir_main = self.run_dir

        self.device = device
        self.test = test
        if config is None:
            self.config = Config(self.run_dir_main / "config.json")
        else:
            self.config = config if type(config) == Config else Config(self.run_dir_main / config)

        if not self.test:
            if fold_name is not None:
                # Explicit fold
                model_path = HTCModel.best_checkpoint(self.run_dir / fold_name)
            elif path is not None:
                # Whatever the user specified (explicit fold if it points to the fold directory, otherwise the best fold will be used)
                model_path = HTCModel.best_checkpoint(path)
            else:
                # Best fold
                model_path = HTCModel.best_checkpoint(self.run_dir)

            self.fold_name = model_path.parent.name

    def compute_table_paths(self, paths: list[DataPath], metrics: list[str]) -> pd.DataFrame:
        """
        Compute a table with results for the given paths and metrics.

        Args:
            paths: List of image paths to evaluate (the images must have associated segmentations).
            metrics: List of metric names to compute. See `evaluate_images()` function for details.

        Returns: Table with the computed metrics.
        """
        if self.test:
            predictor = TableTestPredictor(
                self.run_dir,
                metrics=metrics,
                paths=paths,
                config=self.config,
            )
        else:
            predictor = TableTestPredictor(
                self.run_dir,
                fold_names=[self.fold_name],
                metrics=metrics,
                paths=paths,
                config=self.config,
            )

        with torch.no_grad(), torch.autocast(device_type=self.device):
            predictor.start(task_queue=None, hide_progressbar=False)

        return pd.DataFrame(predictor.rows)
