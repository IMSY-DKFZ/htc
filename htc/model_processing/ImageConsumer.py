# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import traceback
from abc import abstractmethod
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import compress_file
from htc.utils.Config import Config


class ImageConsumer(multiprocessing.Process):
    def __init__(
        self,
        task_queue: multiprocessing.JoinableQueue,
        task_queue_errors: multiprocessing.Queue,
        results_list: list,
        results_dict: dict,
        run_dir: Path | list[Path],
        config: Config | str,
        store_predictions: bool,
        **kwargs,
    ):
        """
        This is the base class for all image consumers and every consumer should inherit from this class. You usually don't create instances of this class by yourself but let the run_image_consumer() function do the job. If you want to share results across consumers (e.g. gather all results and then save them in the run_finished() method).

        Please take a look at the existing consumers for examples.

        Args:
            task_queue: Reference to the queue used to share data between producers (predictors) and consumers.
            task_queue_errors: Queue which is used to communicate errors from the consumers back to the runner.
            results_list: list-like object which can be used to share results across consumers.
            results_dict: dict-like object which can be used to share results across consumers.
            run_dir: Path to the run directory where the predictions are calculated from.
            config: Configuration object to use or name of the configuration file to load (relative to the run directory). If None, the default configuration file of the training run will be loaded.
            store_predictions: Whether to store the predictions for later use (e.g. for faster inference on the next run).
            kwargs: All additional keyword arguments will be stored as attributes in this class.
        """
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.task_queue_errors = task_queue_errors
        self.results_list = results_list
        self.results_dict = results_dict
        self.run_dir = run_dir
        if isinstance(self.run_dir, list):
            self.run_dir = self.run_dir[0]
        self.store_predictions = store_predictions

        for name, value in kwargs.items():
            setattr(self, name, value)

        # All the produced files (tables, prediction files, etc.) should be stored either in the output directory (if the user supplied one) or the run folder
        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.target_dir = self.output_dir
            self.target_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.target_dir = self.run_dir
            # New results should be stored in the run directory
            self.run_dir.set_default_location(str(self.run_dir))

        self.config = config if type(config) == Config else Config(self.run_dir / config)

    def run(self):
        try:
            while True:
                image_data = self.task_queue.get()
                if image_data is None:
                    # Shutdown the consumer process
                    self.task_queue.task_done()
                    self.task_queue_errors.put(None)  # Communicate that no error occurred
                    break

                if image_data == "finished":
                    self.run_finished()
                else:
                    if self.store_predictions:
                        # Save predictions to avoid re-calculations the next time we need the predictions
                        if "activations" in image_data.keys():
                            predictions_dir = self.target_dir / image_data["fold_name"] / "activations"
                            predictions_data = image_data["activations"]
                        elif "reconstructions" in image_data.keys():
                            predictions_dir = self.target_dir / image_data["fold_name"] / "reconstructions"
                            predictions_data = image_data["reconstructions"]
                        elif "predictions" in image_data.keys():
                            predictions_dir = self.target_dir / "predictions"
                            predictions_data = image_data["predictions"]

                        predictions_dir.mkdir(parents=True, exist_ok=True)
                        compress_file(predictions_dir / f'{image_data["path"].image_name()}.blosc', predictions_data)

                    self.handle_image_data(image_data)

                self.task_queue.task_done()
        except Exception as e:
            # Unfortunately, there is no built-in mechanism for exception handling between producers and consumers
            # So, we handle all incoming tasks, do nothing with them and then at the end return the error
            exception_str = traceback.format_exc()
            self.task_queue.task_done()  # Mark the current task (the one which caused the error) as done
            settings.log.error(
                f"An {e.__class__.__name__} error ocurred in one of the consumers. You may want to stop the program"
            )

            if image_data is None:
                self.task_queue_errors.put(exception_str)
                return

            while True:
                image_data = self.task_queue.get()
                if image_data is None:
                    # Shutdown the consumer process and return the error
                    self.task_queue.task_done()
                    self.task_queue_errors.put(exception_str)
                    break
                else:
                    self.task_queue.task_done()

    @abstractmethod
    def handle_image_data(self, image_data: dict[str, torch.Tensor | DataPath | str]) -> None:
        """
        This abstract method must be implemented by every consumer and is called for every new prediction which is available.

        Args: The contents of the dict depend on the predictor but usually include the predictions, the image_name and the run directory.
        """

    def run_finished(self) -> None:
        """
        This is an optional method which is called at the very end, i.e. when all images are processed by all consumers.

        It will only be called for one consumer.
        """
