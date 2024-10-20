# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import copy
import queue
import sys
from functools import cached_property, partial
from pathlib import Path

import psutil
import torch
import torch.multiprocessing as multiprocessing

from htc.model_processing.ImageConsumer import ImageConsumer
from htc.model_processing.Predictor import Predictor
from htc.models.common.HTCModel import HTCModel
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.type_from_string import variable_from_string


class Runner:
    def __init__(self, description: str) -> None:
        r"""
        Helper class to start the producer and the consumers to operate on predictions of images.

        The runner defines same basic arguments which apply to all scripts (e.g. --model and --run-folder to define the trained model). However, scripts can also add their own arguments. Please use the add_argument() method for this purpose. You can use it in the same way as with parser.add_argument() but it already defines some predefined arguments which are needed in multiple scripts.

        Default arguments:
        >>> import re
        >>> runner = Runner(description="My inference script")
        >>> re.sub(r"\s+", " ", runner.parser.format_usage())  # doctest: +ELLIPSIS
        'usage: ... [-h] --model MODEL --run-folder RUN_FOLDER [--config CONFIG] [--num-consumers NUM_CONSUMERS] [--num-workers NUM_WORKERS] [--store-predictions] [--use-predictions] [--hide-progressbar] [--input-dir INPUT_DIR] [--spec SPEC] [--spec-fold SPEC_FOLD] [--spec-split SPEC_SPLIT] [--paths-variable PATHS_VARIABLE] '

        Adding a custom argument with default options:
        >>> runner.add_argument("--annotation-name")
        >>> re.sub(r"\s+", " ", runner.parser.format_usage())  # doctest: +ELLIPSIS
        'usage: ... [-h] --model MODEL --run-folder RUN_FOLDER [--config CONFIG] [--num-consumers NUM_CONSUMERS] [--num-workers NUM_WORKERS] [--store-predictions] [--use-predictions] [--hide-progressbar] [--input-dir INPUT_DIR] [--spec SPEC] [--spec-fold SPEC_FOLD] [--spec-split SPEC_SPLIT] [--paths-variable PATHS_VARIABLE] [--annotation-name ANNOTATION_NAME] '

        Custom argument with a different option (note that --annotation-name is now a required argument):
        >>> runner = Runner(description="My inference script")
        >>> runner.parser.formatter_class = argparse.RawDescriptionHelpFormatter
        >>> runner.add_argument("--annotation-name", required=True)
        >>> re.sub(r"\s+", " ", runner.parser.format_usage())  # doctest: +ELLIPSIS
        'usage: ... [-h] --model MODEL --run-folder RUN_FOLDER [--config CONFIG] [--num-consumers NUM_CONSUMERS] [--num-workers NUM_WORKERS] [--store-predictions] [--use-predictions] [--hide-progressbar] [--input-dir INPUT_DIR] [--spec SPEC] [--spec-fold SPEC_FOLD] [--spec-split SPEC_SPLIT] [--paths-variable PATHS_VARIABLE] --annotation-name ANNOTATION_NAME '

        All custom arguments are automatically passed to the producer and the consumers, i.e. self.input_dir contains the path which the user submitted as command line argument.

        Args:
            description: Short description of what the script is doing. Will be added to the help message.
        """
        self.parser = argparse.ArgumentParser(
            description=(
                "Calculate the inference for a trained (potentially be downloaded) model for a set of images. Script"
                f" description: {description}"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Name of the model to train (e.g. image or pixel).",
        )
        self.parser.add_argument(
            "--run-folder",
            type=str,
            required=True,
            help=(
                "The name of the directory which stores the training results (e.g."
                " 2022-02-03_22-58-44_generated_default_model_comparison). It is also possible to pass a run folder"
                " name with a wildecard (e.g., nested-*-2) to select multiple runs which belong together and should"
                " also be loaded together (e.g., for ensembling)."
            ),
        )
        self.parser.add_argument(
            "--config",
            type=Path,
            default="config.json",
            help=(
                "Path of the config file to load (relative to the run directory or relative to the usual config"
                " search paths). This can be useful to load the same training run but with a different"
                " configuration file."
            ),
        )
        self.parser.add_argument(
            "--num-consumers",
            type=int,
            default=None,
            help=(
                "Number of consumers/processes to spawn which work on the predicted images. Defaults to n_physical_cpus"
                " - 1 to have at least one free CPU for the inference. If the NSD or ASD should be computed, the"
                " default is 2. Note that sometimes less is more and using fewer consumers may speed up the program"
                " time (e.g. if your consumer code uses many threads)."
            ),
        )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=1,
            help=(
                "Number of workers for the dataloader to load the data (dataloader_kwargs/num_workers argument)."
                " Defaults to 1 since this is usually enough for inference tasks if no special processing of the data"
                " is required."
            ),
        )
        self.parser.add_argument(
            "--store-predictions",
            default=False,
            action="store_true",
            help=(
                "Store predictions (<image_name>.blosc file with softmax predictions). If a script (or another script)"
                " is run on the same run directory again and the --use-predictions switch is set, then the"
                " precalculated predictions are used per default."
            ),
        )
        self.parser.add_argument(
            "--use-predictions",
            default=False,
            action="store_true",
            help="Use existing predictions if they already exist.",
        )
        self.parser.add_argument(
            "--hide-progressbar",
            default=False,
            action="store_true",
            help="If set, no progress bar is shown for the predictor.",
        )

        # Path options
        self.parser.add_argument(
            "--input-dir",
            type=Path,
            default=None,
            help="The directory path containing input data for the script.",
        )
        self.parser.add_argument(
            "--spec",
            type=str,
            default=None,
            help=(
                "If the inference needs to be carried out on a specific data spec file. This argument results in"
                " all unique paths from that data spec file to be used for inference."
            ),
        )
        self.parser.add_argument(
            "--spec-fold",
            type=str,
            default=None,
            help=(
                "If the inference-spec argument has been set, then use this argument to specify a particular fold"
                " within the data spec file. All unique paths in that fold inside that data spec file will be used"
                " for inference. This argument can only be used if inference-spec argument has been set."
            ),
        )
        self.parser.add_argument(
            "--spec-split",
            type=str,
            default="test",
            help=(
                "If the inference-spec has been set, then use this argument to specify a split name inside"
                " the data spec file. All unique paths in that split from that data spec files will"
                " be used for inference. This argument can only be used if inference-spec has been set."
            ),
        )
        self.parser.add_argument(
            "--paths-variable",
            type=str,
            default=None,
            help=(
                "Parsable variable string (cf. htc.utils.type_from_string.variable_from_string() function) which"
                " points to a list of DataPath objects. Those can, for example, be stored in a settings file."
            ),
        )

        self._used_args = []

    @cached_property
    def args(self) -> argparse.Namespace:
        return self.parser.parse_args()

    @property
    def run_dir(self) -> Path | list[Path]:
        return HTCModel.find_pretrained_run(self.args.model, self.args.run_folder)

    @cached_property
    def config(self) -> Config:
        possible_paths = Config.get_possible_paths(self.args.config)
        if isinstance(self.run_dir, list):
            possible_paths.insert(0, self.run_dir[0] / self.args.config)
        else:
            possible_paths.insert(0, self.run_dir / self.args.config)

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            raise FileNotFoundError(
                f"Config file {self.args.config} not found. Tried the following locations:\n{possible_paths}"
            )

        return Config(config_path)

    @cached_property
    def paths(self) -> list[DataPath]:
        """
        Returns: List of paths either collected based on the --input-dir, --paths-variable or the --spec argument. If none of these arguments is given, the spec from the config file is used to collect the paths.
        """
        dargs = vars(self.args)

        if dargs.get("input_dir") is not None:
            assert (
                dargs.get("spec") is None and dargs.get("spec_fold") is None and dargs.get("paths_variable") is None
            ), (
                "--spec, --spec-split, --spec-fold and --paths-variable arguments cannot be used together with the"
                " --input-dir argument"
            )

            input_dir = dargs.get("input_dir")
            assert input_dir.exists(), "Directory for which inference should be computed does not exist."

            return list(DataPath.iterate(input_dir, annotation_name=dargs.get("annotation_name")))
        elif dargs.get("paths_variable") is not None:
            assert dargs.get("spec") is None and dargs.get("spec_fold") and dargs.get("input_dir") is None, (
                "--spec, --spec-split, --spec-fold and --input-dir arguments cannot be used together with the"
                " --paths-variable argument"
            )
            return variable_from_string(dargs.get("paths_variable"))
        else:
            assert (
                dargs.get("input_dir") is None and dargs.get("paths_variable") is None
            ), "--input-dir and --paths-variable arguments cannot be used together with the --spec argument"
            if dargs.get("spec") is not None:
                spec = DataSpecification(dargs.get("spec"))
            else:
                try:
                    spec = DataSpecification.from_config(self.config)
                except FileNotFoundError:
                    # Sometimes the name of the spec changed, in this case load the data.json which should always be present
                    if isinstance(self.run_dir, list):
                        spec_path = self.run_dir[0] / "data.json"
                    else:
                        spec_path = self.run_dir / "data.json"
                    assert spec_path.exists(), f"Data specification file {spec_path} does not exist."
                    spec = DataSpecification(spec_path)

            if "test" in dargs.get("spec_split"):
                spec.activate_test_set()
                settings.log.info("Activating the test set of the data specification file")

            if dargs.get("spec_fold") is not None:
                return spec.fold_paths(dargs.get("spec_fold"), dargs.get("spec_split"))
            else:
                return spec.paths(dargs.get("spec_split"))

    def add_argument(self, name: str, **kwargs) -> None:
        """
        Add a custom argument to the runner. If the name is known (e.g. --test), then some defaults will be applied to the argument. It is always possible to overwrite the defaults. The value of all arguments is automatically passed on to the producer and the consumers.

        Args:
            name: Name of the argument (e.g. --test).
            kwargs: All additional keyword arguments passed to parser.add_argument()
        """
        if name == "--test":
            kwargs.setdefault("default", False)
            kwargs.setdefault("action", "store_true")
            kwargs.setdefault("help", "Create predictions on the test set (using the TestPredictor class).")
        elif name == "--test-looc":
            kwargs.setdefault("default", False)
            kwargs.setdefault("action", "store_true")
            kwargs.setdefault("help", "Use the TestLeaveOneOutPredictor class for test predictions (no ensembling).")
        elif name == "--input-dir":
            kwargs.setdefault("type", Path)
            kwargs.setdefault("default", None)
            kwargs.setdefault("help", "The directory path containing input data for the script.")
        elif name == "--output-dir":
            kwargs.setdefault("type", Path)
            kwargs.setdefault("default", None)
            kwargs.setdefault("help", "Output directory where the generated files should be stored.")
        elif name == "--metrics":
            kwargs.setdefault("type", str)
            kwargs.setdefault("nargs", "*")
            kwargs.setdefault("default", ["DSC", "ASD", "NSD"])
            kwargs.setdefault(
                "help",
                "The metrics which are to be calculated. By default, the dice, average surface distance and"
                " normalized surface dice are used.",
            )
        elif name == "--fold-name":
            kwargs.setdefault("type", str)
            kwargs.setdefault("default", None)
            kwargs.setdefault(
                "help",
                "The name of the fold for which the activations have to be calculated. Currently activations"
                " plotting is only implemented for the hyper_diva model.",
            )
        elif name == "--target-domain":
            kwargs.setdefault("type", str)
            kwargs.setdefault("default", None)
            kwargs.setdefault(
                "help",
                "The target domain for hyper_diva activations plotting. If this parameter is specified then the"
                " target_domain from the config files is overridden.",
            )
        elif name == "--annotation-name":
            kwargs.setdefault("type", str)
            kwargs.setdefault("default", None)
            kwargs.setdefault("help", "Filter the paths by this annotation name (default is no filtering).")
        elif name == "--NSD-thresholds":

            def float_or_str(value: float | str):
                try:
                    return float(value)
                except ValueError:
                    return value

            kwargs.setdefault("type", float_or_str)
            kwargs.setdefault("default", "semantic")
            kwargs.setdefault(
                "help",
                "The threshold which will be used for the NSD computation. In the resulting table, a column with"
                " the name surface_dice_metric_VALUE will be inserted. If a float is given, the same threshold for"
                " all classes will be used. May also be a name denoting the name of a set of threshold values, e.g."
                " semantic for the thresholds which we used in the MIA2022 paper. The threshold values are expected"
                " to be stored in the results directory at rater_variability/nsd_thresholds_<name>.csv or in"
                " models/nsd_thresholds_<name>.csv on the S3 storage.",
            )

        self.parser.add_argument(name, **kwargs)
        self._used_args.append(name.removeprefix("--").replace("-", "_"))

    def start(self, PredictorClass: type[Predictor], ConsumerClass: type[ImageConsumer]) -> None:
        """
        Start the producer and the consumers. If you need to pass additional parameters (not CLI arguments, they are automatically available in the producer and the consumers) to your consumers or producer, please use the functools.partial method.

        Args:
            PredictorClass: Class type (not an instance) used to create image predictions (usually TestPredictor or ValidationPredictor).
            ConsumerClass: Your consumer class type (not an instance).
        """
        with multiprocessing.Manager() as manager:
            # We provide two options per default to share results across consumers
            results_list = manager.list()
            results_dict = manager.dict()

            # Start consumers which will work on the image predictions
            if self.args.num_consumers is None:
                if "metrics" in self.args and ("NSD" in self.args.metrics or "ASD" in self.args.metrics):
                    settings.log.info(
                        "Using 2 consumers since boundary-based metrics are requested (too many consumers are harmful"
                        " in this case)"
                    )
                    num_consumers = 2  # Sweet spot
                else:
                    num_consumers = (
                        psutil.cpu_count(logical=False) - 1
                    )  # The main process will calculate the predictions
            else:
                num_consumers = self.args.num_consumers
            task_queue = manager.JoinableQueue(maxsize=num_consumers)
            task_queue_errors = manager.Queue(maxsize=num_consumers)

            # All arguments added via add_argument will be passed to the producer and the consumer
            additional_kwargs = {name: getattr(self.args, name) for name in self._used_args}
            additional_kwargs["paths"] = self.paths
            additional_kwargs["config"] = self.config

            # In principle, we give every argument to the consumers and producer
            # However, we need to make sure that we do not override arguments which the user passes directly (via partial) to the consumers or producer
            consumer_kwargs = {}
            for key, value in additional_kwargs.items():
                if isinstance(ConsumerClass, partial) and key in ConsumerClass.keywords:
                    continue

                consumer_kwargs[key] = value

            # Specify all arguments with explicit names for ConsumerClass to avoid errors due to partial class initialization
            consumer = ConsumerClass(
                task_queue=task_queue,
                task_queue_errors=task_queue_errors,
                results_list=results_list,
                results_dict=results_dict,
                run_dir=self.run_dir,
                store_predictions=self.args.store_predictions,
                **consumer_kwargs,
            )
            # A copy is much cheaper if __init__ is expensive
            consumers = [copy.copy(consumer) for _ in range(num_consumers)]
            for w in consumers:
                w.start()

            predictor_kwargs = {}
            for key, value in additional_kwargs.items():
                if isinstance(PredictorClass, partial) and key in PredictorClass.keywords:
                    continue

                predictor_kwargs[key] = value

            # The producer creates the predictions (only the main process since we use only 1 GPU)
            predictor = PredictorClass(
                run_dir=self.run_dir,
                use_predictions=self.args.use_predictions,
                store_predictions=self.args.store_predictions,
                num_workers=self.args.num_workers,
                **predictor_kwargs,
            )

            exit_code = 0
            try:
                with torch.autocast(device_type="cuda"):
                    predictor.start(task_queue, self.args.hide_progressbar)

                # Wait until all consumers are finished with this run
                task_queue.join()
                # Let one consumer do the final work
                task_queue.put("finished")
            except Exception:
                settings.log.exception("Error occurred in the producer")
                exit_code = 1
            finally:
                # Add a poison pill for each consumer to shut down
                for _ in range(num_consumers):
                    task_queue.put(None)

                task_queue.join()

            # Check whether some errors have happened
            try:
                # Consumers always return error information
                errors = []
                for _ in range(num_consumers):
                    response = task_queue_errors.get(timeout=30)  # timeout in seconds
                    if response is not None:
                        errors.append(response)
            except queue.Empty:
                exit_code = 1
                settings.log.error("Did not receive an answer about errors from every consumer")

            if len(errors) > 0:
                msg = (
                    f'{len(errors)} consumer{"s" if len(errors) > 1 else ""} failed with an error for the run'
                    f" {self.run_dir}:\n"
                )
                msg += "\n".join([str(e) for e in errors])
                settings.log.error(msg)
                exit_code = 1

            sys.exit(exit_code)
