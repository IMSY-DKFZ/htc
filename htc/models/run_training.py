# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from threadpoolctl import threadpool_limits

from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.utils import adjust_num_workers, infer_swa_lr
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.DelayedFileHandler import DelayedFileHandler
from htc.utils.DuplicateFilter import DuplicateFilter
from htc.utils.MeasureTime import MeasureTime


class FoldTrainer:
    def __init__(self, model_name: str, config_name: str, config_extends: Union[str, None]):
        self.model_name = model_name
        self.config = Config.from_model_name(config_name, model_name, use_shared_dict=True)
        if config_extends is not None:
            self.config = self.config.merge(Config(json.loads(config_extends)))

        adjust_num_workers(self.config)

        # There must be a label mapping defined (class names to label ids)
        if not self.config["input/no_labels"] and "label_mapping" not in self.config:
            settings.log.warning(
                "No label mapping specified in the config file. The default mapping from the images will be used which"
                " may not be what you want (e.g. it is different across datasets). Best practice is to explicitly"
                " specify the label mapping in the config"
            )

        self.data_specs = DataSpecification.from_config(self.config)
        self.LightningClass = HTCLightning.class_from_config(self.config)

    def train_fold(self, run_folder: str, fold_name: str, *args) -> None:
        with MeasureTime("training_fold", silent=True) as mt:
            fold_name_tmp = (  # The results are first written to a temporary directory and later renamed back. This helps to easily detect incomplete runs
                f"running_{fold_name}"
            )

            # Directory to store all the results for the current model
            run_path = settings.training_dir / self.model_name / run_folder
            model_dir_tmp = run_path / fold_name_tmp
            model_dir_tmp.mkdir(parents=True, exist_ok=True)

            self._train_fold(model_dir_tmp, fold_name, *args)

        settings.log.info(
            f"Training time for the fold {fold_name}: {mt.elapsed_seconds // 60:.0f} minutes and"
            f" {mt.elapsed_seconds % 60:.2f} seconds"
        )
        settings.log.info(
            f"Peak memory consumption for the fold {fold_name}: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )

        # Fold completed successfully, rename folder back
        model_dir_tmp.rename(run_path / fold_name)

    def _train_fold(self, model_dir: str, fold_name: str, test: bool, file_log_handler: DelayedFileHandler) -> None:
        settings.log.info("The following config will be used for training:")
        settings.log.info(f"{self.config}")

        seed_everything(self.config.get("seed", settings.default_seed), workers=True)

        if "HTC_CUDA_MEM_FRACTION" in os.environ:
            # Set this environment variable if you want to enforce a memory limit as a fraction of the available maximal memory of your GPU (to make sure that the code also works on another device with less memory)
            # 0.44 = 2080 Ti
            fraction = float(os.getenv("HTC_CUDA_MEM_FRACTION", "1.0"))
            fraction = np.clip(fraction, 0, 1).item()
            torch.cuda.set_per_process_memory_fraction(fraction)
            torch.cuda.empty_cache()

        # Log file per fold
        file_log_handler.set_filename(model_dir / "log.txt", mode="w")

        # Start the logging script (it runs as long as this process is running)
        # We need to pipe to devnull as otherwise Popen might hang while running the tests
        monitor_handle = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "htc.utils.run_system_monitor",
                "--output-dir",
                str(model_dir),
                "--pid",
                str(os.getpid()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Create datasets based on the paths in the data specs
        train_paths = []
        test_paths = []
        datasets_val = []
        for name, paths in self.data_specs.folds[fold_name].items():
            assert not name.startswith("test"), "The test set should not be available at this point"

            if name.startswith("train"):
                train_paths += paths
            elif name.startswith("val"):
                dataset = self.LightningClass.dataset(paths=paths, train=False, config=self.config, fold_name=fold_name)
                datasets_val.append(dataset)
            else:
                settings.log_once.info(f"The split {name} is not used for training (neither starts with train nor val)")

        if test:
            # To avoid potential errors, we activate the test set only temporarily to get the paths
            # If other classes access the specs, they cannot accidentally access the test set
            with self.data_specs.activated_test_set():
                test_paths = self.data_specs.fold_paths(fold_name, "^test")

        # We use only one training dataset which uses all available images. Oversampling of images from one dataset can be implemented in the lightning class
        dataset_train = self.LightningClass.dataset(
            paths=train_paths, train=True, config=self.config, fold_name=fold_name
        )

        # Set some defaults if missing in the config
        if "validation/checkpoint_metric" not in self.config:
            self.config["validation/checkpoint_metric"] = "dice_metric"
            settings.log.warning(
                "No value set for validation/checkpoint_metric in the config. This should be the name of the metric"
                " which will be used to determine the best model. Please note that this does not specify the actual"
                " calculation of the metric but just the name of the metric (e.g. used in the checkpoint filename)."
                f" Defaulting to \"{self.config['validation/checkpoint_metric']}\""
            )
        if "validation/dataset_index" not in self.config:
            self.config["validation/dataset_index"] = 0
            settings.log.warning(
                "No value set for validation/dataset_index in the config. This specifies the main validation dataset,"
                " e.g. used for checkpointing. Currently, only one validation dataset can be used. Defaulting to"
                f" \"{self.config['validation/dataset_index']}\""
            )

        # Optional test dataset
        lightning_kwargs = {}
        if test and len(test_paths) > 0:
            dataset_test = self.LightningClass.dataset(
                paths=test_paths, train=False, config=self.config, fold_name=fold_name
            )
            lightning_kwargs["dataset_test"] = dataset_test

        # Main lightning module
        module = self.LightningClass(dataset_train, datasets_val, self.config, fold_name=fold_name, **lightning_kwargs)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [lr_monitor]

        if self.config["validation/checkpoint_saving"] is not False:
            checkpoint_saving = self.config.get("validation/checkpoint_saving", "best")
            if checkpoint_saving == "best":
                save_top_k = 1
                save_last = False
                test_ckpt_path = "best"
            elif checkpoint_saving == "last":
                save_top_k = 0
                save_last = True
                test_ckpt_path = model_dir / "last.ckpt"
            else:
                raise ValueError(f"Invalid option for checkpoint_saving: {checkpoint_saving}")

            checkpoint_callback = ModelCheckpoint(
                dirpath=model_dir,
                filename="{epoch:02d}-{" + self.config["validation/checkpoint_metric"] + ":.2f}",
                save_top_k=save_top_k,
                save_last=save_last,
                monitor=self.config["validation/checkpoint_metric"],
                mode=self.config.get("validation/checkpoint_mode", "max"),
            )
            callbacks.append(checkpoint_callback)
        else:
            self.config["trainer_kwargs"]["enable_checkpointing"] = False

        if self.config["trainer_kwargs/enable_progress_bar"] is not False:
            callbacks.append(RichProgressBar(leave=True))

        # Stochastic Weight Averaging
        if self.config["swa_kwargs"]:
            self.config["swa_kwargs/swa_lrs"] = infer_swa_lr(self.config)
            swa = StochasticWeightAveraging(**self.config["swa_kwargs"])
            callbacks.append(swa)

        # May be faster on a 3090 and should not hurt (https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision)
        torch.set_float32_matmul_precision("high")

        # Logs are stored in the same directory as the checkpoints
        logger = TensorBoardLogger(save_dir=model_dir, name="", version="")

        # Sanity check is disabled since it only leads to problems (duplicate epoch number, incomplete dataset)
        trainer = Trainer(logger=logger, callbacks=callbacks, num_sanity_val_steps=0, **self.config["trainer_kwargs"])

        # There are some problems on the cluster if too many threads are used because then more CPUs are used as available for the job
        # Hopefully, this with block limits the issue (right now there was only an issue on the main process)
        with threadpool_limits(2), warnings.catch_warnings():
            # We store the logs in the run folder, so it will never be empty
            warnings.filterwarnings(
                "ignore", message="Checkpoint directory.*exists and is not empty", category=UserWarning
            )
            warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined", category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message=(
                    ".*from an ambiguous collection. The batch size we found is"
                    f" {self.config['dataloader_kwargs/batch_size']}.*"
                ),
                category=UserWarning,
            )

            if self.config["wandb_kwargs"]:
                wandb_logger = WandbLogger(save_dir=model_dir, **self.config["wandb_kwargs"])
                wandb_logger.watch(module.model, log="all", log_freq=10)

            trainer.fit(module)
            if test and len(test_paths) > 0:
                trainer.test(verbose=True, ckpt_path=test_ckpt_path)

        # It might be good to know which keys in the config have never been accessed
        config_name = self.config["config_name"]
        unused_keys = self.config.unused_keys()
        if len(unused_keys) > 0:
            self.config["unused_keys"] = unused_keys  # For future reference
            settings.log.warning(
                f"The following keys are defined in the config {config_name} but have never been used during training:"
            )
            for key in unused_keys:
                settings.log.warning(key)

        self.config.save_config(model_dir / "config.json")
        shutil.copy2(self.data_specs.path, model_dir / "data.json")

        # Inform the system monitor that the training is finished
        monitor_handle.send_signal(signal.SIGINT)
        try:
            # Wait for a moment to give the system monitor time to finish
            # This is important when the training is run in a Docker container because the container may be stopped before the system monitor is finished
            monitor_handle.wait(timeout=2)
        except subprocess.TimeoutExpired:
            settings.log.warning("The system monitor did not terminate in time. Logs may not be complete")


def train_all_folds(
    model_name: str,
    config_name: str,
    config_extends: Union[str, None],
    run_folder: Union[str, None],
    test: bool,
    file_log_handler: DelayedFileHandler,
) -> None:
    with MeasureTime("training_all", silent=True) as mt:
        config = Config.from_model_name(config_name, model_name)
        if config_extends is not None:
            config = config.merge(Config(json.loads(config_extends)))

        # Unique folder name per run
        if run_folder is None:
            run_folder = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_{config["config_name"]}')
        run_folder_tmp = (  # The results are first written to a temporary directory and later renamed back. This helps to easily detect incomplete runs
            f"running_{run_folder}"
        )

        # Logs for all folds
        run_path = settings.training_dir / model_name / run_folder_tmp
        run_path.mkdir(exist_ok=True, parents=True)
        file_log_handler.set_filename(run_path / "log.txt", mode="w")

        # Start the training script for each fold
        data_specs = DataSpecification.from_config(config)
        error_occurred = False
        for i, fold_name in enumerate(data_specs.fold_names()):
            # We start the training of a fold in a new process so that we can start fresh, i.e. this makes sure that all resources like RAM are freed
            command = (
                f"{sys.executable} {__file__} --model {model_name} --config {config_name} --fold-name"
                f' {fold_name} --run-folder "{run_folder_tmp}"'
            )
            if test:
                command += " --test"
            if config_extends is not None:
                command += f' --config-extends "{config_extends}"'

            settings.log.info(f"Starting training of the fold {fold_name} [{i + 1}/{len(data_specs.fold_names())}]")
            ret = subprocess.run(command, shell=True)
            if ret.returncode != 0:
                settings.log.error(f"Training of the fold {fold_name} was not successful (returncode={ret.returncode}")
                error_occurred = True

        if error_occurred:
            settings.log.error("Some folds were not successful (see error messages above)")

    settings.log.info(
        f"Training time for the all folds: {mt.elapsed_seconds // 60:.0f} minutes and"
        f" {mt.elapsed_seconds % 60:.2f} seconds"
    )

    # Job completed, rename folder back
    run_path_tmp = settings.training_dir / model_name / run_folder_tmp
    if error_occurred:
        run_path = settings.training_dir / model_name / f"error_{run_folder}"
    else:
        run_path = settings.training_dir / model_name / run_folder
    run_path_tmp.rename(run_path)


if __name__ == "__main__":
    # For debugging: python run_training.py --model image --fold fold_P045,P061,P071 --run-folder test
    # This does not call the train_all_folds() function but instead directly the FoldTrainer.train_fold() method which makes it easier to step through the code as it does not spawn additional Python processes
    parser = argparse.ArgumentParser(
        description="Start the training process of a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train (e.g. image or pixel).")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Name of the configuration file to use (either absolute, relative to the current working directory or"
            " relative to the models config folder)."
        ),
    )
    parser.add_argument(
        "--config-extends",
        type=str,
        default=None,
        help="""JSON string which can be used to extend config parameters. For example, --config-extends '{"seed": 1}' will set a specific seed for the training.""",
    )
    parser.add_argument(
        "--fold-name",
        type=str,
        default="all",
        help="The fold to train on or all to train on every fold in the data specification",
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help=(
            "The directory to store the training results (with --fold all this directory will be created automatically)"
        ),
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="Evaluate on the test set at the end of the training (calls trainer.test())",
    )

    args = parser.parse_args()

    # Store logs additionally in a file
    file_log_handler = DelayedFileHandler()
    file_log_handler.addFilter(DuplicateFilter())
    file_log_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d_%H-%M-%S"
        )
    )
    logging.getLogger().addHandler(file_log_handler)

    # Debug messages for training
    settings.log.setLevel(logging.DEBUG)

    # Exceptions should also appear in the logs
    def handle_exception(exc_type, exc_value, exc_traceback):
        settings.log.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
        for handler in settings.log.handlers:
            handler.flush()

        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    if args.fold_name == "all":
        train_all_folds(args.model, args.config, args.config_extends, args.run_folder, args.test, file_log_handler)
    else:
        fold_trainer = FoldTrainer(args.model, args.config, args.config_extends)
        fold_trainer.train_fold(args.run_folder, args.fold_name, args.test, file_log_handler)
