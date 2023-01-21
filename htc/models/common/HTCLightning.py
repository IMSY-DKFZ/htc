# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import importlib
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from htc.models.common.EvaluationMixin import EvaluationMixin
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.transforms import HTCTransformation
from htc.models.common.utils import get_n_classes
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.type_from_string import type_from_string


class HTCLightning(EvaluationMixin, pl.LightningModule):
    def __init__(
        self, dataset_train: HTCDataset, datasets_val: list[HTCDataset], config: Config, dataset_test: HTCDataset = None
    ):
        super().__init__()

        self.dataset_train = dataset_train
        self.datasets_val = datasets_val
        self.dataset_test = dataset_test

        self.config = config
        self.n_classes = get_n_classes(self.config)

        self.checkpoint_metric_logged = {}
        self.df_validation_results = pd.DataFrame()

        # Statistics about the training
        self.training_stats = []

        # GPU transformations
        self.transforms = {}

    def train_dataloader(self) -> DataLoader:
        sampler = RandomSampler(self.dataset_train, replacement=True, num_samples=self.config["input/epoch_size"])
        return DataLoader(
            self.dataset_train, sampler=sampler, persistent_workers=True, **self.config["dataloader_kwargs"]
        )

    def val_dataloader(self, **kwargs) -> list[DataLoader]:
        # We first use the values from the config
        dataloader_kwargs = copy.deepcopy(self.config["dataloader_kwargs"])

        # Then we overwrite it with whatever is passed to this function
        dataloader_kwargs |= kwargs

        # The last step is to overwrite it with some defaults in case they have not been set before
        if "persistent_workers" not in dataloader_kwargs:
            dataloader_kwargs["persistent_workers"] = False

        return [DataLoader(dataset_val, **dataloader_kwargs) for dataset_val in self.datasets_val]

    def test_dataloader(self, **kwargs) -> DataLoader:
        dataloader_kwargs = copy.deepcopy(self.config["dataloader_kwargs"])
        dataloader_kwargs |= kwargs

        return DataLoader(self.dataset_test, **dataloader_kwargs)

    def configure_optimizers(self):
        # Dynamically initialize the optimizer based on the config
        optimizer_param = copy.deepcopy(self.config["optimization/optimizer"])
        del optimizer_param["name"]

        name = self.config["optimization/optimizer/name"]
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, name)

        optimizer = optimizer_class(self.model.parameters(), **optimizer_param)

        if self.config["optimization/lr_scheduler"]:
            # Same for the scheduler, if available
            scheduler_param = copy.deepcopy(self.config["optimization/lr_scheduler"])
            del scheduler_param["name"]

            name = self.config["optimization/lr_scheduler/name"]
            module = importlib.import_module("torch.optim.lr_scheduler")
            scheduler_class = getattr(module, name)

            scheduler = scheduler_class(optimizer, **scheduler_param)
            return [optimizer], [scheduler]
        else:
            return optimizer

    @staticmethod
    @abstractmethod
    def dataset(paths: list[DataPath], train: bool, config: Config, fold_name: str) -> HTCDataset:
        pass

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], dataloader_idx: int):
        transforms = self.parse_transforms_gpu()
        # TODO: add autocasting after this is fixed in kornia
        # with torch.autocast(device_type=self.device.type):
        #     batch = HTCTransformation.apply_valid_transforms(batch, transforms)
        batch = HTCTransformation.apply_valid_transforms(batch, transforms)

        return batch

    def _predict_images(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Create image-level predictions for the given batch. Usually required in the validation phase to ensure that a whole image can be validated irrespective of the underlying model (e.g. the pixel model must create the image based on the single pixel predictions). Should only be called by subclasses, the external interface is `predict_step()`.

        Args:
            batch: Data from the dataloader.

        Returns: Dictionary with predictions of the model (e.g. `result['class']` has a shape of [N, C, H, W]).
        """
        return {"class": self(batch["features"])}

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> dict[str, torch.Tensor]:
        """
        Create predictions for the batch. This function should be used for inference (and not `._predict_images()`) since it evaluates some basic checks and makes sure that the GPU transformations get applied.

        Args:
            batch: Batch for which the predictions should be applied.
            batch_idx: Index of the batch (exists for compatibility with lightning).

        Returns: Dictionary with model predictions (output of ._predict_images()).
        """
        assert torch.is_autocast_enabled(), "Please enable autocast"
        assert not torch.is_grad_enabled(), "Please disable gradients"
        assert not self.training, "Please put your model in .eval() mode"

        # We need to make sure the the GPU transforms get applied to the batch
        if not batch.get("transforms_applied", False):
            batch = self.on_after_batch_transfer(batch, dataloader_idx=0)
            assert batch[
                "transforms_applied"
            ], "Your on_after_batch_transfer method did not apply the GPU transformations"

        predictions = self._predict_images(batch)

        for name, tensor in predictions.items():
            if (nans := tensor.isnan()).any():
                n_nans = nans.nonzero().sum()
                settings.log.warning(f"Found {n_nans} nan values in the predictions {name} ({tensor.shape = })")

        return predictions

    def encode_images(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Output latent representations for the given batch. Only implemented for hyper_diva and pixel models at the moment.

        Args:
            batch: Data from the dataloader.

        Returns: Latent representations with shape dependent upon the model architecture
        """
        raise NotImplementedError

    def reconstruct_images(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Output reconstructions for the given batch. Only implemented for hyper_diva model at the moment.

        Args:
            batch: Data from the dataloader.

        Returns: Reconstructions
        """
        raise NotImplementedError

    def log_checkpoint_metric(self, metric_value: float) -> None:
        """
        Logs the metric which will be used in the checkpointing process, i.e. it is the value used by Lightning to determine the best checkpoint. This function must be called exactly once per epoch. Usually, it aggregates a metric from all samples in the validation dataset, e.g. the dice mean of all images.

        Args:
            metric_value: The metric value used for checkpointing.
        """
        if self.current_epoch in self.checkpoint_metric_logged:
            assert not self.checkpoint_metric_logged[self.current_epoch], (
                f"The checkpoint metric was already logged in the epoch {self.current_epoch}. Did you call"
                " log_checkpoint_metric more than once?"
            )
        else:
            self.checkpoint_metric_logged[self.current_epoch] = False

        self.log(self.config["validation/checkpoint_metric"], metric_value, prog_bar=True)
        self.checkpoint_metric_logged[self.current_epoch] = True

    def on_train_epoch_start(self) -> None:
        # Workaround to always save the last epoch until the issue is fixed in pl: https://github.com/PyTorchLightning/pytorch-lightning/issues/4539
        if self.current_epoch == self.trainer.max_epochs - 1:
            settings.log.info("Changed check_val_every_n_epoch to 1")
            self.trainer.check_val_every_n_epoch = 1

    def training_epoch_end(self, training_step_outputs: list[dict]) -> None:
        if len(training_step_outputs) > 0 and "img_indices" in training_step_outputs[0]:
            img_indices = torch.cat([x["img_indices"] for x in training_step_outputs]).cpu().numpy()

            self.training_stats.append({"img_indices": img_indices})
            np.savez_compressed(Path(self.logger.save_dir) / "trainings_stats.npz", data=self.training_stats)

    def parse_transforms_gpu(self, config_key: str = None) -> list[HTCTransformation]:
        if config_key is None:
            config_key = "input/transforms_gpu" if self.training else "input/test_time_transforms_gpu"

        if config_key not in self.transforms:
            # Only needed during training
            paths = self.dataset_train.paths if self.training else None
            fold_name = self.dataset_train.fold_name if self.dataset_train is not None else None

            if self.config[config_key]:
                self.transforms[config_key] = HTCTransformation.parse_transforms(
                    self.config[config_key],
                    config=self.config,
                    fold_name=fold_name,
                    paths=paths,
                    device=self.device,
                )
            else:
                # Default typing transformation (will cast to fp32)
                self.transforms[config_key] = HTCTransformation.parse_transforms()

        return self.transforms[config_key]

    @staticmethod
    def class_from_config(config: Config) -> type["HTCLightning"]:
        """
        Returns: The lightning class definition (not an instance) as defined in the configuration file.
        """
        assert "lightning_class" in config, "There is no lightning class defined in the config"
        assert type(config["lightning_class"]) == str, "The lightning class must be defined as string"

        return type_from_string(config["lightning_class"])
