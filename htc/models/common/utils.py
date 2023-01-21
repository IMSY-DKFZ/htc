# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import functools
import math
import platform
import warnings
from collections.abc import Callable

import psutil
import torch

from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


def get_n_classes(config: Config) -> int:
    """
    Extracts the number of classes from the config. It will either use the key input/n_classes or infer the number from the label mapping.

    Args:
        config: The config of the training.

    Returns: Number of classes or 0 if no information in the config could be found.
    """
    if "input/n_classes" in config:
        return config["input/n_classes"]

    if "label_mapping" not in config:
        settings.log_once.warning(
            "There is neither a label mapping specified in the config nor is the key input/n_classes present. Cannot"
            " infer the number of classes (will return 0)"
        )
        return 0
    elif config["label_mapping"]:
        return len(LabelMapping.from_config(config))
    else:
        # User can explicitly disable the label mapping in which case no labels are used (e.g. self-supervised learning)
        return 0


def infer_swa_lr(config: Config) -> float:
    """
    Calculate the learning rate at the time when SWA kicks in. This might not be obvious if a custom learning rate scheduler is used.

    This is necessary because pytorch lightning now requires to explicitly set the learning rate for SWA (https://github.com/Lightning-AI/lightning/issues/11822).

    Args:
        config: The configuration of the training.

    Returns: The learning rate which can be used for SWA (`swa_lrs` argument).
    """
    from htc.models.common.HTCLightning import HTCLightning

    if config["swa_kwargs/swa_lrs"]:
        return config["swa_kwargs/swa_lrs"]
    else:
        lr = config["optimization/optimizer/lr"]

        # Dummy lightning class to get the optimizer and scheduler
        LightningClass = HTCLightning.class_from_config(config)
        module = LightningClass(dataset_train=None, datasets_val=None, config=config)

        res = module.configure_optimizers()
        if type(res) == tuple and len(res) == 2:
            scheduler = res[1][0]

            # Run the scheduler up to the epoch where SWA kicks in
            swa_start = config.get("swa_kwargs/swa_epoch_start", 0.8)
            before_swa_epochs = int(swa_start * config["trainer_kwargs/max_epochs"]) - 1
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of", category=UserWarning)

                # Unfortunately, there is no interface to get the lr for a specific epoch, so we have to iterate
                for _ in range(before_swa_epochs):
                    scheduler.step()

            swa_lr = scheduler.get_last_lr()
            if type(swa_lr) == list and len(swa_lr) == 1:
                swa_lr = swa_lr[0]
        else:
            swa_lr = lr

        return swa_lr


def cluster_command(args: str, memory: str = "10.7G", n_gpus: int = 1) -> str:
    """
    Generates a cluster command with some default settings.

    Args:
        args: Argument string for the run_training.py script (model, config, etc.).
        memory: The minimal memory requirements for the GPU.
        n_gpus: The number of GPUs to use.

    Returns: The command to run the job on the cluster.
    """
    # The dgx2 nodes have only 6 cores per job which is not enough for our runs
    bsubs_command = (
        'bsub -R "tensorcore" -R "select[hname!=\'e230-dgx2-1\']" -R "select[hname!=\'e230-dgx2-2\']" -q gpu-lowprio'
        f" -gpu num={n_gpus}:j_exclusive=yes:mode=exclusive_process:gmem={memory} ./runner_htc.sh htc training {args}"
    )

    return bsubs_command


def cpu_count() -> int:
    hostname = platform.node()

    if hostname in ["hdf19-gpu16", "hdf19-gpu17", "e230-AMDworkstation"]:
        return 16
    if hostname.startswith("hdf19-gpu") or hostname.startswith("e071-gpu"):
        return 12
    elif hostname.startswith("e230-dgx1"):
        return 10
    elif hostname.startswith("hdf18-gpu") or hostname.startswith("e132-comp"):
        return 16
    elif hostname.startswith("e230-dgx2"):
        return 6
    elif hostname.startswith("e230-dgxa100-"):
        return 28
    elif hostname.startswith("lsf22-gpu"):
        return 28
    else:
        # No hostname information available, just return the number of physical cores
        return psutil.cpu_count(logical=False)


def adjust_num_workers(config: Config) -> None:
    if config["dataloader_kwargs/num_workers"] == "auto":
        n_cpus = cpu_count()
        config["dataloader_kwargs/num_workers"] = n_cpus - 1  # One core is reserved for the main process
        settings.log.info(
            f'The number of workers are set to {config["dataloader_kwargs/num_workers"]} ({n_cpus} physical cores are'
            " available in total)"
        )

    return config


def adjust_epoch_size(config: Config) -> None:
    """
    Adjusts the config so that the epoch size is divisible by the batch size (the epoch size will be increased to the next multiple).

    >>> config = Config({'input/epoch_size': 19, 'dataloader_kwargs/batch_size': 5})
    >>> print('ignore_line'); adjust_epoch_size(config)  # doctest: +ELLIPSIS
    ignore_line...
    >>> config['input/epoch_size']
    20
    """
    assert (
        "input/epoch_size" in config and "dataloader_kwargs/batch_size" in config
    ), "Required config variables are missing"

    if config["input/epoch_size"] % config["dataloader_kwargs/batch_size"] != 0:
        next_multiple = math.ceil(config["input/epoch_size"] / config["dataloader_kwargs/batch_size"])
        config["input/epoch_size"] = config["dataloader_kwargs/batch_size"] * next_multiple
        settings.log.info(
            f'the epoch_size was set to {config["input/epoch_size"]} to make it divisible by the batch_size'
        )


def samples_equal(sample1: dict, sample2: dict, **allclose_kwargs) -> bool:
    """
    Compare two samples (e.g. from DatasetImage) for equality.

    Args:
        sample1: First sample.
        sample2: Second sample.
        allclose_kwargs: Additional parameters which can be passed to torch.allclose to adjust the comparison setting.

    Returns: True if all tensors are equal (ignoring small precision errors).
    """
    if sample1.keys() != sample2.keys():
        return False

    checks = []
    for key in sample1.keys():
        if type(sample1[key]) == torch.Tensor:
            if sample1[key].is_floating_point():
                checks.append(sample2[key].is_floating_point())
                checks.append(torch.allclose(sample1[key], sample2[key], **allclose_kwargs))
            else:
                checks.append(torch.all(sample1[key] == sample2[key]))
        else:
            checks.append(sample1[key] == sample2[key])

    return all(checks)


# Make sure we always have a batch dimension
def sample_to_batch(func: Callable) -> Callable:
    @functools.wraps(func)
    def _sample_to_batch(self, sample_or_batch: dict[str, torch.Tensor], *args, **kwargs):
        was_batch = True
        if sample_or_batch["features"].ndim == 3:
            was_batch = False

            batch = {}
            for key, value in sample_or_batch.items():
                if type(value) == torch.Tensor:
                    batch[key] = value.unsqueeze(dim=0)
                else:
                    batch[key] = [value]
        else:
            batch = sample_or_batch

        assert batch["features"].ndim == 4, "Features must be in BHWC format"

        batch = func(self, batch, *args, **kwargs)

        if not was_batch:
            # Remove batch dimension
            for key, value in batch.items():
                if type(value) == torch.Tensor:
                    batch[key] = value.squeeze(dim=0)
                else:
                    batch[key] = value[0]

        return batch

    return _sample_to_batch
