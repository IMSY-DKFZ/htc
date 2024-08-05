# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
import functools
import importlib
import math
import os
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import psutil
import torch
import torch.nn as nn

from htc.models.common.torch_helpers import str_to_dtype
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.import_extra import requires_extra
from htc.utils.LabelMapping import LabelMapping

try:
    import paramiko

    _missing_library = ""
except ImportError:
    _missing_library = "paramiko"


def get_n_classes(config: Config) -> int:
    """
    Extracts the number of classes from the config. It will either use the key input/n_classes or infer the number from the label mapping.

    Args:
        config: The configuration of the training.

    Returns: Number of classes or 0 if no information in the config could be found.
    """
    if "input/n_classes" in config:
        return config["input/n_classes"]

    try:
        mapping = LabelMapping.from_config(config)
        return len(mapping)
    except Exception:
        settings.log_once.warning(
            "There is neither a label mapping (or image label mapping) specified in the config nor is the key"
            " input/n_classes present. Cannot infer the number of classes (will return 0)"
        )
        return 0


def model_input_channels(config: Config) -> int:
    """
    Extract the number of input channels to the model from the config.

    Args:
        config: The configuration of the training.

    Returns: Number of input channels.
    """
    return config.get("model/n_input_channels", config["input/n_channels"])


def parse_optimizer(config: Config, model: nn.Module) -> Union[tuple[list, list], Any]:
    """
    Creates an optimizer plus optionally an scheduler which can be used in your lightning module (via `configure_optimizers()`).

    Args:
        config: The configuration of the training (must include an `optimization` key).
        model: The network model containing the parameters to optimize.

    Returns: Optimizer or optimizer and scheduler as tuple (same format as lightning).
    """
    # Dynamically initialize the optimizer based on the config
    optimizer_param = copy.deepcopy(config["optimization/optimizer"])
    del optimizer_param["name"]

    name = config["optimization/optimizer/name"]
    module = importlib.import_module("torch.optim")
    optimizer_class = getattr(module, name)

    if config["optimization/optimizer_layer_settings"]:
        # Transform the layer settings to the format of the optimizer for per-parameter settings (https://pytorch.org/docs/stable/optim.html#per-parameter-options)
        model_parameters_helper = {}
        for pattern in config["optimization/optimizer_layer_settings"].keys():
            model_parameters_helper[pattern] = []
        model_parameters_default = []

        compiled_patterns = {}  # Lookup table
        for layer_name, layer_params in model.named_parameters():
            matched = False
            for pattern in config["optimization/optimizer_layer_settings"].keys():
                if pattern not in compiled_patterns:
                    compiled_patterns[pattern] = re.compile(pattern)

                if compiled_patterns[pattern].search(layer_name):
                    model_parameters_helper[pattern].append(layer_params)
                    matched = True
                    break

            if not matched:
                model_parameters_default.append(layer_params)

        # Check that every pattern matches at least one layer
        # infer_swa_lr() uses a dummy layer and in that case, we do not want to perform this check
        is_dummy_layer = type(model) == nn.Linear and model.in_features == 2 and model.out_features == 1
        if not is_dummy_layer:
            for pattern, layers in model_parameters_helper.items():
                assert len(layers) > 0, f"Found no layers which match the pattern {pattern}"

        model_parameters = [{"params": model_parameters_default}]
        for pattern, layers in model_parameters_helper.items():
            model_parameters.append({"params": layers} | config["optimization/optimizer_layer_settings"][pattern])

        optimizer = optimizer_class(model_parameters, **optimizer_param)
    else:
        optimizer = optimizer_class(model.parameters(), **optimizer_param)

    if config["optimization/lr_scheduler"]:
        # Same for the scheduler, if available
        scheduler_param = copy.deepcopy(config["optimization/lr_scheduler"])
        del scheduler_param["name"]

        name = config["optimization/lr_scheduler/name"]
        module = importlib.import_module("torch.optim.lr_scheduler")
        scheduler_class = getattr(module, name)

        scheduler = scheduler_class(optimizer, **scheduler_param)
        return [optimizer], [scheduler]
    else:
        return optimizer


def dtype_from_config(config: Config) -> torch.dtype:
    """
    Infer the dtype of the features from the config.

    Args:
        config: The configuration of the training.

    Returns: The dtype of the features.
    """
    if config["input/features_dtype"]:
        # User sets the dtype to load the features to the GPU explicitly
        features_dtype = str_to_dtype(config["input/features_dtype"])
    else:
        # If the data is stored as float16, we also load it as float16
        # As a user, it is possible to convert the data via:
        # input/test_time_transforms_cpu": [{"class": "ToType", "dtype": "float32"}]
        default_dtype = (
            torch.float16
            if config["input/preprocessing"] and config["input/preprocessing"].startswith(("raw16", "L1"))
            else torch.float32
        )
        features_dtype = (
            torch.float16 if config["trainer_kwargs/precision"] in [16, "16-mixed", "16-true"] else default_dtype
        )

    return features_dtype


def infer_swa_lr(config: Config) -> float:
    """
    Calculate the learning rate at the time when SWA kicks in. This might not be obvious if a custom learning rate scheduler is used.

    This is necessary because pytorch lightning now requires to explicitly set the learning rate for SWA (https://github.com/Lightning-AI/lightning/issues/11822).

    Args:
        config: The configuration of the training.

    Returns: The learning rate which can be used for SWA (`swa_lrs` argument).
    """
    if config["swa_kwargs/swa_lrs"]:
        return config["swa_kwargs/swa_lrs"]
    else:
        lr = config["optimization/optimizer/lr"]

        # Dummy model to get a valid PyTorch optimizer
        dummy_model = torch.nn.Linear(2, 1)
        res = parse_optimizer(config, dummy_model)

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


def cluster_command(args: str, memory: str = "10.7G", n_gpus: int = 1, excluded_hosts: list[str] = None) -> str:
    """
    Generates a cluster command with some default settings.

    Args:
        args: Argument string for the run_training.py script (model, config, etc.).
        memory: The minimal memory requirements for the GPU.
        n_gpus: The number of GPUs to use.
        excluded_hosts: List of hosts to exclude. If None, no hosts are excluded.

    Returns: The command to run the job on the cluster.
    """
    if excluded_hosts is not None:
        excluded_hosts = " && ".join([f"hname!='{h}'" for h in excluded_hosts])
        excluded_hosts = f'-R "select[{excluded_hosts}]" '
    else:
        excluded_hosts = ""

    bsubs_command = (
        f'bsub -R "tensorcore" {excluded_hosts}-q gpu-lowprio'
        f" -gpu num={n_gpus}:j_exclusive=yes:gmem={memory} ./runner_htc.sh htc training {args}"
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


@requires_extra(_missing_library)
def run_jobs(jobs: list[str]) -> None:
    # Make sure the cluster is up-to-date
    res = subprocess.run([sys.executable, settings.src_dir / "cluster/run_update_cluster.py"])
    assert res.returncode == 0, "Could not update the cluster files"

    sftpURL = "bsub01.lsf.dkfz.de"
    sftpUser = settings.dkfz_userid

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()

    # Automatically add keys without requiring human intervention
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(sftpURL, username=sftpUser)
    except paramiko.SSHException:
        # Sometimes the key cannot be found automatically, try the default location in this case
        ssh.connect(sftpURL, username=sftpUser, key_filename=os.path.expanduser("~/.ssh/cluster"))

    # We cannot submit too many jobs at once so we submit them in multiple submissions
    settings.log.info(f"Submitting {len(jobs)} jobs to the cluster...")
    MAX_SUBMISSION_JOBS = 100
    for jobs_split in np.array_split(jobs, np.ceil(len(jobs) / MAX_SUBMISSION_JOBS)):
        jobs_str = "\n".join(jobs_split.tolist()).replace(
            '"', '\\"'
        )  # We need to excape any " as we wrap the command in ""
        _, stdout, stderr = ssh.exec_command(
            f'bash --login -c "{jobs_str}"'
        )  # We need to login into the shell so that the cluster commands are available
        if len(out_lines := stdout.readlines()) > 0:
            settings.log.info(out_lines)
        if len(err_lines := stderr.readlines()) > 0:
            settings.log.error(err_lines)

    ssh.close()


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

    Args:
        config: The configuration of the training.
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
        elif type(sample1[key]) == dict and type(sample2[key]) == dict:
            checks.append(samples_equal(sample1[key], sample2[key], **allclose_kwargs))
        else:
            checks.append(sample1[key] == sample2[key])

    return all(checks)


# Make sure we always have a batch dimension
def sample_to_batch(func: Callable) -> Callable:
    @functools.wraps(func)
    def _sample_to_batch(sample_or_batch: dict[str, torch.Tensor], *args, **kwargs):
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

        batch = func(batch, *args, **kwargs)

        if not was_batch:
            # Remove batch dimension
            for key, value in batch.items():
                if type(value) == torch.Tensor:
                    batch[key] = value.squeeze(dim=0)
                else:
                    batch[key] = value[0]

        return batch

    return _sample_to_batch


def multi_label_condensation(logits: torch.Tensor, config: Config) -> dict[str, torch.Tensor]:
    """
    Convert the output of a multi-label network to a single prediction per pixel.

    In case this decision is ambiguous (multiple classes with a confidence > 0.5 or no class at all), the pixel will be marked as "network_unsure".

    Args:
        logits: The output of the network (batch, class, *).
        config: The configuration of the training.

    Returns: Dictionary with the entries:
        - `predictions`: Predicted labels (batch, *).
        - `confidences`: Corresponding confidences of the prediction (batch, *).
    """
    if (logits >= 0).all() and (logits <= 1).all():
        settings.log_once.warning(
            "The logits seem to be already in the range [0, 1]. Please provide the raw logits of the network and not"
            " the sigmoid activations"
        )

    confidences = logits.sigmoid()
    preds = confidences > 0.5  # [BCHW]
    valid = preds.count_nonzero(dim=1) == 1  # [BHW]

    # We only use predictions if there is exactly one class predicted, the rest will be unsure
    # This is because we are not evaluating a multi-label scenario yet as we want to be comparable to other runs
    label_mapping = LabelMapping.from_config(config)
    predicted_labels = torch.full(
        valid.shape, label_mapping.name_to_index("network_unsure"), dtype=torch.int64, device=valid.device
    )

    # We can use argmax here because we explicitly use only pixels with exactly one predicted class
    predicted_labels[valid] = preds.transpose(0, 1)[:, valid].float().argmax(dim=0)

    confidences = confidences.gather(dim=1, index=predicted_labels.unsqueeze(dim=1)).squeeze(dim=1)
    return {
        "predictions": predicted_labels,
        "confidences": confidences,
    }
