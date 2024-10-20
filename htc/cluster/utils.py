# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import platform
from pathlib import Path

import numpy as np
import psutil

from htc.cluster.UpdateCluster import UpdateCluster
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.import_extra import requires_extra

try:
    import paramiko

    _missing_library = ""
except ImportError:
    _missing_library = "paramiko"


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


@requires_extra(_missing_library)
def run_jobs(jobs: list[str], dataset_names: list[str] = None) -> None:
    # Make sure the cluster is up-to-date
    with UpdateCluster(dataset_names=dataset_names) as cluster:
        cluster.check_local_and_remote_dirs(remote="cluster")
        f = cluster.update_container_files()
        cluster.generate_runner_htc(image_filename=f)

    sftpURL = settings.cluster_submission_host
    sftpUser = settings.dkfz_userid

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()

    # Automatically add keys without requiring human intervention
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(sftpURL, username=sftpUser)
    except paramiko.SSHException:
        # Sometimes the key cannot be found automatically, try the default location in this case
        ssh.connect(sftpURL, username=sftpUser, key_filename=Path("~/.ssh/cluster").expanduser())

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


def cpu_count() -> int:
    hostname = platform.node()

    if hostname in ["hdf19-gpu16", "hdf19-gpu17", "e230-AMDworkstation"]:
        return 16
    if hostname.startswith(("hdf19-gpu", "e071-gpu")):
        return 12
    elif hostname.startswith("e230-dgx1"):
        return 10
    elif hostname.startswith(("hdf18-gpu", "e132-comp")):
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
