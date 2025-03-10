# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import os
import re
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Self

from rich.progress import Progress, TimeElapsedColumn

from htc.settings import settings
from htc.utils.import_extra import requires_extra

try:
    import paramiko

    _missing_library = ""
except ImportError:
    _missing_library = "paramiko"


class ClusterConnection:
    @requires_extra(_missing_library)
    def __init__(self, dataset_names: list[str] = None):
        """
        Manages the connection to the cluster.

        Args:
            dataset_names: Optional explicit list of dataset names which will for example be used during sync checking or for mounting the available datasets in the container. If not set, all available datasets will be used.
        """
        self.checkpoints_dir = f"{settings.cluster_checkpoints_dir}/{settings.cluster_user_folder}/hsi"
        self.training_dir = f"{self.checkpoints_dir}/training"
        self.host = settings.cluster_worker_host
        self.host_alternative = settings.cluster_worker_host_alternative

        self.shared_folder = Path(f"{settings.cluster_data_dir}/{settings.cluster_shared_folder}")
        self.user_folder = Path(f"{settings.cluster_data_dir}/{settings.cluster_user_folder}/htc")

        self.cluster_dataset_dirs = {}
        self.local_dataset_dirs = {}
        self.network_dataset_dirs = {}
        self.environment_names = {}

        if dataset_names is None:
            dataset_names = [
                name
                for name in settings.datasets.dataset_names
                if settings.datasets.get(name, local_only=True) is not None
            ]

        # For the overlapping datasets, we also need the linked datasets
        if (
            "2023_04_22_Tivita_multiorgan_kidney" in dataset_names
            and "2021_02_05_Tivita_multiorgan_masks" not in dataset_names
        ):
            dataset_names.append("2021_02_05_Tivita_multiorgan_masks")
        if (
            "2021_02_05_Tivita_multiorgan_masks" in dataset_names
            and "2021_02_05_Tivita_multiorgan_semantic" not in dataset_names
        ):
            dataset_names.append("2021_02_05_Tivita_multiorgan_semantic")

        for name in dataset_names:
            entry = settings.datasets.get(name, local_only=True)
            assert entry is not None, (
                f"Cannot continue because the dataset {name} is not available locally but is needed for the requested operation"
            )

            self.cluster_dataset_dirs[name] = self.shared_folder / name
            self.local_dataset_dirs[name] = entry["path_dataset"]
            self.network_dataset_dirs[name] = settings.datasets.network_location(name, path_data=False)
            self.environment_names[name] = entry["env_name"]

        # skip syncing the directories which contains the following keywords in the intermediates directory
        self.skip_intermediate_dirs = ["view_"]

        self.results_folder = f"{settings.cluster_checkpoints_dir}/{settings.cluster_user_folder}/hsi"
        self.remote_datasets_synced = dict.fromkeys(self.local_dataset_dirs.values(), True)

    def __enter__(self) -> Self:
        """
        Create a ssh and ftp connection to the server

        Returns: connected instance of the class

        """
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()

        # Automatically add keys without requiring human intervention
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the worker
        sftpURL = self.host
        sftpUser = settings.dkfz_userid
        try:
            self.ssh.connect(sftpURL, username=sftpUser)
        except TimeoutError:
            settings.log.warning(f"Timeout on host {self.host}. Trying host {self.host_alternative} instead...")
            self.ssh.connect(sftpURL.replace(self.host, self.host_alternative), username=sftpUser)
            self.host = self.host_alternative  # Use as new default
        except paramiko.SSHException as e:
            # Sometimes the key cannot be found automatically, try the default location in this case
            self.ssh.connect(sftpURL, username=sftpUser, key_filename=Path("~/.ssh/cluster").expanduser())

        self.ftp = self.ssh.open_sftp()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Close the ssh and ftp connection to the server

        Args:
            exc_type: string representing the execution type
            exc_value: string representing the execution value
            traceback: show the traceback log of the
        """
        self.ftp.close()
        self.ssh.close()

    @cached_property
    def running_jobs(self) -> list[dict[str, str]]:
        # Login is important as otherwise the bjobs command is not available
        stdin, stdout, stderr = self.ssh.exec_command(
            "bash --login -c \"bjobs -o 'jobid command effective_resreq' -json\""
        )
        exit_status = stdout.channel.recv_exit_status()
        assert exit_status == 0, f"bjobs command failed. Cannot extract the running jobs ({exit_status = })"

        error = stderr.read().decode()
        if error != "":
            settings.log.error(f"The command to extract information about running jobs returned an error:\n{error}")

        jobs_data = json.loads(stdout.read().decode())
        jobs = []
        for record in jobs_data.get("RECORDS", []):
            args = dict(re.findall(r"--(\S+)\s+(\S+)", record["COMMAND"]))
            args["config_name"] = Path(args["config"]).name
            jobs.append(args)

            if hasattr(self, "memory") and self.memory is None:
                self.memory = re.search(r"gmem=(\d+.?\d*)", record["EFFECTIVE_RESREQ"]).group(1)
                settings.log.info(
                    f"Found memory requirement of {self.memory} MB from a running job (will be used for resubmission)"
                )

        settings.log.info(f"Found {len(jobs)} running jobs")
        return jobs

    def check_training_dir(self) -> bool:
        try:
            self.ftp.stat(self.training_dir)
            return True
        except FileNotFoundError:
            msg = (
                f"Could not find the training directory {self.training_dir} on the cluster. Please make sure it exists"
                f" ([var]{settings.cluster_user_folder = }[/])"
            )
            if "CLUSTER_FOLDER" in os.environ:
                msg += (
                    "\nNote that the environment variable [var]CLUSTER_FOLDER[/] is set to"
                    f" [var]{os.getenv('CLUSTER_FOLDER')}[/]. Please make sure that this is correct"
                )
            settings.log.error(msg)

            return False

    def get_local_file_stats(self, path: str, search_pattern: str) -> dict[Path, dict[str, int | float | Path]]:
        """

        Check the status of the local (or mounted network drive) files. This function returns the name of the files and the file stats.
        This information is used to mark which files will be uploaded.

        Args:
            path: path to the local files
            search_pattern: pattern to the files which should be looked at

        Returns: list of names and corresponding file stats

        """
        out = self.run_command(
            command=f"find {path} -name '{search_pattern}' -type f -printf '%p:%s:%T@,'",
            task="finding local files",
            capture_output=True,
            print_success=False,
        ).decode("utf-8")
        file_stats = {}

        for path in out.split(",")[:-1]:
            stats = path.split(":")
            file_stats[Path(stats[0])] = {"size": int(stats[1]), "mtime": float(stats[2]), "path": Path(stats[0])}

        file_stats = dict(sorted(file_stats.items()))

        return file_stats

    def get_cluster_file_stats(self, path: str, search_pattern: str) -> dict[Path, dict[str, int | float | Path]]:
        """

        Check the status of the cluster files. This function returns the name of the files and the file stats.
        This information is used to mark which files will be uploaded.

        Args:
            path: path to the network files
            search_pattern: pattern to the files which should be looked at

        Returns: list of names and corresponding file stats

        """
        stdin, stdout, stderr = self.ssh.exec_command(
            f"find {path} -name '{search_pattern}' -type f -printf '%p:%s:%T@,'"
        )
        out = stdout.readlines()

        file_stats = {}

        for paths in out:
            for path in paths.split(",")[:-1]:
                stats = path.split(":")
                file_stats[Path(stats[0])] = {"size": int(stats[1]), "mtime": float(stats[2]), "path": Path(stats[0])}

        file_stats = dict(sorted(file_stats.items()))

        return file_stats

    def check_local_and_remote_dirs(self, remote: str = "cluster") -> None:
        """
        Check if the local and remote folder (can be the cluster or network drive) are in sync.

        Args:
            remote: Set to either "cluster" or "network" to specify the remote location.
        """
        # Request sudo already in the beginning so that the script does not have to wait for the user in the middle
        self.run_command("sudo ls", task="request sudo", capture_output=True, print_success=False)

        with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress:
            task_datasets = progress.add_task(f"Checking datasets on the {remote}", total=len(self.local_dataset_dirs))
            for dataset_name, local_dataset_dir in self.local_dataset_dirs.items():
                # set the correct remote dataset directory path depending upon the remote type
                if remote == "cluster":
                    remote_dataset_dir = self.cluster_dataset_dirs[dataset_name]
                elif remote == "network":
                    remote_dataset_dir = self.network_dataset_dirs[dataset_name]
                else:
                    raise ValueError(
                        f"The argument remote has been set to an invalid value {remote}. Valid values are: cluster or"
                        " network"
                    )

                info_logs = []
                subdirs = sorted(Path(local_dataset_dir / "intermediates").glob("*"))

                # compare the intermediate files
                for subdir in subdirs:
                    skip_entry = any(
                        skip_intermediate_dir in subdir.name for skip_intermediate_dir in self.skip_intermediate_dirs
                    )

                    if skip_entry:
                        continue

                    local_intermediate_dir = local_dataset_dir / "intermediates" / subdir.name
                    remote_intermediate_dir = remote_dataset_dir / "intermediates" / subdir.name

                    local_intermediate_stats = self.get_local_file_stats(
                        path=local_intermediate_dir, search_pattern="*"
                    )

                    if remote == "cluster":
                        remote_intermediate_stats = self.get_cluster_file_stats(
                            path=remote_intermediate_dir, search_pattern="*"
                        )
                    elif remote == "network":
                        remote_intermediate_stats = self.get_local_file_stats(
                            path=remote_intermediate_dir, search_pattern="*"
                        )

                    # if the dataset is determined to be not in sync then there is no need to check the rest of the folders
                    self.remote_datasets_synced[local_dataset_dir] = (
                        self.check_file_stats(
                            task="intermediates",
                            local_files_stats=local_intermediate_stats,
                            remote_files_stats=remote_intermediate_stats,
                            dataset=local_dataset_dir.name,
                            current_dir=subdir,
                        )
                        if self.remote_datasets_synced[local_dataset_dir]
                        else self.remote_datasets_synced[local_dataset_dir]
                    )

                    info_logs.append({"directory": subdir, "num_files": len(local_intermediate_stats)})

                info_text = f"\nChecked intermediates for dataset {local_dataset_dir.name}, more info:"
                for info_log in info_logs:
                    info_text += f"\n(Directory: {info_log['directory']}, Number of files: {info_log['num_files']})"

                settings.log.info(info_text)

                progress.advance(task_datasets)

    @staticmethod
    def check_file_stats(
        task: str, local_files_stats: dict, remote_files_stats: dict, dataset: str, current_dir: Path
    ) -> bool:
        """
        Compare the file stats of local and remote files (can be the cluster or network drives). Check the file names and sizes.
        In the future, compare the modification times as well.

        Args:
            task: description of the task for which file stats are being compared
            local_files_stats: the file stats of the local files
            remote_files_stats: the file stats of the files on the remote
            dataset: the dataset which is being checked currently
            current_dir: the path to the current directory being scanned

        Returns: a boolean specifying if the local and network datasets are in sync

        """
        if len(local_files_stats) != len(remote_files_stats):
            raise ValueError(
                f"The local version of the dataset {dataset} contains differing number of {task} compared to the"
                " remote version. The network version has to be synced to the cluster in the user directory (files:"
                f" {len(local_files_stats)} vs {len(remote_files_stats)}, path: {current_dir})."
            )

        # compare sizes and modification times of the annotations to check if the annotations have been modified
        for local_file, remote_file in zip(local_files_stats.keys(), remote_files_stats.keys(), strict=True):
            local_file_stats = local_files_stats[local_file]
            remote_file_stats = remote_files_stats[remote_file]

            if local_file.name != remote_file.name:
                raise ValueError(f"The local file {local_file.name} is not the same as remote file {remote_file.name}.")

            if local_file_stats["size"] != remote_file_stats["size"]:
                raise ValueError(
                    f"The {task} file at local path {local_file} is different between local and"
                    f" cluster versions of the file (sizes: {local_file_stats['size']} vs"
                    f" {remote_file_stats['size']})."
                )

            # commenting out the modification time comparison for now, as we can't ensure that the modification times stay the same between workstations
            # if math.ceil(local_file_stats["mtime"]) != math.ceil(remote_file_stats["mtime"]):
            #     settings.log.info(f"The {task} file for the image name {local_file} is different between local and cluster versions of the file (mtime: {math.ceil(local_file.stat().st_mtime)} vs {math.ceil(remote_file_stats['mtime'])}).")
            #     return False

        return True

    def get_rsync_command(self, local_dataset_dir: str | Path, remote_dataset_dir: str) -> str:
        """
        Formulates the rsync command to run, given a local path and the remote path

        Args:
            local_dataset_dir: the path to the local dataset
            remote_dataset_dir: the path to the remote dataset

        Returns:
            command: the rsync command

        """
        # set the 775 permissions for rsynced files, so that everyone can use the files
        rsync_base = (
            "rsync --recursive --specials --times --copy-links --delete --info=progress2 --omit-dir-times"
            " --chmod=775 -p"
        )

        if "2021_03_30_Tivita_studies" in str(local_dataset_dir):
            # Not every HSI image of the studies directory is also stored in the intermediates, so we have to sync the raw data as well
            command = (
                f"{rsync_base} --include='*/' --include='*.dat' --include='*.log' --include='*.xml'"
                " --include='*RGB-Image' --include='annotations/*' --include='intermediates/**' --exclude='*'"
                f" {local_dataset_dir}/ {settings.dkfz_userid}@{self.host}:{remote_dataset_dir}/"
            )
        else:
            command = (
                f"{rsync_base} --exclude='.git' --include='*_RGB-Image.png' --exclude='*.png*' --exclude='*.dat*'"
                " --exclude='*.dcm*' --exclude='*.tif*' --exclude='*.html*' --exclude='data_original'"
                f" {local_dataset_dir}/ {settings.dkfz_userid}@{self.host}:{remote_dataset_dir}/"
            )

        return command

    @staticmethod
    def run_command(command: str, task: str, capture_output: bool = False, print_success: bool = True) -> bytes | int:
        """
        Function for executing a command on the cluster

        Args:
            command: statement to be executed
            task: description of the task that the statement is executing
            capture_output: should the output be captured
            print_success: print a success method

        Returns: the output if capture output flag is set, otherwise the return code

        """
        res = subprocess.run(command, shell=True, capture_output=capture_output, cwd=settings.src_dir)

        if res.returncode != 0:
            # if the find command could not find the directory then return stdout
            if res.stderr is not None and b"No such file or directory" in res.stderr:
                return res.stdout if capture_output else res.returncode

            settings.log.error(f"Could not execute the command for {task}")
            settings.log.error("Exiting script...")
            exit(1)
        else:
            if print_success:
                settings.log.info(f"Successfully executed the command for {task}")

            return res.stdout if capture_output else res.returncode
