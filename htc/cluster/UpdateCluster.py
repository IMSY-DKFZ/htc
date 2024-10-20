# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from datetime import datetime
from io import BytesIO
from pathlib import Path

from rich.progress import Progress, TimeElapsedColumn

from htc.cluster.ClusterConnection import ClusterConnection
from htc.settings import settings


class UpdateCluster(ClusterConnection):
    """
    This class is responsible for creating an environment, such that the user can submit jobs to the cluster. It works in the following steps:
        - Making sure that the local datasets are in sync with shared folder on the cluster
        - If they aren't, it provides the functionality for syncing local datasets to the user folder on the cluster
        - Creating and uploading the apptainer container which contains the environment for running the code on cluster
        - Creating runner_htc script for running the jobs on the cluster
    """

    def update_user_cluster(self):
        """
        Update the user folder on the cluster, if local datasets are not in sync with the shared folder on the cluster
        This can only work if the sync_user flag is set through command line argument
        """
        with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress:
            datasets = progress.add_task("Updating datasets in user folder", total=len(self.local_dataset_dirs))
            for dataset_name, local_dataset_dir in self.local_dataset_dirs.items():
                cluster_dataset_dir = self.cluster_dataset_dirs[dataset_name]

                # set the cluster dir as the user cluster dir, in case the local and shared cluster versions of the dataset are not in sync
                cluster_dataset_dir = self.user_folder / cluster_dataset_dir.name

                # if the dataset is already synced with the shared cluster files than skip the dataset
                if self.remote_datasets_synced[local_dataset_dir]:
                    continue

                settings.log.warning(
                    f"The dataset {local_dataset_dir} will be synced with your user folder on the cluster. This is"
                    " done, as your local dataset version is different from the one in the shared folder on the"
                    " cluster. We recommend that you should delete the user folder for this dataset after carrying out"
                    f" the trainings. The dataset is located at {cluster_dataset_dir}."
                )

                # sync the local version to cluster
                self.run_command(
                    command=(
                        self.get_rsync_command(
                            local_dataset_dir=local_dataset_dir, remote_dataset_dir=cluster_dataset_dir
                        )
                    ),
                    task=f"transferring the dataset {dataset_name} from local to the cluster",
                )

                self.cluster_dataset_dirs[dataset_name] = cluster_dataset_dir / cluster_dataset_dir.name

                progress.advance(datasets)

    def update_container_files(self) -> str:
        """
        Create docker and apptainer containers and upload them to the cluster.

        Returns: Filename of the apptainer container uploaded to the cluster so that it can be referenced again in the runner_htc script.
        """
        # Check docker and apptainer installation. If these tools are not found on the system than they will be installed
        self.run_command(
            command=f"bash {Path(__file__).parent / 'container_tools_installation.sh'}",
            task="installing apptainer and docker on the local system",
            print_success=False,
        )

        # Request sudo already in the beginning so that the script does not have to wait for the user in the middle
        self.run_command("sudo ls", task="request sudo", capture_output=True, print_success=False)

        # Build docker and apptainer container
        self.run_command(
            command="docker build --tag htc-base --file base.Dockerfile .",
            task="building the htc-base container",
        )
        self.run_command(
            command="docker build --tag htc-cluster --file htc/cluster/Dockerfile .",
            task="building the htc-cluster command",
        )
        self.run_command(
            command="apptainer build --force ~/htc.sif docker-daemon:htc-cluster:latest",
            task="converting docker container to apptainer container",
        )

        # Transferring the built container
        image_filename = f'htc_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.sif'
        self.run_command(
            command=(
                "rsync --recursive --specials --times --copy-links --delete --info=progress2 ~/htc.sif"
                f" {settings.dkfz_userid}@{self.host}:/home/{settings.dkfz_userid}/{image_filename}"
            ),
            task="transferring the apptainer container to cluster",
        )

        return image_filename

    def generate_runner_htc(self, image_filename: str) -> None:
        """
        Generates the runner_htc script and transfers it to the cluster.

        Args:
            image_filename: The name of the apptainer image which has been transferred to the cluster.
        """
        dataset_env_paths = [
            f"PATH_{local_dataset_dir}={self.cluster_dataset_dirs[local_dataset_dir].absolute()}"
            for local_dataset_dir in self.local_dataset_dirs
        ]
        dataset_env_paths = "\n".join(dataset_env_paths)

        container_exports = [
            f'export {self.environment_names[local_dataset_dir]}="/home/{local_dataset_path.name}"'
            for local_dataset_dir, local_dataset_path in self.local_dataset_dirs.items()
        ]
        container_exports = "\n".join(container_exports)

        apptainer_command = "apptainer exec "
        for local_dataset_dir, local_dataset_path in self.local_dataset_dirs.items():
            apptainer_command += f"--bind $PATH_{local_dataset_dir}:/home/{local_dataset_path.name} "
        apptainer_command += '--bind $PATH_RESULTS:/home/results --nv $IMAGE_FILENAME "$@"'

        runner_htc_script = f"""\
#!/bin/bash
# This is an automatically generated file, use the python script run_update_cluster.py for generation

# These variables get replaced automatically by the run_update_cluster.py script
{dataset_env_paths}
PATH_RESULTS={self.results_folder}
IMAGE_FILENAME={image_filename}

# Avoid that too many threads are used (some tools use all system threads)
OMP_NUM_THREADS=2

if [ ! -d $PATH_RESULTS ]; then
    echo "The results directory does not exist. Creating one at $PATH_RESULTS"
    mkdir --parents $PATH_RESULTS
    chmod -R 750 $PATH_RESULTS
fi

# Run the container
{container_exports}
{apptainer_command}

# Remove core files in the home directory (they require a lot of space and lead to 'cluster_home quota exceeded' warnings)
rm -f core*
"""

        # Copy the file to the cluster
        self.ftp.putfo(BytesIO(runner_htc_script.encode()), "runner_htc.sh")
        self.ftp.chmod("runner_htc.sh", 0o755)
