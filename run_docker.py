# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Start any htc command in a Docker container. You do not need to have htc installed to run this script (but"
            " `python-dotenv` is required if you want to load an .env file from the same directory as this script). Please note that per default all results are stored in the container (in"
            " /home/results) and will be deleted as soon as you exit the container. If you want to change this"
            " behavior, you can set the environment variable `PATH_HTC_DOCKER_RESULTS`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-name",
        default=None,
        type=str,
        help="Name of the existing Docker image to use (no build will be performed if this argument is given). Without this argument, the local registry is used while performing a rebuild if necessary.",
    )
    parser.add_argument(
        "cmd_args", nargs=argparse.REMAINDER, help="Command and arguments which will be run in the Docker container"
    )
    args = parser.parse_args()

    # Read the .env file from the repository root (if available)
    file_dir = Path(__file__).parent
    dotenv_path = file_dir / ".env"
    if dotenv_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=dotenv_path)
        except ImportError:
            print(
                f"WARNING: A .env file was found at {dotenv_path}, but python-dotenv is not installed. No environment variables are loaded from the .env file. Run `pip install python-dotenv` to make use of your environment variables."
            )

    volumes = []
    envs = []

    # Per default, results are stored inside the container to avoid damage of existing result files
    if "PATH_HTC_DOCKER_RESULTS" in os.environ:
        # But this behavior can be changed by the user
        volumes.append("$PATH_HTC_DOCKER_RESULTS:/home/results")
        envs.append("PATH_HTC_DOCKER_RESULTS=/home/results")

    # Mount volumes and set environment variables for all available datasets/result directories
    symlinks = []
    for env_name in os.environ.keys():
        if not env_name.upper().startswith(("PATH_TIVITA", "PATH_HTC_RESULTS")):
            continue

        if path_env := os.getenv(env_name, False):
            path_env = Path(path_env).expanduser().absolute()

            # Check for symlinks in the first level of the dataset folder and automatically mount it
            for f in sorted(path_env.iterdir()):
                if f.is_symlink():
                    symlinks.append(f.readlink())

            volumes.append(f"{path_env}:/home/{path_env.name}-ro:ro")
            envs.append(f"{env_name}=/home/{path_env.name}")

    # Additional mount points are e.g. useful to include symbolic link locations in the container (if not done automatically)
    if "HTC_DOCKER_MOUNTS" in os.environ:
        mounts = os.environ["HTC_DOCKER_MOUNTS"].split(";")
        for mount in mounts:
            if not mount.endswith(":ro"):
                mount = f"{mount}:ro"
            volumes.append(mount)

    for s in symlinks:
        volumes.append(f"{s}:{s}:ro")

    compose_file_cmd = ["-f", "dependencies/docker-compose.yml"]

    # Override file for environment settings
    override_file = file_dir / "dependencies" / "docker-compose.override.yml"
    if len(volumes) > 0:
        volumes = "\n      - ".join(volumes)
        envs = "\n      - ".join(envs)

        override_yml = f"""
services:
  {"imsy-htc" if args.image_name is None else "custom_image_name"}:
    environment:
      - {envs}
    volumes:
      - {volumes}
"""

        with override_file.open("w") as f:
            f.write(override_yml)

        compose_file_cmd.append("-f")
        compose_file_cmd.append(f"dependencies/{override_file.name}")
    else:
        if override_file.exists():
            override_file.unlink()

    override_gpu_file = file_dir / "dependencies" / "docker-compose_gpu.override.yml"
    try:
        subprocess.check_output("nvidia-smi")
        override_gpu_yml = f"""
services:
  {"imsy-htc" if args.image_name is None else "custom_image_name"}:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

        with override_gpu_file.open("w") as f:
            f.write(override_gpu_yml)

        compose_file_cmd.append("-f")
        compose_file_cmd.append(f"dependencies/{override_gpu_file.name}")
    except Exception:
        if override_gpu_file.exists():
            override_gpu_file.unlink()

    # Override file for custom image name
    override_service_file = file_dir / "dependencies" / "docker-compose_service.override.yml"
    if args.image_name is not None:
        override_service_yml = f"""\
services:
  custom_image_name:
    image: {args.image_name}
    container_name: {args.image_name}
    network_mode: host
    shm_size: 10gb
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    cap_add:
      - SYS_ADMIN
    security_opt:
      - apparmor:unconfined
"""

        with override_service_file.open("w") as f:
            f.write(override_service_yml)

        compose_file_cmd.append("-f")
        compose_file_cmd.append(f"dependencies/{override_service_file.name}")
    else:
        if override_service_file.exists():
            override_service_file.unlink()

    compose_cmd = ["docker", "compose", *compose_file_cmd]
    if args.image_name is not None:
        subprocess.run([*compose_cmd, "run", "--rm", "custom_image_name", *args.cmd_args], cwd=file_dir, check=True)
    else:
        subprocess.run([*compose_cmd, "build"], cwd=file_dir, check=True)
        subprocess.run([*compose_cmd, "run", "--rm", "imsy-htc", *args.cmd_args], cwd=file_dir, check=True)
