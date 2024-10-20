# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

from htc.settings import settings
from htc.utils.Config import Config


class Benchmark:
    def __init__(self, config_paths: list[Path] | None) -> None:
        self.timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if config_paths is None:
            self.configs = []
            for config_name in ["original", "blosc", "fp16", "gpu-aug", "ring-buffer"]:
                self.configs.append(Config(f"htc_projects/benchmarking/configs/{config_name}.json"))
        else:
            self.configs = [Config(p) for p in config_paths]

    def run_multiple(self, speed_limits: list[str]) -> None:
        errors = []
        for speed_limit in speed_limits:
            errors += self.run_benchmark(speed_limit=speed_limit)

        if len(errors) > 0:
            settings.log.error(f"An error occurred for {len(errors)} jobs:")
            settings.log.error("\n".join(errors))

    def run_benchmark(self, speed_limit: str) -> list[str]:
        device = self._find_semantic_disk()

        # Build the Docker image
        for tag, dockerfile in [("htc-base", "base.Dockerfile"), ("htc", "Dockerfile")]:
            res = subprocess.run(f"docker build --tag {tag} --file {dockerfile} .", shell=True, cwd=settings.src_dir)
            assert res.returncode == 0, "Could not build the base docker image"

        cmd = "docker run -it --rm"
        # We need a privileged container to be able to clear the system memory cache (https://unix.stackexchange.com/a/381441)
        cmd += " --privileged"
        cmd += " -v /proc:/writable_proc --gpus all --shm-size=20gb"
        if speed_limit != "unlimited":
            cmd += f" --device-read-bps {device}:{speed_limit}"

        # We only need access to the semantic dataset
        dataset = settings.datasets.semantic["path_dataset"]
        cmd += f" -v {dataset}:/home/{dataset.name}:ro"
        cmd += f" -e {settings.datasets.semantic['env_name']}=/home/{dataset.name}"

        # We write the benchmarking results to the default results directory
        cmd += f" -v {settings.results_dir}:/home/results"

        # Mount config files if they are outside of this repository
        for config in self.configs:
            config_path = config.path_config
            if not config_path.is_relative_to(settings.htc_package_dir):
                cmd += f" -v {config_path}:{config_path}:ro"

        # Log more often than the default
        cmd += " -e HTC_SYSTEM_MONITOR_REFRESH_RATE=0.15"

        # Container name
        cmd += " htc"

        # Training command
        n_seeds = 3

        errors = []
        for config in self.configs:
            config_path = config.path_config
            if config_path.is_relative_to(settings.htc_package_dir):
                config_path = config_path.relative_to(settings.htc_package_dir)
            config_name = config["config_name"]

            for seed in range(n_seeds):
                run_folder = f"{self.timestring}_benchmarking_{config_name}_{seed}_{speed_limit}"
                cmd_iteration = (
                    f"{cmd} htc training --model image --fold 'fold_all' --run-folder {run_folder} --config"
                    f""" {config_path} --config-extends '{{"seed": {seed}}}'"""
                )

                res = subprocess.run(cmd_iteration, shell=True, cwd=settings.src_dir)
                if res.returncode != 0:
                    errors.append(f"The run {config_name} ({seed = }, {speed_limit = }) was not successful")

        return errors

    def _find_semantic_disk(self) -> str:
        # Find device (disk) which stores the semantic data
        res = subprocess.run(["df", str(settings.data_dirs.semantic)], capture_output=True, text=True)
        assert res.returncode == 0, "Could not find the device where the semantic data is stored"

        match = re.search(r"Mounted on[^/]+(/dev/\w+)", res.stdout)
        assert match is not None, "Could not find the device where the semantic data is stored"
        device = match.group(1)
        settings.log.info(f"Identified the device {device} where the semantic data is stored")

        return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark data loading performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--io-speed",
        type=str,
        nargs="+",
        default=["unlimited"],
        help=(
            "Read speed limits on the device which stores the semantic dataset. This is used to simulate slower IO"
            " speeds. If more than one limit is given, the benchmark is repeated for each speed limit."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Path to one or more config files which should be used in the benchmark. If not set, use the default"
            " benchmark configs (original, blosc, fp16, gpu-aug, ring-buffer)."
        ),
    )
    args = parser.parse_args()

    bench = Benchmark(config_paths=args.config)
    bench.run_multiple(speed_limits=args.io_speed)
