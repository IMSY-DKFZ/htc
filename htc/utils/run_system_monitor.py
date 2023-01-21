# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import GPUtil
import psutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logs utilization of system resources (CPU, RAM, GPU) as long as this program is running",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Output directory for the log file",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Process ID to watch. This is used to determine when to stop collecting statistics",
    )

    args = parser.parse_args()

    total = {"time": [], "main_memory": [], "cpus_load": [], "gpus_memory": [], "gpus_load": []}

    log_path = args.output_dir / f'system_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'

    def log_file(total: dict, log_path: Path) -> None:
        if log_path.parent.exists():
            with log_path.open("w") as f:
                json.dump(total, f)
        else:
            # If the parent folder does not exist anymore, stop logging
            exit(0)

    try:
        count = 0
        while True:
            main_memory = psutil.virtual_memory().used / psutil.virtual_memory().total
            cpus_load = psutil.cpu_percent(percpu=True)
            cpus_load = [cpu_load / 100 for cpu_load in cpus_load]
            gpus = GPUtil.getGPUs()

            total["time"].append(time.time())
            total["main_memory"].append(main_memory)
            total["cpus_load"].append(cpus_load)
            total["gpus_memory"].append([gpu.memoryUtil for gpu in gpus])
            total["gpus_load"].append([gpu.load for gpu in gpus])

            if count % 100 == 0:
                # Don't log too often to reduce resource consumption of this script
                log_file(total, log_path)

            count += 1

            if args.pid is not None and psutil.pid_exists(args.pid):
                time.sleep(5)
            else:
                break
    except KeyboardInterrupt:
        log_file(total, log_path)
        exit(0)

    try:
        log_file(total, log_path)
    except FileNotFoundError:
        # Try again without the running_ prefix in case the folder was already renamed (happens when the run is finished)
        log_file(total, str(log_path).replace("running_", ""))
