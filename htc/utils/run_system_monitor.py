# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
import os
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

    log_path = args.output_dir / f"system_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    log_path_finished = Path(str(log_path).replace("running_", ""))
    refresh_rate = float(os.getenv("HTC_SYSTEM_MONITOR_REFRESH_RATE", "5"))

    def log_file(total: dict) -> None:
        if log_path.parent.exists():
            with log_path.open("w") as f:
                json.dump(total, f)
        elif log_path_finished.parent.exists():
            # Try again without the running_ prefix in case the folder was already renamed (happens when the run is finished)
            with log_path_finished.open("w") as f:
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
                log_file(total)

            count += 1

            if args.pid is not None and psutil.pid_exists(args.pid):
                time.sleep(refresh_rate)
            else:
                break
    except KeyboardInterrupt:
        pass

    # It is possible that the monitoring gets interrupted before the current event is added to every list in the dict
    # In that case, we remove the last elements from the lists which have more than the smallest list
    lengths = [len(values) for values in total.values()]
    if len(set(lengths)) != 1:
        for k, v in total.items():
            total[k] = v[: min(lengths)]

    log_file(total)
