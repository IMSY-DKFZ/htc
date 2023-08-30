# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # The purpose of this script is to mount every read-only volume as copy-on-write so that we can still write to the volumes but the changed files are stored inside the Docker container (we never write to the read-only-volumes)
    # Based on: https://stackoverflow.com/a/54465442
    tmp_overlay = Path("/tmp/overlay")
    tmp_overlay.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(f"mount -t tmpfs tmpfs {tmp_overlay}", shell=True)
    if process.returncode != 0:
        print("WARNING: Could not create the overlay layer. Access to the mounted directories will likely not work")

    for mount_dir_read in sorted(Path("/home").iterdir()):
        if mount_dir_read.is_dir() and mount_dir_read.name.endswith("-ro"):
            mount_name = mount_dir_read.name.removesuffix("-ro")
            tmp_upper = tmp_overlay / mount_name / "upper"
            tmp_work = tmp_overlay / mount_name / "work"
            tmp_upper.mkdir(parents=True, exist_ok=True)
            tmp_work.mkdir(parents=True, exist_ok=True)

            mount_dir_write = mount_dir_read.with_name(mount_name)
            mount_dir_write.mkdir(parents=True, exist_ok=True)

            process = subprocess.run(
                "mount -t overlay overlay -o"
                f" lowerdir={mount_dir_read},upperdir={tmp_upper},workdir={tmp_work} {mount_dir_write}",
                shell=True,
            )
            if process.returncode != 0:
                print(
                    f"WARNING: Could not mount the writing layer for {mount_dir_read}. Access to this directory will"
                    " likely not work"
                )

    # Run the main command
    process = subprocess.run(sys.argv[1:])
    sys.exit(process.returncode)
