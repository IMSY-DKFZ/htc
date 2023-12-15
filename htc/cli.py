# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import sys

from rich import print

from htc.settings import settings
from htc.utils.general import subprocess_run


def main() -> int:
    # General entry point for all run_*.py script files

    assert len(sys.argv) > 0, "At least the program name must be supplied as argument"

    # Collect all scripts in this repository
    scripts = []
    for path in sorted(settings.src_dir.rglob("run*.py")):
        module_path = path.relative_to(settings.src_dir)  # e.g. cameras/data/run_cam_dataset.py
        module_name = str(module_path.parent).replace("/", ".")  # e.g. 'cameras.data'
        name = path.stem.removeprefix("run_")  # e.g. cam_dataset (from run_cam_dataset.py)

        if module_name == ".":
            full_name = name  # e.g. tests
        else:
            full_name = f"{module_name}.{name}"  # e.g. cameras.data.cam_dataset

        scripts.append({
            "name": name,
            "full_name": full_name,
            "path": path,
            "module_path": module_path,
        })

    scripts = sorted(scripts, key=lambda x: x["full_name"])
    assert len(scripts) > 0, "No scripts found in the repository"

    exitcode = 0
    if len(sys.argv) == 1:
        print(
            "This is a direct entry point for all the scripts in this repository (all Python files which start with"
            " run_*). For example, you can start the training of a model via"
        )
        print("[cyan]htc training --model image --fold fold_P048,P057,P058 --run-folder test[/]\n")

        print(
            "It is usually not necessary to supply the full module path as long as the name is unique. For example,"
            " [cyan]htc training[/] is equivalent to [cyan]htc htc.models.training[/] since there is no other script"
            " file with this name.\n"
        )

        print("The following script files were found in this repository:")
        for s in scripts:
            print(f'{s["full_name"]} [dim](script path = {s["module_path"]})[/]')
    else:
        needle = sys.argv[1]

        # First try to find a match by name
        candidates = [s for s in scripts if s["name"] == needle]

        # Then try full name (if nothing was found)
        if len(candidates) == 0:
            candidates = [s for s in scripts if s["full_name"].endswith(needle)]

        if len(candidates) == 0:
            print(f"Could not find a matching script for {needle}")
            exitcode = 1
        elif len(candidates) == 1:
            path = candidates[0]["path"]
            assert path.exists() and path.is_file(), f"Cannot find the script {path}"

            process = subprocess_run([sys.executable, path, *sys.argv[2:]])
            exitcode = process.returncode
        else:
            print(f"Found more than one candidate script for [cyan]{needle}[/]:")
            for s in candidates:
                print(f'{s["full_name"]} [dim](script path = {s["module_path"]})[/]')

            exitcode = 1

    return exitcode
