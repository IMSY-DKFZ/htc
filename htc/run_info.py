# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
import subprocess
from importlib import metadata

from rich import print
from torch.utils import collect_env

from htc.settings import settings

if __name__ == "__main__":
    print("[b]htc framework[/]")
    print("- version: " + metadata.version("imsy-htc"))
    print("- url: " + metadata.metadata("imsy-htc")["Home-page"])
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        git_commit = "(no repository)"
    except Exception:
        git_commit = "(git not found)"
    print(f"- git commit: {git_commit}")

    print("\n[b]User settings:[/]")
    if settings.user_settings_path.exists():
        print(f"Found user settings at {settings.user_settings_path}:")
        with settings.user_settings_path.open() as f:
            print(f.read())
    else:
        print(
            "No user settings found. If you want to use your user settings to specify environment variables, please"
            f" create the file {settings.user_settings_path} and add your environment variables, for example:\nexport"
            ' PATH_HTC_NETWORK="/path/to/your/network/dir"\nexport'
            ' PATH_Tivita_my_dataset="~/htc/Tivita_my_dataset:shortcut=my_shortcut"'
        )

    print("\n[b].env settings:[/]")
    if settings.dotenv_path.exists():
        print(f"Found .env file at {settings.dotenv_path}:")
        with settings.dotenv_path.open() as f:
            print(f.read())
    else:
        print(
            "No .env file found. If you cloned the repository and installed the htc framework in editable mode, you"
            f" can create a .env file in the repository root (more precisely, at {settings.dotenv_path}) and fill it"
            ' with variables, for example:\nexport PATH_HTC_NETWORK="/path/to/your/network/dir"\nexport'
            ' PATH_Tivita_my_dataset="~/htc/Tivita_my_dataset:shortcut=my_shortcut"'
        )

    print("\n[b]Environment variables:[/]")
    for key in os.environ.keys():
        if key.startswith(settings.known_envs):
            print(f"{key}={os.environ[key]}")

    print("\n[b]Datasets:[/]")
    print(f"{settings.data_dirs}\n")

    print("\n[b]Other directories:[/]")
    print(repr(settings.results_dir))
    print(repr(settings.intermediates_dir_all))
    print(f"src_dir={settings.src_dir}")
    print(f"htc_package_dir={settings.htc_package_dir}")
    print(f"htc_projects_dir={settings.htc_projects_dir}")

    print("\n[b]System:[/]")
    collect_env.main()
