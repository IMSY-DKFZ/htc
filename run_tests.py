# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import subprocess
import sys

from htc.settings import settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs the tests in this repository. Additional arguments passed to this script are forwarded to pytest,"
            " e.g. htc tests -v passes the -v switch to pytest for immediate test status report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--slow", default=False, action="store_true", help="Include tests which are marked as slow.")
    parser.add_argument("--notebooks", default=False, action="store_true", help="Include notebook tests.")
    parser.add_argument(
        "--cov",
        default=False,
        action="store_true",
        help=(
            "Create a coverage report. Note: coverage reporting is error-prone since it does not work reliable with"
            " multiprocessing. Only add this switch if you already know that all tests pass!"
        ),
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default=None,
        help="Run tests in parallel. Supply the number of cores to use or auto to use all.",
    )
    args, pytest_args = parser.parse_known_args()

    # Add -v to see failures directly
    command = "py.test --doctest-modules --durations=5"
    if len(pytest_args) > 0:
        command += " " + " ".join(pytest_args)

    test_directories = ["tests", "htc", "htc_projects", "tutorials"]
    for paper_dir in sorted((settings.src_dir / "paper").iterdir()):
        # Some paper notebooks are already run by special tests (e.g. tests/paper/test_paper_semantic_files.py)
        if paper_dir.name not in ["MIA2022", "MICCAI2023"]:
            test_directories.append(f"paper/{paper_dir.name}")
    test_directories = " ".join([str(settings.src_dir / d) for d in test_directories])

    if not args.slow:
        command += ' -m "not slow"'

    if args.notebooks:
        command += " --nbval-lax --nbval-current-env"

    if args.cov:
        command += " --cov=htc --cov-report=html"

    if args.parallel is not None:
        command += f" -n {args.parallel} --dist loadscope"

    res = subprocess.run(f"{command} {test_directories}", shell=True)
    sys.exit(res.returncode)
