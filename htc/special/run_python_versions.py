# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
from functools import partial
from pathlib import Path

from htc.settings import settings


class PythonVersionUpgrader:
    python_versions = ("3.11", "3.12", "3.13")

    def __init__(self):
        """
        Small helper class to upgrade Python versions in all relevant files.

        Simply adjust the Python versions in this file and then run this script to update all files.
        """
        # Adjust this list whenever a new file with a hardcoded Python version is added or when an existing file is updated
        self.files = {
            "dependencies/base.Dockerfile": [PythonVersionUpgrader.replace_conda],
            "README.md": [PythonVersionUpgrader.replace_conda],
            "README_public.md": [PythonVersionUpgrader.replace_conda],
            "htc/special/run_create_public.py": [
                PythonVersionUpgrader.replace_conda,
                partial(PythonVersionUpgrader.replace_cpython, mode="latest"),
            ],
            ".gitlab-ci.yml": [
                partial(PythonVersionUpgrader.replace_conda, n_matches=2),
                partial(PythonVersionUpgrader.replace_cpython, mode="latest"),
            ],
            "setup.py": [PythonVersionUpgrader.replace_classifiers, PythonVersionUpgrader.replace_requires],
            ".github/workflows/dataset.yml": [partial(PythonVersionUpgrader.replace_workflows, mode="all")],
            ".github/workflows/tests.yml": [partial(PythonVersionUpgrader.replace_workflows, mode="latest")],
            ".github/workflows/release.yml": [partial(PythonVersionUpgrader.replace_workflows, mode="latest")],
            "pyproject.toml": [
                partial(PythonVersionUpgrader.replace_cpython, mode="all"),
                PythonVersionUpgrader.replace_ruff,
            ],
        }

    def upgrade_all(self) -> None:
        for file, functions in self.files.items():
            file = settings.src_dir / file
            assert file.is_file(), f"{file} is not a file"

            for func in functions:
                func(self, file)

    def replace_conda(self, file: Path, n_matches: int = 1) -> None:
        latest = self.python_versions[-1]

        content = file.read_text()
        pattern = re.compile(r"python=\d\.\d+")

        matches = re.findall(pattern, content)
        assert len(matches) == n_matches, (
            f"Unexpected number of Python versions in {file} (expected {n_matches}, got {len(matches)})"
        )

        content = re.sub(pattern, f"python={latest}", content)
        file.write_text(content)

    def replace_cpython(self, file: Path, mode: str) -> None:
        if mode == "all":
            version_line = " ".join([f"cp{v.replace('.', '')}-*" for v in self.python_versions])
        elif mode == "latest":
            version_line = f"cp{self.python_versions[-1].replace('.', '')}-*"
        else:
            raise ValueError(f"Invalid mode {mode}")

        content = file.read_text()
        pattern = re.compile(r"(cp\d+-\*\s*){1," + str(len(self.python_versions)) + "}")

        assert re.search(pattern, content) is not None, f"cp* not found in {file}"
        content = re.sub(pattern, version_line, content)

        file.write_text(content)

    def replace_classifiers(self, file: Path) -> None:
        content = file.read_text()

        matches = list(re.finditer(r"Programming Language :: Python :: \d\.\d+", content))
        assert len(matches) == len(self.python_versions), (
            f"Unexpected Python version listing in {file} (expected {len(self.python_versions)}, got {len(matches)})"
        )

        for match, version in zip(matches, self.python_versions, strict=True):
            content = content[: match.start()] + f"Programming Language :: Python :: {version}" + content[match.end() :]

        file.write_text(content)

    def replace_requires(self, file: Path) -> None:
        earliest = self.python_versions[0]

        content = file.read_text()
        pattern = re.compile(r'python_requires=">=\d\.\d+"')

        assert re.search(pattern, content) is not None, f"python_requires not found in {file}"
        content = re.sub(pattern, f'python_requires=">={earliest}"', content)

        file.write_text(content)

    def replace_workflows(self, file: Path, mode: str) -> None:
        if mode == "all":
            versions = ", ".join([f'"{v}"' for v in self.python_versions])
            version_line = f"python-version: [{versions}]"
        elif mode == "latest":
            version_line = f'python-version: ["{self.python_versions[-1]}"]'
        else:
            raise ValueError(f"Invalid mode {mode}")

        content = file.read_text()

        pattern = re.compile(r"python-version: \[.*\]")
        assert re.search(pattern, content) is not None, f"python-version not found in {file}"
        content = re.sub(pattern, version_line, content)

        file.write_text(content)

    def replace_ruff(self, file: Path) -> None:
        content = file.read_text()

        pattern = re.compile(r'target-version = "py\d+"')
        assert re.search(pattern, content) is not None, f"target-version version not found in {file}"
        content = re.sub(pattern, f'target-version = "py{self.python_versions[0].replace(".", "")}"', content)

        file.write_text(content)


if __name__ == "__main__":
    upgrader = PythonVersionUpgrader()
    upgrader.upgrade_all()
