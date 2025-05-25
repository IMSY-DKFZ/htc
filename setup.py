# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
import re
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.sdist import sdist
from torch.utils import cpp_extension


class ExtendedSdist(sdist):
    def make_distribution(self):
        # The requirements files need to be part of the source distribution as otherwise setuptools can't build the package
        # The wheels don't need the requirements files as they have already been parsed and added to the package metadata
        self.filelist.append(requirements)
        self.filelist.append(requirements_extra)
        super().make_distribution()


def parse_requirements(requirements_file: Path) -> list[str]:
    """
    Parses an requirements file into a list of dependencies.

    Args:
        requirements_file: Path to the requirements file.

    Returns: List of dependencies with version information (e.g. ==1.12.1) but without extra specifications (e.g. +cu116).
    """
    req = []

    for line in requirements_file.read_text("utf-8").splitlines():
        match = re.search(r"^\w+[^+@]*", line)
        if match is not None:
            lib = match.group(0)

            req.append(lib)

    return req


def parse_readme(readme_file: Path) -> str:
    """
    Reads and adjusts a readme file so that it can be used for PyPi.

    Args:
        readme_file: Path to the readme file.

    Returns: Readme as string with adjusted links.
    """
    readme = readme_file.read_text("utf-8")

    # Make local anchors and links to files absolute since they don't work on PyPi
    readme = re.sub(r"\(\./([^)]+)\)", r"(https://github.com/IMSY-DKFZ/htc/tree/main/\1)", readme)
    readme = re.sub(r"\(#([^)]+)\)", r"(https://github.com/IMSY-DKFZ/htc/#\1)", readme)

    return readme


repo_root = Path(__file__).parent
requirements = "dependencies/requirements.txt"
requirements_extra = "dependencies/requirements-extra.txt"

source_files = sorted(repo_root.rglob("htc/cpp/*.cpp"))
source_files = [str(f.relative_to(repo_root)) for f in source_files]

if os.name == "nt":
    compiler_flags = ["/O2", "/std:c++20"]
else:
    compiler_flags = ["-O3", "-std=c++2a"]

setup(
    name="imsy-htc",
    version="0.0.21",
    # We are using find_namespace_packages() instead of find_packages() to resolve this deprecation warning: https://github.com/pypa/setuptools/issues/3340
    packages=find_namespace_packages(include=["htc*"]),
    author="Division of Intelligent Medical Systems, DKFZ",
    license="MIT",
    url="https://github.com/imsy-dkfz/htc",
    description="Framework for automatic classification and segmentation of hyperspectral images.",
    keywords=[
        "surgical data science",
        "open surgery",
        "hyperspectral imaging",
        "organ segmentation",
        "semantic scene segmentation",
        "deep learning",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    long_description=parse_readme(repo_root / "README.md"),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=parse_requirements(repo_root / requirements),
    extras_require={
        "extra": parse_requirements(repo_root / requirements_extra),
    },
    package_data={"": ["*.json", "*.h", "*.js"]},
    entry_points={
        "console_scripts": ["htc=htc.cli:main"],
    },
    ext_modules=[
        cpp_extension.CppExtension(
            name="htc._cpp",
            sources=source_files,
            extra_compile_args=compiler_flags,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension, "sdist": ExtendedSdist},
)
