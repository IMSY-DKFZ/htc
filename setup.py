# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
import re
from pathlib import Path

from setuptools import find_namespace_packages, setup
from torch.utils import cpp_extension


def read_file(path: Path) -> str:
    """
    Reads the content of a file into a string.

    Args:
        path: Path to the file to read.

    Returns:
        The content of the file.
    """
    with path.open() as f:
        content = f.read()

    return content


def parse_requirements(requirements_file: Path) -> list[str]:
    """
    Parses an requirements file into a list of dependencies.

    Args:
        requirements_file: Path to the requirements file.

    Returns: List of dependencies with version information (e.g. ==1.12.1) but without extra specifications (e.g. +cu116).
    """
    req = []

    for line in read_file(requirements_file).splitlines():
        match = re.search(r"^\w+[^+]*", line)
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
    readme = read_file(readme_file)

    # Make local anchors and links to files absolute since they don't work on PyPi
    readme = re.sub(r"\(\./([^)]+)\)", r"(https://github.com/IMSY-DKFZ/htc/tree/main/\1)", readme)
    readme = re.sub(r"\(#([^)]+)\)", r"(https://github.com/IMSY-DKFZ/htc/#\1)", readme)

    return readme


repo_root = Path(__file__).parent

source_files = sorted(repo_root.rglob("htc/cpp/*.cpp"))
source_files = [str(f.relative_to(repo_root)) for f in source_files]

if os.name == "nt":
    compiler_flags = ["/O2", "/std:c++20"]
else:
    compiler_flags = ["-O3", "-std=c++2a"]

setup(
    name="imsy-htc",
    version="0.0.7",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
    python_requires=">=3.9",
    install_requires=parse_requirements(repo_root / "requirements.txt"),
    extras_require={
        "extra": parse_requirements(repo_root / "requirements-extra.txt"),
    },
    include_package_data=True,
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
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
