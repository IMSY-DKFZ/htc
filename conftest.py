# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import doctest
import re

import pytest


def pytest_addoption(parser):
    """Add section to configuration files."""
    parser.addini(
        "serial_notebooks",
        type="linelist",
        help="List of notebooks which should be executed sequentially (e.g. because they need the GPU).",
        default=[],
    )
    parser.addini(
        "blacklisted_notebooks",
        type="linelist",
        help="List of notebooks which should be skipped for testing (e.g., because they need access to special files).",
        default=[],
    )
    parser.addini(
        "serial_doctests",
        type="linelist",
        help=(
            "List of files where the corresponding doctests should be executed sequentially (e.g. because they need the"
            " GPU)."
        ),
        default=[],
    )


blacklisted_notebooks = {}


def pytest_collectstart(collector: pytest.File):
    global blacklisted_notebooks
    if collector.path and collector.nodeid in collector.config.getini("serial_doctests"):
        collector.add_marker("serial")

    # Test settings for nbval (Jupyter notebook testing)
    if collector.path and collector.path.suffix == ".ipynb":
        if len(blacklisted_notebooks) == 0:
            for notebook in collector.config.getini("blacklisted_notebooks"):
                notebook, reason = notebook.split(":")
                blacklisted_notebooks[notebook] = reason

        if collector.nodeid in blacklisted_notebooks:
            collector.add_marker(
                pytest.mark.skip(f"The notebook is blacklisted: {blacklisted_notebooks[collector.nodeid]}")
            )
        else:
            if collector.nodeid in collector.config.getini("serial_notebooks"):
                collector.add_marker("serial")
            else:
                # Automatically mark every notebook which uses p_map (multiprocessing) as serial to avoid multiprocessing issues (notebooks are executed at the end of the tests)
                with collector.path.open() as f:
                    notebook = f.read()
                if "p_map" in notebook:
                    collector.add_marker("serial")


class DoctestOutputChecker(doctest.OutputChecker):
    def check_output(self, want: str, got: str, optionflags: int):
        if optionflags & doctest.ELLIPSIS:
            # Allow [...] to be used as additional marker to skip complete outputs
            want = re.sub(r"\[\.\.\.\]\s*", "...", want)

        return super().check_output(want, got, optionflags)


def pytest_configure(config):
    doctest.OutputChecker = DoctestOutputChecker
