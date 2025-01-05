# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
import sys
from pathlib import Path


def check_public_readme(file: Path) -> None:
    """
    Check internal links in the public README file. All links must be relative so that they work correctly on GitHub.

    Args:
        file: Path to the README file.
    """
    text = file.read_text()
    for match in re.findall(r"\]\(([^)]+)\)", text):
        assert match.startswith(("https", "#", "./")), (
            f"Found an invalid internal link to the file {match} in {file}. All links must be relative and start with ./"
        )


if __name__ == "__main__":
    returncode = 0

    for file in sys.argv[1:]:
        try:
            check_public_readme(Path(file))
        except Exception as e:
            print(e)
            returncode = 1

    sys.exit(returncode)
