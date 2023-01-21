# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Union


def unify_path(path: Union[str, Path]) -> Path:
    """
    Tries to bring some consistency to paths:
        - Resolve home directories (~ → /home/username).
        - Make paths absolute.

    Note: this requires access to the filesystem and when the path points to a high-latency location, then this may introduce a lag.

    Args:
        path: The original path.

    Returns:
        The unified path.
    """
    if isinstance(path, str):
        path = Path(path)

    if str(path).startswith("~"):
        # Unfortunately, the resolve() function cannot handle paths starting with ~. The workaround is to expand ~ to the home path in this case
        path = path.expanduser()

    # Normalize the path (this makes it also absolute)
    return path.resolve()
