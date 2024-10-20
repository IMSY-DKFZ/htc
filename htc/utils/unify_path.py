# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
from pathlib import Path


def unify_path(path: str | Path, resolve_symlinks: bool = True) -> Path:
    """
    Tries to bring some consistency to paths:
        - Resolve home directories (~ → /home/username).
        - Make paths absolute.
        - Resolves symbolic links (optional).

    Note: this requires access to the filesystem and when the path points to a high-latency location, then this may introduce a lag.

    Args:
        path: The original path.
        resolve_symlinks: If true, also resolve symbolic links.

    Returns: The unified path.
    """
    if isinstance(path, str):
        path = Path(path)

    # Unfortunately, the resolve() function cannot handle paths starting with ~. The workaround is to expand ~ to the home path in this case
    path = path.expanduser()

    if resolve_symlinks:
        # Normalize, make absolute and resolve symlinks
        return path.resolve()
    else:
        path = str(path)
        if path.startswith("//"):
            # This is not done by abspath
            path = path.replace("//", "/")

        # Normalize and make absolute but don't resolve symlinks
        path = os.path.abspath(path)  # noqa: PTH100
        return Path(path)
