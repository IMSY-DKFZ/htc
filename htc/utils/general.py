# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def apply_recursive(func: Callable, obj: dict | list | Any) -> Any:
    """
    Apply a callback to every element of a potentially nested structure.

    >>> import torch
    >>> data = [{"my_data": torch.tensor(1)}]
    >>> apply_recursive(lambda x: x.numpy(), data)
    [{'my_data': array(1)}]

    Args:
        func: The callback to apply to every element (non-lists or non-dicts).
        obj: The nested data structure.

    Returns: The same data structure but with the callback applied to each element.
    """
    if type(obj) == dict:
        for key, value in obj.items():
            obj[key] = apply_recursive(func, value)

        return obj
    elif type(obj) == list or type(obj) == np.ndarray:
        for i in range(len(obj)):
            obj[i] = apply_recursive(func, obj[i])

        return obj
    else:
        return func(obj)


def merge_dicts_deep(dict1: dict, dict2: dict) -> Iterator[tuple[str, dict]]:
    """
    Merges two dictionaries on every level with matching keys (dict1 | dict2 only merges on the outer level).

    In case of conflicts, the second dictionary has precedence over the first.

    >>> dict1 = {1: {"a": "A"}, 2: {"b": "B", "conflict": "with_b"}}
    >>> dict2 = {2: {"c": "C", "conflict": "with_c"}, 3: {"d": "D"}}
    >>> dict(merge_dicts_deep(dict1, dict2))
    {1: {'a': 'A'}, 2: {'b': 'B', 'c': 'C', 'conflict': 'with_c'}, 3: {'d': 'D'}}

    Args:
        dict1: First dictionary object.
        dict2: Second dictionary object.

    Yields: Combined dictionary with inner values merged. All dictionary keys are sorted.
    """
    # Based on: https://stackoverflow.com/a/7205672
    for k in sorted(set(dict1.keys()) | set(dict2.keys())):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield k, dict(merge_dicts_deep(dict1[k], dict2[k]))
            else:
                # Only values left or only one of the values is a dict
                yield k, dict2[k]
        elif k in dict1:
            yield k, dict1[k]
        else:
            yield k, dict2[k]


def sha256_file(path: Path) -> str:
    """
    Calculate the SHA256 hash of a file. To reduce memory consumption, the file is not read at once but rather sequentially.

    Args:
        path: Path to the file.

    Returns: Hex string representation of the file.
    """
    # From: https://stackoverflow.com/a/44873382
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with path.open("rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])

    return h.hexdigest()


def safe_copy(src: Path, dst: Path, **kwargs) -> None:
    """
    Copies files or directories via shutil.copy2 and raises an error if the copy is different to the source file. For this, the hash of the source file(s) is compared with the hash of the destination file(s).

    Note: This function was introduced because we got reports of image copies with random errors after copying and we want to check whether we can reproduce the problem.

    Args:
        src: Path to the source file or directory.
        dst: Path to the destination file or directory.
        **kwargs: Keyword arguments passed to shutil.copy2.

    Raises:
        ValueError: In case the copy is different to the original file.
    """
    if src.is_dir():
        src_paths = sorted(src.iterdir())
        if len(src_paths) == 0:
            dst.mkdir(parents=True, exist_ok=True)
        for path in src_paths:
            dst_path = dst / path.name
            safe_copy(path, dst_path)
    else:
        dst.parent.mkdir(exist_ok=True, parents=True)
        hash_src = sha256_file(src)
        shutil.copy2(src, dst, **kwargs)
        hash_dst = sha256_file(dst)

        if hash_src != hash_dst:
            raise ValueError(
                f"Could not safely copy the file {src} to {dst}. The sha256 of the source file ({hash_src}) is"
                f" different to the destination file {hash_dst}"
            )


def subprocess_run(command, **kwargs) -> subprocess.Popen:
    """
    Similar to subprocess.run() but explicitly kills the subprocess if a CTRL + C is sent.

    Additionally, it is ensured that the process's returncode is 0.

    Args:
        command: Command to run in a subprocess.
        **kwargs: Keyword arguments passed to Popen.

    Returns: Reference to the process. Can be used to check the returncode (`subprocess_run().returncode`).
    """
    process = subprocess.Popen(command, **kwargs)
    try:
        process.wait()
    except KeyboardInterrupt:
        # Explicitly kill the subprocess (https://stackoverflow.com/a/58536371)

        # We use SIGTERM instead of SIGKILL so that the process has a chance to shut down (this has e.g. the effect that pytorch lightning still prints the GPU memory usage)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()

    return process


def clear_directory(path: Path) -> None:
    """
    Delete the content of the directory without deleting the directory itself (e.g. to keep symlinks working).

    >>> import tempfile
    >>> tmp_dir_handle = tempfile.TemporaryDirectory()
    >>> tmp_dir = Path(tmp_dir_handle.name)

    Let's create some test files
    >>> n_written = (tmp_dir / "file.txt").write_text("test")
    >>> tmp_subdir = tmp_dir / "subdir"
    >>> tmp_subdir.mkdir(parents=True, exist_ok=True)
    >>> len(sorted(tmp_dir.rglob("*")))
    2
    >>> clear_directory(tmp_dir)

    The directory is still there but empty
    >>> tmp_dir.exists()
    True
    >>> len(sorted(tmp_dir.rglob("*")))
    0
    >>> tmp_dir_handle.cleanup()

    Args:
        path: Path to the directory to clear.
    """
    if path.is_dir():
        for f in path.iterdir():
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
