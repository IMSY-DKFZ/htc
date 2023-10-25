# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path
from typing import Any, Union

import blosc
import numpy as np


def compress_file(path: Path, data: Union[np.ndarray, dict[Any, np.ndarray]]) -> None:
    """
    Compresses the numpy array(s) using blosc (https://github.com/Blosc/c-blosc).

    blosc can be really fast to decompress (compression may be slow though, e.g. 2 h for 10000 images) and produces surprisingly small files (e.g. 35.52 MiB instead of 60 MiB for one HSI cube). The full benchmark is in the DataLoadingPerformance.ipynb notebook.

    Args:
        path: The path where the compressed file should be stored.
        data: The numpy array to store or a dict of numpy arrays.
    """
    if type(data) == dict:
        compressed_data = []
        meta = {}
        for key, array in data.items():
            array = np.ascontiguousarray(array)
            array_compressed = blosc.compress_ptr(
                array.__array_interface__["data"][0],
                array.size,
                array.dtype.itemsize,
                clevel=9,
                cname="zstd",
                shuffle=blosc.SHUFFLE,
            )

            compressed_data.append(array_compressed)
            meta[key] = (array.shape, array.dtype, len(array_compressed))

        with path.open("wb") as f:
            pickle.dump(meta, f)
            for c in compressed_data:
                f.write(c)
    else:
        array = data
        # Based on https://stackoverflow.com/a/56761075
        array = np.ascontiguousarray(array)  # Does nothing if already contiguous (https://stackoverflow.com/a/51457275)

        # A bit ugly, but very fast (http://python-blosc.blosc.org/tutorial.html#compressing-from-a-data-pointer)
        compressed_data = blosc.compress_ptr(
            array.__array_interface__["data"][0],
            array.size,
            array.dtype.itemsize,
            clevel=9,
            cname="zstd",
            shuffle=blosc.SHUFFLE,
        )

        with path.open("wb") as f:
            pickle.dump((array.shape, array.dtype), f)
            f.write(compressed_data)


def decompress_file(
    path: Path, start_pointer: Union[int, dict[str, int]] = None
) -> Union[Union[np.ndarray, int], dict[Any, Union[np.ndarray, int]]]:
    """
    Decompresses a blosc file.

    Args:
        path: File to the blosc data.
        start_pointer: If not None must be a valid memory address. It will be used to store the decompressed data directly into the provided memory location. This is, for example, useful if the data should be directly loaded into a shared memory buffer. If the compressed data contains a dictionary, the pointers must also be a dictionary with the keys corresponding to the (expected) keys in the compressed data. A pointer can only be used if the size and dtype of the decompressed data is known in advance.

    Returns: Decompressed array data or the given pointer address. Depending on the file, this will either be directly the numpy array or a dict with all numpy arrays.
    """
    res = {}

    with path.open("rb") as f:
        meta = pickle.load(f)
        if type(meta) == tuple:
            shape, dtype = meta
            data = f.read()

            if start_pointer is not None:
                blosc.decompress_ptr(data, start_pointer)
                res = start_pointer
            else:
                array = np.empty(shape=shape, dtype=dtype)
                blosc.decompress_ptr(data, array.__array_interface__["data"][0])
                res = array
        else:
            for name, (shape, dtype, size) in meta.items():
                data = f.read(size)

                if start_pointer is not None:
                    blosc.decompress_ptr(data, start_pointer[name])
                    res[name] = start_pointer[name]
                else:
                    array = np.empty(shape=shape, dtype=dtype)
                    blosc.decompress_ptr(data, array.__array_interface__["data"][0])
                    res[name] = array

    return res
