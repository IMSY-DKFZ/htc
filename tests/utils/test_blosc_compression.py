# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import torch

from htc.utils.blosc_compression import compress_file, decompress_file


class TestBloscCompression:
    def test_array(self, tmp_path: Path) -> None:
        array = np.random.rand(480, 640).astype(np.float16)
        compress_file(tmp_path / "array.blosc", array)
        array_decompressed, meta = decompress_file(tmp_path / "array.blosc", return_meta=True)
        assert meta == (array.shape, array.dtype)

        assert np.all(array == array_decompressed)

        tensor = torch.empty(480, 640, dtype=torch.float16)
        pointer = decompress_file(tmp_path / "array.blosc", start_pointer=tensor.data_ptr())
        assert pointer == tensor.data_ptr()
        assert np.all(array == tensor.numpy())

    def test_dict(self, tmp_path: Path) -> None:
        data = {i: np.random.randint(0, 10, (480, 640)) for i in range(4)}
        compress_file(tmp_path / "array.blosc", data)
        data_decompressed = decompress_file(tmp_path / "array.blosc")

        data_tensor = {i: torch.empty(480, 640, dtype=torch.int64) for i in range(4)}
        pointers = decompress_file(
            tmp_path / "array.blosc", start_pointer={k: v.data_ptr() for k, v in data_tensor.items()}
        )
        assert pointers.keys() == data.keys()
        assert list(pointers.values()) == [t.data_ptr() for t in data_tensor.values()]

        assert data.keys() == data_decompressed.keys()
        assert data.keys() == data_tensor.keys()
        for k in data.keys():
            assert np.all(data[k] == data_decompressed[k])
            assert np.all(data[k] == data_tensor[k].numpy())

        data_decompressed2 = decompress_file(tmp_path / "array.blosc", load_keys=[0, 2])
        assert data_decompressed2.keys() == {0, 2}
        assert np.all(data[0] == data_decompressed2[0])
        assert np.all(data[2] == data_decompressed2[2])
