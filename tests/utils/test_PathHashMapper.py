# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest

from htc.tivita.DataPath import DataPath
from htc.utils.PathHashMapper import PathHashMapper


class TestPathHashMapper:
    def test_basics(self) -> None:
        paths = [
            DataPath.from_image_name("P058#2020_05_13_18_09_26"),
            DataPath.from_image_name("P044#2020_02_01_09_51_15"),
        ]
        mapper = PathHashMapper(paths)
        h0 = mapper.path_to_hash(paths[0])
        h1 = mapper.path_to_hash(paths[1])

        assert mapper.hash_to_path(h0) == paths[0]
        assert mapper.hash_to_path(h1) == paths[1]

        with pytest.raises(AssertionError, match="not a valid sha256 hash"):
            mapper.hash_to_path("700")

        with pytest.raises(AssertionError, match="No path mapping for the hash"):
            mapper.hash_to_path("700dbf8b4aed1b855389615876dbf532a0408f1d5344b1437559f4baa7534184")
