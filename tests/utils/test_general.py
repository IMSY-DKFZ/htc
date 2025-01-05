# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib
from pathlib import Path

from htc.utils.general import safe_copy, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    content = "test string"
    hash_str = hashlib.sha256(content.encode()).hexdigest()

    tmp_file = tmp_path / "file.txt"
    tmp_file.write_text(content)

    assert sha256_file(tmp_file) == hash_str == "d5579c46dfcc7f18207013e65b44e4cb4e2c2298f4ac457ba8f82743f31e930b"


def test_save_copy(tmp_path: Path) -> None:
    content = "test string 1"
    hash_str1 = hashlib.sha256(content.encode()).hexdigest()
    tmp_file1 = tmp_path / "file1.txt"
    tmp_file1.write_text(content)

    content = "test string 2"
    hash_str2 = hashlib.sha256(content.encode()).hexdigest()
    tmp_dir2 = tmp_path / "subfolder" / "subsubfolder"
    tmp_dir2.mkdir(parents=True)
    tmp_file2 = tmp_dir2 / "file2.txt"
    tmp_file2.write_text(content)

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir(parents=True)

    # test save copy for single file
    tmp_dir3 = tmp_path / "subfolder2"
    tmp_dir3.mkdir(parents=True)
    tmp_file3 = tmp_dir3 / "file3.txt"
    safe_copy(tmp_file1, tmp_file3)
    assert sha256_file(tmp_file1) == sha256_file(tmp_file3)

    # test save copy for entire folder
    new_dir = tmp_path / "new_dir"
    safe_copy(tmp_path, new_dir)
    assert sha256_file(tmp_file1) == sha256_file(new_dir / "file1.txt") == hash_str1
    assert sha256_file(tmp_file2) == sha256_file(new_dir / "subfolder" / "subsubfolder" / "file2.txt") == hash_str2
    assert sha256_file(tmp_file3) == sha256_file(new_dir / "subfolder2" / "file3.txt")
    empty_dir_copied = new_dir / "empty_dir"
    assert empty_dir_copied.exists()
