# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from htc.settings import settings
from htc.special.run_python_versions import PythonVersionUpgrader
from htc.utils.general import safe_copy, sha256_file


class TestPythonVersionUpgrader:
    @pytest.fixture(scope="function")
    def tmp_scripts(self, tmp_path: Path) -> Path:
        upgrader = PythonVersionUpgrader()
        for file in upgrader.files:
            file_src = settings.src_dir / file
            file_tmp = tmp_path / file
            file_tmp.parent.mkdir(parents=True, exist_ok=True)

            safe_copy(file_src, file_tmp)

        return tmp_path

    def test_same(self, monkeypatch: pytest.MonkeyPatch, tmp_scripts: Path) -> None:
        monkeypatch.setattr(settings, "src_dir", tmp_scripts)

        # Per default, the script should not change anything
        upgrader = PythonVersionUpgrader()

        hashes_old = [sha256_file(tmp_scripts / f) for f in upgrader.files]
        upgrader.upgrade_all()
        hashes_new = [sha256_file(tmp_scripts / f) for f in upgrader.files]

        assert hashes_old == hashes_new

    def test_upgrade(self, monkeypatch: pytest.MonkeyPatch, tmp_scripts: Path) -> None:
        monkeypatch.setattr(settings, "src_dir", tmp_scripts)

        # Every file should change
        upgrader = PythonVersionUpgrader()
        upgrader.python_versions = ("X", "Y", "Z")

        hashes_old = [sha256_file(tmp_scripts / f) for f in upgrader.files]
        upgrader.upgrade_all()
        hashes_new = [sha256_file(tmp_scripts / f) for f in upgrader.files]

        assert all(old != new for old, new in zip(hashes_old, hashes_new, strict=True))
