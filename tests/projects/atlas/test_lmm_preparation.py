# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import MonkeyPatch
from pytest_console_scripts import ScriptRunner

import htc_projects.atlas.run_lmm_preparation as run_lmm_preparation


def test_lmm_preparation(tmp_path: Path, monkeypatch: MonkeyPatch, script_runner: ScriptRunner) -> None:
    monkeypatch.setenv("PATH_HTC_RESULTS", str(tmp_path))
    res = script_runner.run(run_lmm_preparation.__file__)
    assert res.success

    feather_path = tmp_path / "lmm" / "mergedData_normalized.feather"
    assert feather_path.exists()
    assert feather_path.stat().st_size > 0
