# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import MonkeyPatch

from htc.cli import main
from htc.settings import settings


def test_cli(tmp_path: Path, monkeypatch: MonkeyPatch, capsys) -> None:
    example_paths = [
        "run_root_example_script.py",
        "module1/run_example_script.py",
        "module2/run_example_script.py",
    ]
    (tmp_path / "module1").mkdir(exist_ok=True, parents=True)
    (tmp_path / "module2").mkdir(exist_ok=True, parents=True)
    for p in example_paths:
        (tmp_path / p).write_text("")

    monkeypatch.setattr(settings, "src_dir", tmp_path)
    assert len(sorted(settings.src_dir.rglob("run_*.py"))) == 3

    monkeypatch.setattr("sys.argv", ["cli.py"])
    assert main() == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "direct entry point" in out and "root_example_script" in out and "example_script" in out

    monkeypatch.setattr("sys.argv", ["cli.py", "non_existing_script"])
    assert main() == 1
    out, err = capsys.readouterr()
    assert err == ""
    assert "Could not find" in out

    monkeypatch.setattr("sys.argv", ["cli.py", "module1.example_script"])
    assert main() == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""

    monkeypatch.setattr("sys.argv", ["cli.py", "example_script"])
    assert main() == 1
    out, err = capsys.readouterr()
    assert err == ""
    assert (
        "Found more than one candidate" in out and "module1.example_script" in out and "module2.example_script" in out
    )
