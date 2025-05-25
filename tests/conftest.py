# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os
import random
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from appdirs import user_cache_dir
from pytest import MonkeyPatch
from xdist import is_xdist_controller
from xdist.scheduler import LoadScopeScheduling

# from htc.dataset_preparation.DatasetGenerator import DatasetGenerator
# from htc.dataset_preparation.run_dataset_semantic import DatasetGeneratorSemantic
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.parallel import p_map

# Run all tests marked as serial in sequential order, other tests in load scope
# https://github.com/pytest-dev/pytest-xdist/issues/385#issuecomment-962288342


def pytest_configure(config):
    config.pluginmanager.register(XDistSerialPlugin())


class XDistSerialPlugin:
    def __init__(self):
        self._nodes = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection(self, session):
        if is_xdist_controller(session):
            self._nodes = {item.nodeid: item for item in session.perform_collect(None)}
            return True

    def pytest_xdist_make_scheduler(self, config, log):
        return SerialScheduling(config, log, nodes=self._nodes)


class SerialScheduling(LoadScopeScheduling):
    def __init__(self, config, log, *, nodes):
        super().__init__(config, log)
        self._nodes = nodes

    def _split_scope(self, nodeid):
        node = self._nodes[nodeid]
        if node.get_closest_marker("serial"):
            # put all `@pytest.mark.serial` tests in same scope, to
            # ensure they're all run in the same worker
            marker = "__serial__"
        else:
            # otherwise, same scope as in load scope
            marker = super()._split_scope(nodeid)

        return marker


@pytest.fixture()
def make_tmp_example_data(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[Callable]:
    """Creates a new temporary dataset which is a shortened version of the original one (less images). The intermediates directory is empty (if not specified otherwise) and it can be used to check whether intermediate files are produced correctly."""

    def _make_tmp_example_data(
        n_images: int = 4,
        paths: list[DataPath] = None,
        include_intermediates: bool = False,
        data_dir: Path = settings.data_dirs.semantic,
    ) -> Path:
        # Same directory structure in the temporary directory
        tmp_dataset_dir = tmp_path / f"{data_dir.parent.name}"
        tmp_dataset_dir.mkdir(parents=True, exist_ok=True)

        if include_intermediates:
            intermediates_dir = settings.datasets.find_intermediates_dir(data_dir)
            assert intermediates_dir.is_dir(), f"Intermediates directory {intermediates_dir} does not exist"
            os.symlink(intermediates_dir, tmp_dataset_dir / "intermediates")
        else:
            (tmp_dataset_dir / "intermediates").mkdir(parents=True, exist_ok=True)

        if paths is None:
            # Find some example images
            paths = list(DataPath.iterate(data_dir))

            random.seed(2)
            random.shuffle(paths)
            paths = paths[:n_images]
        else:
            n_images = len(paths)

        tmp_data = tmp_dataset_dir / "data"
        tmp_data.mkdir(parents=True, exist_ok=True)

        # Make the example images in the temporary directory available
        for p in paths:
            assert "overlap" not in str(p), "Overlap file detected, best to use a different seed"
            tmp_subject_dir = (tmp_data / p().relative_to(data_dir)).parent
            tmp_subject_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(p(), tmp_subject_dir / p.timestamp)

        shutil.copy2(data_dir / "dataset_settings.json", tmp_data / "dataset_settings.json")

        if (data_dir / "overlap").exists():
            (tmp_data / "overlap").mkdir(parents=True, exist_ok=True)

        if not include_intermediates:
            # The meta table is required for DataPath.from_image_name()
            paths = list(DataPath.iterate(tmp_data))
            if (context_dir := tmp_data / "context_experiments").exists():
                shutil.copy2(data_dir / "dataset_settings.json", context_dir / "dataset_settings.json")
                paths += list(DataPath.iterate(context_dir))

            if all(
                p.dataset_settings is not None
                and "2021_02_05_Tivita_multiorgan_semantic" in p.dataset_settings.get("dataset_name", "")
                for p in paths
            ):
                GeneratorClass = DatasetGeneratorSemantic
            else:
                GeneratorClass = DatasetGenerator

            gen = GeneratorClass(output_path=tmp_dataset_dir)
            assert paths == gen.paths

            # Segmentations also affect the meta table
            if hasattr(gen, "segmentations"):
                # Multiprocessing may not work well in the testing environment
                p_map(gen.segmentations, gen.paths, num_cpus=2, task_name="Segmentation files", use_threads=True)
            gen.meta_table()

            assert len(paths) == n_images

        # Make sure the script operates on the temporary instead of the real data
        env_name = settings.datasets.path_to_env(data_dir)
        assert env_name is not None
        monkeypatch.setenv(env_name, str(tmp_dataset_dir))
        monkeypatch.setattr(settings, "_intermediates_dir_all", tmp_dataset_dir / "intermediates")

        # Reset global caches for this test
        monkeypatch.setattr(settings, "_datasets", None)
        monkeypatch.setattr(DataPath, "_local_meta_cache", None)
        monkeypatch.setattr(DataPath, "_network_meta_cache", None)
        monkeypatch.setattr(DataPath, "_meta_labels_cache", {})
        monkeypatch.setattr(DataPath, "_data_paths_cache", {})

        return tmp_dataset_dir

    yield _make_tmp_example_data


@pytest.fixture(scope="session")
def path_test_files() -> Iterator[Path]:
    # Folder on the network drive with files which are needed during testing
    network_path = settings.datasets.network_project / "test_files"

    # Sync the test files to the local cache directory
    cache_dir = Path(user_cache_dir("htc", "IMSY")) / "test_files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(["rsync", "-a", "--delete", f"{network_path}/", f"{cache_dir}/"])
    assert res.returncode == 0, "Could not copy test files from network drive"

    yield cache_dir


def check_data_accessible(data_dir: Path) -> Callable:
    if data_dir.exists():
        try:
            # Similar to the logic in the settings file
            next(data_dir.iterdir())
            is_accessible = True
        except PermissionError:
            is_accessible = False
    else:
        is_accessible = False

    # We return a callable so that we can make this check inside a test function
    def _check_data_accessible() -> None:
        if not is_accessible:
            pytest.skip(f"No permissions to access the data directory {data_dir}")

    return _check_data_accessible


@pytest.fixture(scope="session")
def check_human_data_accessible() -> Iterator[Callable]:
    yield check_data_accessible(settings.data_dirs.human)


@pytest.fixture(scope="session")
def check_sepsis_data_accessible() -> Iterator[Callable]:
    yield check_data_accessible(settings.data_dirs.sepsis)


@pytest.fixture(scope="session")
def check_sepsis_ICU_data_accessible() -> Iterator[Callable]:
    yield check_data_accessible(settings.data_dirs.sepsis_ICU)
