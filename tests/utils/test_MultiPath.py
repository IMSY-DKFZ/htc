# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path

import pytest

from htc.utils.MultiPath import MultiPath


class TestMultiPath:
    def test_existing(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()

        tmp_file = dir2 / "file.txt"
        tmp_file.write_text("Test")

        assert not (dir1 / "file.txt").exists()
        assert tmp_file.exists()

        path = MultiPath(dir1)
        path.add_alternative(dir2)
        assert path.possible_locations() == [dir1, dir2]
        assert path.possible_locations(only_existing=True) == [dir1, dir2]
        assert path.possible_locations(filter_func=lambda p: p.name == "dir2") == [dir2]

        path_file = path / "file.txt"
        assert path_file.exists()
        assert path_file.possible_locations() == [dir1 / "file.txt", tmp_file]
        assert path_file.name == "file.txt"
        assert path_file.stem == "file"
        assert path_file.suffix == ".txt"

        with path_file.open() as f:
            assert f.read() == "Test"

        subpath = path / "a"
        subpath.mkdir(exist_ok=True, parents=True)

        assert (dir1 / "a").exists() and (dir1 / "a").is_dir()
        assert not (dir2 / "a").exists()

        # If one of the paths exists, we want to make the dir in the root location
        subpath2 = path / "aa"
        assert not (dir1 / "aa").exists() and not (dir2 / "aa").exists()
        (dir2 / "aa").mkdir()
        assert (dir2 / "aa").exists()
        subpath2.mkdir(exist_ok=True, parents=True)
        assert (dir1 / "aa").exists()

    def test_nonexisting(self) -> None:
        path = MultiPath("/a/b/c")
        path.add_alternative("/a/b/cc")
        path /= "xx/a"

        assert not path.exists()
        assert path.possible_locations() == [Path("/a/b/c/xx/a"), Path("/a/b/cc/xx/a")]
        assert path.possible_locations(only_existing=True) == []

        with pytest.raises(FileNotFoundError, match=r"None of the locations.*matches the needle"):
            path.find_location("wrong_needle")

    def test_multiple(self, tmp_path: Path) -> None:
        dirs = []
        path = MultiPath("/a/b")
        for i in range(10):
            tmp_dir = tmp_path / f"dir{i}"
            tmp_dir.mkdir()
            dirs.append(tmp_dir)
            path.add_alternative(tmp_dir)

        tmp_file = dirs[5] / "test.txt"
        tmp_file.write_text("Test")

        target_file = path / "test.txt"
        assert target_file.exists()

    def test_path_resolving(self) -> None:
        path = MultiPath("~/xx")
        path.add_alternative("//home")
        path.add_alternative("./a")
        path.add_alternative("a/b")

        assert path.possible_locations() == [
            Path("~/xx").expanduser(),
            Path("/home"),
            Path.cwd() / "a",
            Path.cwd() / "a/b",
        ]

    def test_default_location(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        dir3 = tmp_path / "dir3"
        dir3.mkdir()

        path = MultiPath(dir1)
        path.add_alternative(dir2)
        path.add_alternative(dir3)

        # The first path is per default the default location
        file_path = path / "file1.txt"
        file_path.write_text("test")
        assert str(file_path) == str(dir1 / "file1.txt")
        assert Path(file_path) == dir1 / "file1.txt"
        assert (dir1 / "file1.txt").exists()
        assert not (dir2 / "file1.txt").exists() and not (dir3 / "file1.txt").exists()

        path.set_default_location("dir1")

        file_path = path / "file2.txt"
        file_path.write_text("test")
        assert str(file_path) == str(dir1 / "file2.txt")
        assert Path(file_path) == dir1 / "file2.txt"
        assert (dir1 / "file2.txt").exists()
        assert not (dir2 / "file2.txt").exists() and not (dir3 / "file2.txt").exists()

        path.set_default_location("dir2")

        file_path = path / "file3.txt"
        file_path.write_text("test")
        assert str(file_path) == str(dir2 / "file3.txt")
        assert Path(file_path) == dir2 / "file3.txt"
        assert (dir2 / "file3.txt").exists()
        assert not (dir1 / "file3.txt").exists() and not (dir3 / "file3.txt").exists()

        # dir2 is the default location, so both subdirs should exist after the following lines
        (dir1 / "subdir").mkdir(parents=True, exist_ok=True)
        (path / "subdir").mkdir(parents=True, exist_ok=True)
        assert (dir1 / "subdir").exists()
        assert (dir2 / "subdir").exists()

        # Full path always works
        path.set_default_location(str(dir3))

        file_path = path / "file4.txt"
        file_path.write_text("test")
        assert str(file_path) == str(dir3 / "file4.txt")
        assert Path(file_path) == dir3 / "file4.txt"
        assert (dir3 / "file4.txt").exists()
        assert not (dir1 / "file4.txt").exists() and not (dir2 / "file4.txt").exists()

        # Filename is ignored
        dir_path = path / "dir2.txt"
        dir_path.set_default_location("dir2")
        assert dir_path.find_best_location() == dir2 / "dir2.txt"

        # A filename does not affect the best location because it is identical for all paths
        file_path.set_default_location("file4.txt")
        assert file_path.find_best_location() == dir3 / "file4.txt"

    def test_iteration(self, tmp_path: Path) -> None:
        all_file_paths = []

        intermediates1 = tmp_path / "intermediates1"
        intermediates1.mkdir()
        target_dir = intermediates1 / "preprocessing"
        target_dir.mkdir()

        for i in range(10):
            file_path = target_dir / f"{i}"
            file_path.write_text("test")
            all_file_paths.append(file_path)

        intermediates2 = tmp_path / "intermediates2"
        intermediates2.mkdir()
        target_dir = intermediates2 / "preprocessing"
        target_dir.mkdir()

        for i in range(10, 15):
            file_path = target_dir / f"{i}"
            file_path.write_text("test")
            all_file_paths.append(file_path)

        # One without the preprocessing subfolder
        intermediates3 = tmp_path / "intermediates3"
        intermediates3.mkdir()

        path = MultiPath(intermediates1)
        path.add_alternative(intermediates2)
        path.add_alternative(intermediates3)

        for p in (path / "preprocessing").iterdir():
            assert p in all_file_paths
            assert int(p.name) in range(15)
        for p in (path / "preprocessing").iterdir():
            assert p in all_file_paths
            assert int(p.name) in range(15)
        for p in (path / "preprocessing").rglob("*"):
            assert p in all_file_paths
            assert int(p.name) in range(15)

    def test_pickle(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()

        path = MultiPath(dir1)
        path.add_alternative(dir2)
        subpath = path / "a"

        saved_object = pickle.dumps(subpath)
        subpath_loaded = pickle.loads(saved_object)

        assert subpath == subpath_loaded
        assert type(subpath) == type(subpath_loaded) == MultiPath
        assert subpath.possible_locations() == subpath_loaded.possible_locations()

    def test_parts(self, tmp_path: Path) -> None:
        path = MultiPath(tmp_path, "a", "b")
        assert path.find_best_location() == Path(tmp_path, "a", "b")

    def test_parent(self) -> None:
        path_parent = MultiPath("/a/b/c")
        path_parent.add_alternative("/xx")
        path_child = path_parent / "child"

        assert path_child.parents[0] == path_child.parent == path_parent
        assert path_child.parents[1] == path_child.parent.parent == Path("/a/b")
        assert isinstance(path_child.parents[1], Path)

        path_child.set_default_location("xx")
        assert path_child.parents[0].find_best_location() == path_child.parent.find_best_location() == Path("/xx")
        assert path_child.parents[1] == path_child.parent.parent == Path("/")

        with pytest.raises(IndexError):
            path_child.parents[2]

        with pytest.raises(IndexError):
            path_child.parents[3]

    def test_multiple_matches(self, tmp_path: Path) -> None:
        path = MultiPath(tmp_path / "results")
        path.add_alternative(tmp_path / "results_camera")
        path.add_alternative(tmp_path / "results_semantic")
        path.set_default_location(tmp_path / "results")
        assert path.find_best_location() == tmp_path / "results"

        subpath = path / "x"
        assert subpath.find_best_location() == tmp_path / "results" / "x"

        (tmp_path / "results_camera" / "training" / "image").mkdir(parents=True, exist_ok=True)
        training_dir = path / "training"
        assert (training_dir / "image").possible_locations() == [
            tmp_path / "results" / "training" / "image",
            tmp_path / "results_camera" / "training" / "image",
            tmp_path / "results_semantic" / "training" / "image",
        ]

    def test_multiple_matches_root(self, tmp_path: Path) -> None:
        path = MultiPath(tmp_path / "results")
        path.add_alternative(tmp_path / "results_camera")
        path.add_alternative(tmp_path / "results_semantic")
        path.set_default_location(tmp_path / "results")

        (tmp_path / "results" / "training" / "image").mkdir(parents=True, exist_ok=True)
        training_dir = path / "training"
        assert (training_dir / "image").possible_locations() == [
            tmp_path / "results" / "training" / "image",
            tmp_path / "results_camera" / "training" / "image",
            tmp_path / "results_semantic" / "training" / "image",
        ]

    def test_writing(self, tmp_path: Path) -> None:
        location1 = tmp_path / "location1"
        location2 = tmp_path / "location2"
        location2.mkdir(parents=True, exist_ok=True)

        existing_file = location2 / "file.txt"
        existing_file.write_text("test")

        path = MultiPath(location1)
        path.add_alternative(location2)
        path.set_default_location("location1")

        assert str(path / "file.txt") == str(existing_file), (
            "Needle is not used since the file does not exist on the needle location"
        )
        assert Path(path / "file.txt") == existing_file

        write_dir = path / "write_dir"
        write_dir.mkdir(parents=True, exist_ok=True)
        assert str(write_dir) == str(location1 / "write_dir"), "Needle is used"
        assert Path(write_dir) == location1 / "write_dir"
        assert str(path / "file.txt") == str(existing_file)
        assert Path(path / "file.txt") == existing_file

        write_dir2 = (path / "write_dir2").find_best_location(writing=True)
        write_dir2.mkdir(parents=True, exist_ok=True)
        assert str(write_dir2) == str(location1 / "write_dir2"), "Needle is used"
        assert Path(write_dir2) == location1 / "write_dir2"
        assert (location1 / "write_dir2").exists() and (location1 / "write_dir2").is_dir()

    def test_resolve(self, tmp_path: Path) -> None:
        location1 = tmp_path / "location1"
        location2 = tmp_path / "../location2"

        path = MultiPath(location1)
        path.add_alternative(location2)

        locations = path.possible_locations()
        assert locations[0] == location1.resolve()
        assert locations[1] == location2.resolve()

    def test_absolute_child(self, tmp_path: Path) -> None:
        path = MultiPath(tmp_path)

        new_path = path / Path("/a")
        assert new_path == Path("/a")

        new_path = path / Path("/a") / "/b"
        assert new_path == Path("/b")

        path2 = MultiPath(tmp_path / "2")
        new_path = path / path2
        assert new_path == path2

    def test_relative_to(self, tmp_path: Path) -> None:
        path = MultiPath(tmp_path / "a" / "1")
        path2 = MultiPath(tmp_path / "b")
        path2.add_alternative(tmp_path / "a")
        path3 = MultiPath(tmp_path / "c")

        assert path.relative_to(path2) == Path("1")
        assert path.relative_to(tmp_path / "a") == Path("1")

        with pytest.raises(ValueError):
            path.relative_to(path3)
            path.relative_to(tmp_path / "c")

    def test_with_name(self, tmp_path: Path) -> None:
        path = MultiPath(tmp_path / "a")
        path.add_alternative(tmp_path / "b")
        path_folder = path / "run*"

        existing = tmp_path / "b" / "run1"
        existing.mkdir(parents=True, exist_ok=True)
        path_name = path_folder.with_name("run1")
        path_stem = path_folder.with_stem("run1")

        assert existing.exists() and path_name.exists() and path_stem.exists()
        assert str(existing) == str(path_name) == str(path_stem)

        path_file = path / "run*.txt"
        existing = tmp_path / "b" / "run1.txt"
        existing.write_text("test")

        path_name = path_file.with_name("run1.txt")
        path_stem = path_file.with_stem("run1")

        assert existing.exists() and path_name.exists() and path_stem.exists()
        assert str(existing) == str(path_name) == str(path_stem)

        path_suffix = path_file.with_suffix(".bin")
        assert not path_suffix.exists()
        assert str(path_suffix) == str(tmp_path / "a" / "run*.bin")
