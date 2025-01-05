# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import sys
from collections.abc import Callable
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Union

from htc.utils.unify_path import unify_path


class MultiPathSequence:
    def __init__(self, path: "MultiPath"):
        self.path = path

    def __getitem__(self, idx: int) -> Union["MultiPath", Path]:
        assert not isinstance(idx, slice), "slice is not supported for MultiPath"
        assert idx >= 0, "Negative indices are not supported for MultiPath"

        path = self.path
        for _ in range(idx + 1):
            if len(path.parts) == 1:
                raise IndexError("Already reached the root path")

            path = path.parent

        return path


class MultiPathMixin:
    def __repr__(self):
        """
        Generates a user-friendly description of this multi-path instance.

        >>> path = MultiPath("/a/b")
        >>> path.add_alternative("/")
        >>> path.add_alternative("/x")
        >>> print(repr(path))
        Class: MultiPath
        Used location: / (exists=True)
        All locations:
        /a/b (exists=False)
        / (exists=True)
        /x (exists=False)

        >>> subpath = path / "y"
        >>> subpath.set_default_location("x")
        >>> print(repr(subpath))
        Class: MultiPath
        Used location (considering needle x): /x/y (exists=False)
        All locations:
        /a/b/y (exists=False)
        /y (exists=False)
        /x/y (exists=False)
        """
        text = f"Class: {self.__class__.__name__}\n"

        if self._default_needle is not None:
            repr_needle = f" (considering needle {self._default_needle})"
        else:
            repr_needle = ""
        best_location = self.find_best_location()
        text += f"Used location{repr_needle}: {best_location} (exists={best_location.exists()})\n"

        text += "All locations:\n"
        text += "\n".join([str(a) + f" (exists={a.exists()})" for a in self.possible_locations()])

        return text

    def __str__(self):
        # Paths are always converted to strings when they are used, e.g. on open or on .exists()
        # Here, we overwrite it to return the first existing path from all alternatives
        if hasattr(self, "_alternatives") and len(self._alternatives) > 0:
            return str(self.find_best_location())
        else:
            return super().__str__()

    @property
    def parent(self):
        alternatives = self._alternatives
        if str(self) in alternatives:
            # We go back to a standard Path object when we walk outside of one of the alternative locations because then there is no common subdirectory anymore to share between the alternatives
            return Path(str(self)).parent
        else:
            p = super().parent
            p._alternatives = self._alternatives
            p._default_needle = self._default_needle

            return p

    @property
    def parents(self):
        # We directly go to a standard Path since we don't know how many parents should be accessed at this point
        return MultiPathSequence(self)

    @property
    def name(self) -> str:
        # Some methods also rely on this property
        return self.find_best_location().name

    def iterdir(self, filter_func: Callable[[Path], bool] = None):
        # We also need to override the iterate methods to return paths from all alternatives
        for location in self.possible_locations(only_existing=True, filter_func=filter_func):
            yield from location.iterdir()

    def glob(self, pattern, filter_func: Callable[[Path], bool] = None):
        for location in self.possible_locations(only_existing=True, filter_func=filter_func):
            yield from location.glob(pattern)

    def rglob(self, pattern, filter_func: Callable[[Path], bool] = None):
        for location in self.possible_locations(only_existing=True, filter_func=filter_func):
            yield from location.rglob(pattern)

    def mkdir(self, *args, **kwargs):
        if self._default_needle is not None:
            location = self.find_best_location(writing=True)
        else:
            # If there is no default location, we want to create the dir in the root location (even if one of the other locations may exists)
            location = Path(super().__str__())

        location.mkdir(*args, **kwargs)

    def relative_to(self, *other) -> Path:
        if len(other) == 1 and isinstance(other[0], MultiPath):
            other = other[0]
            error = None

            # The first other location which is relative to self will be used
            for location in other.possible_locations():
                try:
                    return self.relative_to(location)
                except ValueError as e:
                    error = e

            assert error is not None, "The parent class should have risen an exception"
            raise error
        else:
            return super().relative_to(*other)

    def write_text(self, *args, **kwargs):
        return self.find_best_location().write_text(*args, **kwargs)

    def write_bytes(self, *args, **kwargs):
        return self.find_best_location().write_bytes(*args, **kwargs)

    def with_name(self, *args, **kwargs) -> Path:
        # New path with replaced name but same custom attributes
        path = MultiPath(super().with_name(*args, **kwargs))
        path._alternatives = self._alternatives
        path._default_needle = self._default_needle

        return path

    def with_stem(self, *args, **kwargs) -> Path:
        path = MultiPath(super().with_stem(*args, **kwargs))
        path._alternatives = self._alternatives
        path._default_needle = self._default_needle

        return path

    def with_suffix(self, *args, **kwargs) -> Path:
        path = MultiPath(super().with_suffix(*args, **kwargs))
        path._alternatives = self._alternatives
        path._default_needle = self._default_needle

        return path

    def resolve(self, *args, **kwargs) -> "Path":
        return self.find_best_location().resolve(*args, **kwargs)

    def add_alternative(self, path: Path | str) -> None:
        """Adds an alternative location to this path which will be replaced with the root location."""
        self._alternatives.append(str(unify_path(path, resolve_symlinks=False)))

    def set_default_location(self, location_needle: str | Path) -> None:
        """
        Sets a needle to select one of the possible locations for find_best_location(). This is mainly interesting for write operations to determine where new folders/files should be stored.

        Args:
            location_needle: Part of path of the default location (it is sufficient if a subset matches). The best match (according to the string similarity) will be used in case of non-unique matches.
        """
        if isinstance(location_needle, Path):
            location_needle = str(location_needle)

        self._default_needle = location_needle

    def find_location(self, needle: str) -> Path:
        """
        Searches all locations for the given needle.

        >>> path = MultiPath("/a/c")
        >>> path.add_alternative("/b/c")
        >>> str(path.find_location("b"))
        '/b/c'

        Args:
            needle: The search string which must match any part of any of the paths.

        Raises:
            FileNotFoundError: In case the needle matches none of the locations.

        Returns: The first matched location.
        """
        for location in self.possible_locations():
            if needle in str(location):
                return location

        raise FileNotFoundError(
            f"None of the locations ({[str(l) for l in self.possible_locations()]}) matches the needle {needle}"
        )

    def find_best_location(self, writing: bool = False) -> Path:
        """
        Selects the most suitable path from all alternatives using some heuristics (in this order):
            * If a needle is set and it matches, then:
                ** the match will be used if it exists
                ** it will also be used if the path is explicitly set for writing (see below)
            * If a needle is set but does not match, then:
                ** it will be used if no other alternative matches
            * If no needle is set, then:
                ** the first matching location will be used or
                ** the root location

        Args:
            writing: If True, explicitly indicates that you want to write to the current path (this enforces the needle to be used). If the needle matches, the matching path will be used even if it does not exist.

        Returns: The best path among all alternatives.
        """
        locations = self.possible_locations()

        # If we have a default location, we try this one first
        matched_location = None
        if self._default_needle is not None:
            if all(locations[0].name == l.name for l in locations):
                # Whenever we have a multipath point to a new file, it will be identical for all possible locations (e.g. dir1/file.txt and dir2/file.txt)
                # It does not make sense to include the file in this case since it is identical for all possible locations anyway
                # What is more, if "file" is also part of the common path, all locations will match (e.g. file/dir1/file.txt and dir2/file.txt, only the first one should match)
                matches = [l for l in locations if self._default_needle in str(l.parent)]
            else:
                matches = [l for l in locations if self._default_needle in str(l)]

            if len(matches) == 1:
                if matches[0].exists() or writing:
                    return matches[0]
                else:
                    matched_location = matches[0]
            elif len(matches) > 1:
                # Select the match with the highest string similarity to the needle
                matches_diff = [(m, SequenceMatcher(a=self._default_needle, b=str(m))) for m in matches]
                matches_diff = sorted(matches_diff, key=lambda x: x[1].ratio(), reverse=True)
                matches_diff = [m[0] for m in matches_diff]
                return matches_diff[0]

        # Otherwise, we look for existing files
        for location in locations:
            if location.exists():
                return location

        if matched_location is None:
            # None of the paths exists (or none of the default location matches), just return the root location
            return locations[0]
        else:
            # There was a match, but the path does not exist, still better than the root location
            return matched_location

    def possible_locations(self, only_existing=False, filter_func: Callable[[Path], bool] = None) -> list[Path]:
        """
        Lists all locations which can be accessed by this multi path.

        >>> path = MultiPath("/a/b")
        >>> path.add_alternative("/xx/a")
        >>> [str(p) for p in path.possible_locations()]
        ['/a/b', '/xx/a']

        Args:
            only_existing: Include only locations which exist.
            filter_func: Filter function to select locations. The function receives a paths and must return True if the path should be used.

        Returns: All possible locations for the current path.
        """
        path_str = super().__str__()

        # Find the root of the current main path (last match)
        alternatives_root = None
        for alternative in self._alternatives:
            if path_str.startswith(alternative):
                alternatives_root = alternative
                break

        if alternatives_root is None:
            # The base path is not part of any alternative so it is the only location (e.g., happens if a path is combined with an absolute path)
            return [Path(path_str)]
        else:
            # Replace the root with all alternatives to get all possible locations (including the main location)
            locations = []
            for alternative in self._alternatives:
                new = path_str.replace(alternatives_root, alternative)
                new = unify_path(new, resolve_symlinks=False)
                locations.append(new)

            if filter_func is not None:
                locations = [l for l in locations if filter_func(l)]

            if only_existing:
                locations = [l for l in locations if l.exists()]

            return locations


if sys.version_info >= (3, 12):

    class MultiPath(MultiPathMixin, Path):
        """
        This class can be used as an substitute for Path objects. It offers the possibility to have multiple root paths defined. If a new path is constructed and used, all alternatives are checked and the first which exists is used. This works best if relative file paths are unique across all alternatives.

        >>> path = MultiPath("/a/b")  # This path does not exist
        >>> path.add_alternative("/home")  # This path does exist
        >>> str(path)  # Print the string representation of the path which exists
        '/home'
        >>> path.name
        'home'

        For read operations, the path can be used as-is without further considerations. It will go over all locations and return the first one which exist.

        For write operations, the situation is a bit more complex, however. In principle, you also write to the best match, e.g. an existing location. However, you may want to explicitly write to a specific directory. For this, you need to make sure that you have a needle set. Then, you can either construct your path and mkdir your folders in which case a matching needle will always be used even if the match does not exist or explicitly construct your path and resolve it via `find_best_location(writing=True)`. Usually, the mkdir approach is sufficient.
        """

        def __init__(self, *pathsegments, _alternatives=None, _default_needle=None):
            super().__init__(*pathsegments)

            self._alternatives = _alternatives if _alternatives is not None else [str(self)]
            self._default_needle = _default_needle

            if self._alternatives is not None:
                # Always make sure that the path is set to the best location
                location = self.find_best_location()
                super().__init__(location)

        def with_segments(self, *pathsegments):
            # Called whenever a derivative of a path is created (https://docs.python.org/3.12/library/pathlib.html#pathlib.PurePath.with_segments)
            return type(self)(*pathsegments, _alternatives=self._alternatives, _default_needle=self._default_needle)

        def __reduce__(self):
            # Called when pickling path objects (e.g. multiprocessing)
            deserializer = partial(
                self.__class__, _alternatives=self._alternatives, _default_needle=self._default_needle
            )
            args = self.parts
            return deserializer, args

else:

    class MultiPath(MultiPathMixin, type(Path())):
        """
        This class can be used as an substitute for Path objects. It offers the possibility to have multiple root paths defined. If a new path is constructed and used, all alternatives are checked and the first which exists is used. This works best if relative file paths are unique across all alternatives.

        >>> path = MultiPath("/a/b")  # This path does not exist
        >>> path.add_alternative("/home")  # This path does exist
        >>> str(path)  # Print the string representation of the path which exists
        '/home'
        >>> path.name
        'home'

        For read operations, the path can be used as-is without further considerations. It will go over all locations and return the first one which exist.

        For write operations, the situation is a bit more complex, however. In principle, you also write to the best match, e.g. an existing location. However, you may want to explicitly write to a specific directory. For this, you need to make sure that you have a needle set. Then, you can either construct your path and mkdir your folders in which case a matching needle will always be used even if the match does not exist or explicitly construct your path and resolve it via `find_best_location(writing=True)`. Usually, the mkdir approach is sufficient.
        """

        def __new__(cls, *args, **kwargs):
            if len(args) == 1:
                if type(args[0]) == dict:
                    # Construction from pickled object

                    # e.g. .../2021_02_05_Tivita_multiorgan_masks/intermediates/preprocessing/L1
                    path = super().__new__(cls, args[0]["path"])
                    # e.g. [.../2021_02_05_Tivita_multiorgan_semantic/intermediates, .../2021_02_05_Tivita_multiorgan_masks/intermediates, ...]
                    path._alternatives = args[0]["alternatives"]
                    # e.g. 2021_02_05_Tivita_multiorgan_masks
                    path._default_needle = args[0]["default_needle"]

                    return path
                else:
                    # Default construction, we just make sure that the path is expanded

                    # Normalize the path and make it absolute without resolving symbolic links as this may break the logic of the MultiPath class which relies on path replacements
                    # For example, after resolving a symlink the path may not contain any of the alternatives anymore so it is impossible to do the replacements
                    new_args = [str(unify_path(args[0], resolve_symlinks=False))]
            else:
                # Construction from parts
                new_args = args

            super_path = super().__new__(cls, *new_args, **kwargs)

            # Custom attributes
            super_path._alternatives = [str(super_path)]
            super_path._default_needle = None
            super_path._set_attributes()

            return super_path

        def _make_child(self, args):
            if len(args) == 1 and Path(args[0]).is_absolute():
                # If the child path is already absolute, we can just use it as-is
                abs_path = args[0]
                if type(abs_path) == str:
                    abs_path = Path(abs_path)
                return abs_path
            else:
                # Any child path which is created via base / new should also receive the additional class attributes
                child = super()._make_child(args)
                child._alternatives = self._alternatives
                child._default_needle = self._default_needle
                child._set_attributes()

                return child

        def _set_attributes(self):
            # The attributes are always based on the current best location
            location = self.find_best_location()
            self._drv = location._drv
            self._root = location._root
            self._parts = location._parts

        def __reduce__(self):
            # Called when pickling path objects (e.g. multiprocessing)
            kwargs = {
                "path": super().__str__(),
                "alternatives": self._alternatives,
                "default_needle": self._default_needle,
            }
            return (self.__class__, (kwargs,))
