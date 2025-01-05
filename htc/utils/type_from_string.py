# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib
import re
import sys
from pathlib import Path
from typing import Any

from htc.settings import settings

_type_cache = {}


def type_from_string(class_definition: str) -> type:
    """
    Parses a string for a class definition and imports the class type.

    Note: The class definition is caches so that it is only parsed once per program.

    This works for any class which can be imported
    >>> ConfigClass = type_from_string("htc.utils.Config>Config")
    >>> config = ConfigClass({"data": 1})
    >>> config
    {'data': 1}

    Similarly for any class with the path to the script defining that class
    >>> from htc.settings import settings
    >>> ConfigClass = type_from_string(str(settings.htc_package_dir / "utils/Config.py") + ">Config")
    >>> config = ConfigClass({"data": 2})
    >>> config
    {'data': 2}

    Args:
        class_definition: Class definition in the form module>class (e.g. htc.models.image.LightningImage>LightningImage). The first part (module) may also be the path to the Python file (absolute, relative, or relative to the src/htc/htc_projects directory).

    Returns: Class type.
    """
    global _type_cache

    if "htc.context" in class_definition:
        # Some models out there may still contain the path to the old module
        class_definition = class_definition.replace("htc.context", "htc_projects.context")

    if class_definition not in _type_cache:
        match = re.search(r"^([^>]+)>(\w+)$", class_definition)
        assert match is not None, (
            f"Could not parse the string {class_definition} as a class definition. It must be in the format"
            " module>class (e.g. htc.models.image.LightningImage>LightningImage) and must refer to a valid Python"
            " class"
        )

        try:
            module = importlib.import_module(match.group(1))
        except ModuleNotFoundError:
            path = _find_existing_path(match.group(1))
            # Try path importing (https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly)
            spec = importlib.util.spec_from_file_location(match.group(2), path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[match.group(2)] = module
            spec.loader.exec_module(module)

        _type_cache[class_definition] = getattr(module, match.group(2))

    return _type_cache[class_definition]


def variable_from_string(definition: str) -> Any:
    """
    Parses a string for a variable definition and imports the variable.

    This works for any variable which can be imported
    >>> mapping = variable_from_string("htc.settings_seg>label_mapping")
    >>> len(mapping)
    19

    It is also possible to import a variable via the path to the script
    >>> from htc.settings import settings
    >>> mapping = variable_from_string(str(settings.htc_package_dir / "settings_seg.py") + ">label_mapping")
    >>> len(mapping)
    19

    Args:
        definition: Variable definition in the form module>variable (e.g. htc.settings_seg>label_mapping). The first part (module) may also be the path to the Python file (absolute, relative, or relative to the src/htc/htc_projects directory).

    Returns: The imported variable.
    """
    match = re.search(r"^([^>]+)>(\w+)$", definition)
    assert match is not None, (
        f"Could not parse the string {definition} as a valid variable definition. It must be in the format"
        " module>variable (e.g. htc.settings_seg>label_mapping) and must refer to a valid Python script"
    )

    try:
        module = importlib.import_module(match.group(1))
        is_path = False
    except ModuleNotFoundError:
        # Try path importing (https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly)
        path = _find_existing_path(match.group(1))
        spec = importlib.util.spec_from_file_location(match.group(2), path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[match.group(2)] = module
        spec.loader.exec_module(module)
        is_path = True

    if not hasattr(module, match.group(2)):
        if is_path:
            name = Path(match.group(1)).stem
        else:
            name = match.group(1).split(".")[-1]

        # For example, if settings is an object
        module = getattr(module, name)

    return getattr(module, match.group(2))


def _find_existing_path(file_location: str) -> Path:
    possible_paths = [
        Path(file_location),
        settings.htc_package_dir / file_location,
        settings.htc_projects_dir / file_location,
        settings.src_dir / file_location,
    ]
    selected_path = None
    for path in possible_paths:
        if path.exists():
            selected_path = path
            break

    if selected_path is None:
        raise FileNotFoundError(
            f"Could not find the file {file_location}. Tried the following locations:\n{possible_paths}"
        )

    return selected_path
