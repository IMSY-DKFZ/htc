# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib
import re
import sys

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
        class_definition: Class definition in the form module>class (e.g. htc.models.image.LightningImage>LightningImage). The first part (module) may also be the full path to the Python file.

    Returns: Class type.
    """
    global _type_cache
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
            # Try path importing (https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly)
            spec = importlib.util.spec_from_file_location(match.group(2), match.group(1))
            module = importlib.util.module_from_spec(spec)
            sys.modules[match.group(2)] = module
            spec.loader.exec_module(module)

        _type_cache[class_definition] = getattr(module, match.group(2))

    return _type_cache[class_definition]
