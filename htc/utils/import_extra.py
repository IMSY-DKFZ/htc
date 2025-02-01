# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import functools
from collections.abc import Callable


def requires_extra(missing_library: str) -> Callable:
    """
    Decorator which can be used to specify that a function needs an extra library which is not installed by default (the user has to install it via `pip install imsy-htc[extra]`).

    The general pattern is to import the library at the top in a try block:
    ```python
    try:
        from challenger_pydocker import ChallengeR

        _missing_library = ""
    except ImportError:
        _missing_library = "challenger_pydocker"


    @requires_extra(_missing_library)
    def my_function():
        pass
    ```

    Args:
        missing_library: Name of the library which is missing or empty string if the library is present.
    """

    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if missing_library != "":
                try:
                    import pytest

                    pytest.skip(f"{missing_library} is missing but required for the test")
                except ImportError:
                    pass

                raise ImportError(
                    f"{missing_library} library is missing. You can fix this by installing the extra dependencies"
                    " (`imsy-htc[extra]`) as described in the README or by installing additional dependencies as described"
                    " in the documentation of the class/function you want to use or by installing the requirements of the project where the class/function you want to use is located (e.g., dependencies/requirements-organ-clamping.txt)."
                )
            return function(*args, **kwargs)

        return wrapper

    return decorator
