# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from timeit import default_timer


class MeasureTime:
    def __init__(self, name: str = "", silent: bool = False):
        """
        Easily measure the time of a Python code block.

        >>> import time
        >>> with MeasureTime() as m:
        ...     time.sleep(1)  # doctest: +ELLIPSIS
        Elapsed time: 0 m and 1... s
        >>> round(m.elapsed_seconds)
        1

        Args:
            name: Name which is included in the time info message.
            silent: Whether to print the time info message.
        """
        self.name = name
        self.silent = silent
        self.elapsed_seconds = 0

    def __enter__(self):
        self.start = default_timer()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = default_timer()
        seconds = end - self.start

        if self.name:
            tag = "[" + self.name + "] "
        else:
            tag = ""

        self.elapsed_seconds = seconds

        if not self.silent:
            print("%sElapsed time: %d m and %.2f s" % (tag, seconds // 60, seconds % 60))
