# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging
from collections.abc import Callable
from pathlib import Path


class DelayedFileHandler(logging.Handler):
    """
    This class can be used similarly to logging.FileHandler but any log messages will be cached until set_filename is called.

    This is useful if the path to the log file does not exist from the beginning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_records = []
        self.file_handler = None

    def set_filename(self, filename: Path, **kwargs) -> None:
        """
        Set the filename of the log file. After this function is called, the cached logs are written to disk.

        Args:
            filename: Path to the log file.
            kwargs: Any additional keyword arguments passed to logging.FileHandler.
        """
        self.file_handler = logging.FileHandler(filename, **kwargs)

        # Apply existing settings to the new file handler
        for f in self.filters:
            self.file_handler.addFilter(f)
        self.file_handler.setFormatter(self.formatter)
        self.file_handler.setLevel(self.level)

        # Emit all cached log records
        for record in self.cached_records:
            self.file_handler.emit(record)
        self.cached_records = []

    def addFilter(self, filter_func: logging.Filter | Callable) -> None:
        if self.file_handler is None:
            super().addFilter(filter_func)
        else:
            self.file_handler.addFilter(filter_func)

    def setFormatter(self, fmt: str) -> None:
        if self.file_handler is None:
            super().setFormatter(fmt)
        else:
            self.file_handler.setFormatter(fmt)

    def setLevel(self, level: str | int) -> None:
        if self.file_handler is None:
            super().setLevel(level)
        else:
            self.file_handler.setLevel(level)

    def emit(self, record: logging.LogRecord) -> None:
        if self.file_handler is None:
            self.cached_records.append(record)
        else:
            self.file_handler.emit(record)
