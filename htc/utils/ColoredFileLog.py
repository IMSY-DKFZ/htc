# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io
import logging
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.logging import RichHandler

from htc.settings import ColoredFormatter, settings
from htc.utils.import_extra import requires_extra

try:
    import ansi2html

    _missing_library = ""
except ImportError:
    _missing_library = "ansi2html"


class ColoredFileLog:
    @requires_extra(_missing_library)
    def __init__(self, log_file: Union[str, Path]):
        """
        Write colored log messages to a formatted HTML file.

        >>> import tempfile
        >>> logging.getLogger().handlers.clear()  # We disable logging just for the doctest
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     tmp_dir = Path(tmp_dir)
        ...     with ColoredFileLog(tmp_dir / "log.html"):
        ...         settings.log.info("[red]My colored message![/]")
        ...     (tmp_dir / 'log.html').exists() and (tmp_dir / 'log.html').stat().st_size > 0
        True

        Args:
            log_file: Path where the HTML logfile should be stored. You can use log_file=__file__ and the log will be stored in the same location as the script file but with .html extension.
        """
        if type(log_file) == str:
            log_file = Path(log_file)
        if log_file.suffix != ".html":
            log_file = log_file.with_suffix(".html")
        self.log_file = log_file
        self.string_io = io.StringIO()

    def __enter__(self):
        file_handler = RichHandler(
            markup=True,
            show_time=False,
            show_level=False,
            show_path=False,
            console=Console(color_system="truecolor", file=self.string_io),
        )
        file_handler.setFormatter(ColoredFormatter("[%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(file_handler)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.string_io.seek(0)
        html = ansi2html.Ansi2HTMLConverter().convert(self.string_io.read())
        with self.log_file.open("w") as f:
            f.write(html)
        settings.log.info(f"Log for this script stored at {self.log_file}")
