# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging


class DuplicateFilter(logging.Filter):
    # Show logger messages only once (https://stackoverflow.com/a/31953563/2762258)
    def __init__(self):
        super().__init__()
        self.msgs = set()

    def filter(self, record):  # noqa: A003
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv
