# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json

from htc.settings import settings
from htc.utils.JSONSchemaMeta import JSONSchemaMeta


class TestJSONSchemaMeta:
    def test_basics(self) -> None:
        with (settings.data_dirs.masks / "meta.schema").open() as f:
            schema = json.load(f)
        js = JSONSchemaMeta(schema)
        assert js.meta("type") == "object"

        assert js["label_meta/kidney/angle"].meta("type") == "integer"
        assert js["label_meta/something/nonexistent"] is None

        js = js["label_meta"]
        assert js.meta("type") == "object"
        assert not js.has_meta("unit")
        assert js.meta("unit") is None
        assert js.meta("level_of_measurement") is None

        js = js["kidney"]
        assert js.meta("type") == "object"

        js = js["angle"]
        assert js.meta("type") == "integer"
        assert js.has_meta("unit")
        assert js.meta("unit") == "degree"
        assert js.meta("level_of_measurement") == "interval"
