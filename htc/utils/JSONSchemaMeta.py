# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
from pathlib import Path
from typing import Union

from typing_extensions import Self


class JSONSchemaMeta:
    def __init__(self, schema: Union[dict, Path]) -> None:
        """
        Helper class to access meta information from a JSON schema.

        This is useful if you want to access meta information about attributes, e.g. type, unit, etc.:
        >>> js = JSONSchemaMeta({"type": "object", "properties": {"a": {"type": "integer", "unit": "m"}}})

        An example instance of this schema could be:
        >>> import jsonschema
        >>> obj = {
        ...    "a": 2
        ... }
        >>> jsonschema.validate(instance=obj, schema=js.schema)

        Type of the (root) object:
        >>> js.meta("type")
        'object'

        Similar to a nested dictionary, the meta information is nested as well and meta information about sub-attributes can be accessed by the same key as of the original dictionary:
        >>> js["a"].meta("type")
        'integer'
        >>> js["a"].meta("unit")
        'm'

        Args:
            schema: The schema to use. Can be a dict or a Path to a JSON file. The schema must be valid.
        """
        if isinstance(schema, Path):
            with schema.open() as f:
                self.schema = json.load(f)
        else:
            self.schema = schema

        try:
            import jsonschema

            # Make sure the schema is valid
            jsonschema.Draft4Validator.check_schema(self.schema)
        except ImportError:
            pass

    def __getitem__(self, identifier: str) -> Union[Self, None]:
        """
        Access a sub-schema by its identifier.

        Args:
            identifier: Identifier of the sub-schema, i.e. key of the corresponding object. The identifier can also be used to access nested subschemas (e.g. `key1/key2`) similar to the `Config` class.

        Returns: The sub-schema or None if the identifier does not exist.
        """
        current_obj = self
        keys = identifier.split("/")

        for key in keys:
            if current_obj is None:
                break

            if "properties" in current_obj.schema and key in current_obj.schema["properties"]:
                current_obj = JSONSchemaMeta(current_obj.schema["properties"][key])
            elif "patternProperties" in current_obj.schema:
                match = None
                for pattern, subschema in current_obj.schema["patternProperties"].items():
                    if re.search(pattern, key) is not None:
                        match = JSONSchemaMeta(subschema)
                        break

                if match is not None:
                    current_obj = match
                else:
                    current_obj = None
            else:
                current_obj = None

        return current_obj

    def meta(self, name: str) -> Union[str, None]:
        """
        Access meta information about the current object.

        Args:
            name: Name of the meta attribute (e.g. `type`).

        Returns: The associated meta information or None if the name does not exist.
        """
        return self.schema.get(name)

    def has_meta(self, name: str) -> bool:
        """
        Check if the current object has meta information.

        Args:
            name: Name of the meta attribute (e.g. `type`).

        Returns: True if the meta information exists, False otherwise.
        """
        return name in self.schema
