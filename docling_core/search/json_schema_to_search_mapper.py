#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Methods to convert a JSON Schema into a search database schema."""
import re
from copy import deepcopy
from typing import Any, Optional, Pattern, Tuple, TypedDict

from jsonref import JsonRef


class SearchIndexDefinition(TypedDict):
    """Data type for an index basic definition (settings and mappings)."""

    settings: dict
    mappings: dict


class JsonSchemaToSearchMapper:
    """Map a JSON Schema to an search database schema.

    The generated database schema is a mapping describing the fields from the
    JSON Schema and how they should be indexed in a Lucene index of a search database.

    Potential issues:
    - Tuples may not be converted properly (e.g., Tuple[float,float,float,str,str])
    - Method `_remove_keys` may lead to wrong results if a field is named `properties`.
    """

    def __init__(
        self,
        settings_extra: Optional[dict] = None,
        mappings_extra: Optional[dict] = None,
    ):
        """Create an instance of the mapper with default settings."""
        self.settings = {
            "analysis": {
                # Create a normalizer for lowercase ascii folding,
                # this is used in keyword fields
                "normalizer": {
                    "lowercase_asciifolding": {
                        "type": "custom",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            }
        }

        self.settings_extra = settings_extra
        self.mappings_extra = mappings_extra

        self._re_es_flag = re.compile(r"^(?:x-es-)(.*)")

        self._rm_keys = (
            "description",
            "required",
            "title",
            "additionalProperties",
            "format",
            "enum",
            "pattern",
            "$comment",
            "default",
            "minItems",
            "maxItems",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "$defs",
            "const",
        )

        self._suppress_key = "x-es-suppress"

        self._type_format_mappings: dict[tuple[str, str], str] = {
            ("string", "date-time"): "date",
        }

        self._type_mappings = {
            "number": "double",
            "string": "text",
        }

        self._types_to_remove = ("object",)

    def get_index_definition(self, schema: dict) -> SearchIndexDefinition:
        """Generates a search database schema from a JSON Schema.

        The search database schema consists of the sections `settings` and `mappings`,
        which define the fields, their data types, and other specifications to index
        JSON documents into a Lucene index.
        """
        mapping = JsonRef.replace_refs(schema)

        mapping = self._merge_unions(mapping)

        mapping = self._clean_types(mapping)

        mapping = self._collapse_arrays(mapping)

        mapping = self._remove_keys(mapping, self._rm_keys)

        mapping = self._suppress(mapping, self._suppress_key)

        mapping = self._translate_keys_re(mapping)

        mapping = self._clean(mapping)

        mapping.pop("definitions", None)

        result = SearchIndexDefinition(
            settings=self.settings,
            mappings=mapping,
        )

        if self.mappings_extra:
            result["mappings"] = {**result["mappings"], **self.mappings_extra}

        if self.settings_extra:
            result["settings"] = {**result["settings"], **self.settings_extra}

        return result

    def _merge_unions(self, doc: dict) -> dict:
        """Merge objects of type anyOf, allOf, or oneOf (options).

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.

        Returns:
            A transformation of a JSON schema by merging option fields.
        """

        def _clean(value: Any) -> Any:
            if isinstance(value, list):
                return [_clean(v) for v in value]

            if isinstance(value, dict):
                union: list = []
                merged_union: dict = {}

                for k, v in value.items():
                    if k in ("oneOf", "allOf", "anyOf"):
                        union.extend(v)
                    else:
                        merged_union[k] = v

                if not union:
                    return {k: _clean(v) for k, v in value.items()}

                for u in union:
                    if not isinstance(u, dict):
                        continue

                    for k, v in u.items():
                        if k == "type" and v == "null":  # null values are irrelevant
                            continue
                        elif not isinstance(v, dict) or k not in merged_union:
                            merged_union[k] = _clean(v)
                        elif isinstance(v, dict) and k in merged_union:
                            merged_union[k] = _clean({**merged_union[k], **v})

                return merged_union

            return value

        return _clean(doc)

    def _clean_types(self, doc: dict) -> dict:
        """Clean field types originated from a JSON schema to obtain search mappings.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.

        Returns:
            A transformation of a JSON schema by merging option fields.
        """

        def _clean(value: Any) -> Any:
            if isinstance(value, list):
                return [_clean(v) for v in value]

            if isinstance(value, dict):
                if isinstance(value.get("type"), str):
                    t: str = value["type"]

                    # Tuples
                    if t == "array" and isinstance(value.get("items"), list):
                        items: list = value["items"]

                        if items:
                            value["items"] = value["items"][0]
                        else:
                            value["items"] = {}

                    # Unwanted types, such as 'object'
                    if t in self._types_to_remove:
                        value.pop("type", None)

                    # Map formats
                    f: str = value.get("format", "")
                    if (t, f) in self._type_format_mappings:
                        value["type"] = self._type_format_mappings[(t, f)]
                        value.pop("format", None)

                    # Map types, such as 'string' to 'text'
                    elif t in self._type_mappings:
                        value["type"] = self._type_mappings[t]

                return {k: _clean(v) for k, v in value.items()}

            return value

        return _clean(doc)

    @staticmethod
    def _collapse_arrays(doc: dict) -> dict:
        """Collapse arrays from a JSON schema to match a search database mappings.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.

        Returns:
            A transformation of a JSON schema by collapsing arrays.
        """

        def __collapse(d_: Any) -> Any:
            if isinstance(d_, list):
                return [v for v in (__collapse(v) for v in d_)]

            if isinstance(d_, dict):
                if "type" in d_ and d_["type"] == "array" and "items" in d_:
                    collapsed = __collapse(d_["items"])

                    d_ = deepcopy(d_)
                    d_.pop("items", None)
                    d_.pop("type", None)

                    merged = {**d_, **collapsed}

                    return merged

                return {k: __collapse(v) for k, v in d_.items()}

            return d_

        return __collapse(doc)

    @staticmethod
    def _suppress(doc: dict, suppress_key: str) -> dict:
        """Remove a key from a JSON schema to match a search database mappings.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.
            key: The name of a field to be removed from the `doc`.

        Returns:
            A transformation of a JSON schema by removing the field `suppress_key`.
        """

        def __suppress(d_: Any) -> Any:
            if isinstance(d_, list):
                return [v for v in (__suppress(v) for v in d_)]

            if isinstance(d_, dict):
                if suppress_key in d_ and d_[suppress_key] is True:
                    return {}
                else:
                    return {
                        k: v for k, v in ((k, __suppress(v)) for k, v in d_.items())
                    }
            return d_

        return __suppress(doc)

    @staticmethod
    def _remove_keys(doc: dict, keys: Tuple[str, ...]) -> dict:
        """Remove keys from a JSON schema to match a search database mappings.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.
            keys: Fields to be removed from the `doc`.

        Returns:
            A transformation of a JSON schema by removing the fields in `keys`.
        """

        def __remove(d_: Any) -> Any:
            if isinstance(d_, list):
                return [v for v in (__remove(v) for v in d_)]

            if isinstance(d_, dict):
                result = {}
                for k, v in d_.items():
                    if k == "properties" and isinstance(v, dict):
                        # All properties must be included, they are not to be removed,
                        # even if they have a name of a key that's to be removed.
                        result[k] = {p_k: __remove(p_v) for p_k, p_v in v.items()}
                    elif k not in keys:
                        result[k] = __remove(v)

                return result

            return d_

        return __remove(doc)

    @staticmethod
    def _remove_keys_re(doc: dict, regx: Pattern) -> dict:
        """Remove keys from a JSON schema to match a search database mappings.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.
            keys: A pattern defining the fields to be removed from the `doc`.

        Returns:
            A transformation of a JSON schema by removing fields with a name pattern.
        """

        def __remove(d_: Any) -> Any:
            if isinstance(d_, list):
                return [v for v in (__remove(v) for v in d_)]

            if isinstance(d_, dict):
                return {
                    k: v
                    for k, v in (
                        (k, __remove(v)) for k, v in d_.items() if not regx.match(k)
                    )
                }

            return d_

        return __remove(doc)

    def _translate_keys_re(self, doc: dict) -> dict:
        """Translate marked keys from a JSON schema to match a search database mappings.

        The keys to be translated should have a name that matches the pattern defined
        by this class patter, for instance, a name starting with `x-es-`.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.

        Returns:
            A transformation of a JSON schema towards a search database mappings.
        """

        def __translate(d_: Any) -> Any:
            if isinstance(d_, list):
                return [v for v in (__translate(v) for v in d_)]

            if isinstance(d_, dict):
                new_dict = {}
                for k, v in d_.items():
                    new_dict[k] = __translate(v)

                delkeys = []
                for k in list(new_dict.keys()):
                    k_ = self._re_es_flag.sub(r"\1", k)
                    if k_ != k:
                        new_dict[k_] = new_dict[k]
                        delkeys.append(k)

                for k in delkeys:
                    new_dict.pop(k, None)

                return new_dict

            return d_

        return __translate(doc)

    @staticmethod
    def _clean(doc: dict) -> dict:
        """Recursively remove empty lists, dicts, strings, or None elements from a dict.

        Args:
            doc: A JSON schema or a transformation towards a search database mappings.

        Returns:
            A transformation of a JSON schema by removing empty objects.
        """

        def _empty(x) -> bool:
            return x is None or x == {} or x == [] or x == ""

        def _clean(d_: Any) -> Any:
            if isinstance(d_, list):
                return [v for v in (_clean(v) for v in d_) if not _empty(v)]

            if isinstance(d_, dict):
                return {
                    k: v
                    for k, v in ((k, _clean(v)) for k, v in d_.items())
                    if not _empty(v)
                }

            return d_

        return _clean(doc)
