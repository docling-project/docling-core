#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the methods in module search.json_schema_to_search_mapper."""
import json
import os

import jsondiff

from docling_core.search.json_schema_to_search_mapper import JsonSchemaToSearchMapper
from docling_core.types.legacy_doc.document import ExportedCCSDocument
from docling_core.types.rec.record import Record


def _load(filename):
    doc = {}
    with open(filename, "r", encoding="utf-8") as fid:
        doc = json.load(fid)
    return doc


def test_json_schema_to_search_mapper_0():
    """Test the class JsonSchemaToSearchMapper."""
    s = ExportedCCSDocument.model_json_schema()

    mapper = JsonSchemaToSearchMapper(
        mappings_extra={
            "_meta": {
                "license": "",
                "created": "2021-09-27T17:42:10.407214+00:00",
                "description": "",
                "source": "",
                "display_name": "",
                "version": "1.0",
                "$ref": "ccs:schemas#/Document",
            },
            "dynamic": False,
            "_size": {"enabled": True},
        }
    )

    index_def = mapper.get_index_definition(s)

    assert index_def is not None

    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data/json_schemas/document-ref.json")
    )
    index_ref = _load(filename)

    diff = jsondiff.diff(index_ref, index_def)
    # print(json.dumps(index_def, indent=2))
    assert (
        index_def == index_ref
    ), f"Error in search mappings of ExportedCCSDocument. Difference:\n{json.dumps(diff, indent=2)}"


def test_json_schema_to_search_mapper_1():
    """Test the class JsonSchemaToSearchMapper."""
    s = Record.model_json_schema()
    print(json.dumps(s, indent=2))

    _meta = {
        "aliases": [".production", "ccc"],
        "created": "2022-11-03T11:22:32.432+00:00",
        "description": "description of the collection",
        "source": "https://ccc",
        "storage": "storage location",
        "display_name": "display name",
        "type": "Record",
        "classification": ["Public", "PI"],
        "version": [
            {"name": "my-library", "version": "0.1.0-post.6+ed04c14"},
            {"name": "docling-core", "version": "0.1.0"},
        ],
        "document_license": {"code": ["NO-CC CODE", "CC BY"], "text": []},
        "license": "https://www.ccc",
        "filename": "ccc-gs.json",
        "domain": ["Healthcare & Life Sciences"],
        "$ref": "ccs:schemas#/Document",
    }

    mapper = JsonSchemaToSearchMapper(
        mappings_extra={
            "_meta": _meta,
            "dynamic": False,
            "_size": {"enabled": True},
        }
    )

    index_def = mapper.get_index_definition(s)

    assert index_def is not None

    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data/json_schemas/dbrecord-ref.json")
    )
    index_ref = _load(filename)

    diff = jsondiff.diff(index_ref, index_def)
    # print(json.dumps(index_def, indent=2))
    assert (
        index_def == index_ref
    ), f"Error in search mappings of Record. Difference:\n{json.dumps(diff, indent=2)}"
