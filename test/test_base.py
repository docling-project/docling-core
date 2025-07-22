#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module data_types.base.py."""
import json
from datetime import datetime, timezone
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from docling_core.types.base import (
    CollectionDocumentInfo,
    CollectionInfo,
    CollectionRecordInfo,
    FileInfoObject,
    Identifier,
    Log,
    StrictDateTime,
)
from docling_core.types.legacy_doc.document import CCSDocumentDescription
from docling_core.types.rec.record import RecordDescription


def test_identifier():
    """Validate data with Identifier model."""
    gold_dict = {"type": "id", "value": "abc", "_name": "id#abc"}
    data = Identifier(type="id", value="abc", _name="id#abc")

    # dict(): important to set by_alias=True, if the model has aliases
    assert data.model_dump(by_alias=True) == gold_dict
    assert data.model_dump_json(by_alias=True, indent=2) == json.dumps(
        gold_dict, indent=2
    )

    # schema_json(): no need to set by_alias since it is True by the default
    with open("test/data/json_schemas/base_identifier.json", encoding="utf-8") as tf:
        gold_json = json.load(tf)

    assert Identifier.model_json_schema() == gold_json

    gold_dict = {"type": "id", "value": "ABC", "_name": "id#abc"}
    valid_keys = Literal["id", "doi", "uid"]
    data = Identifier[valid_keys](type="id", value="ABC", _name="id#abc")

    assert data.model_dump(by_alias=True) == gold_dict
    assert data.model_dump(by_alias=True, exclude_unset=True) == gold_dict

    with pytest.raises(ValidationError, match="type"):
        Identifier[valid_keys](type="arxivid", value="ABC", _name="arxivid#abc")

    with pytest.raises(ValidationError, match="concatenation"):
        Identifier[str](type="id", value="ABC", _name="id#ABC")

    with pytest.raises(ValidationError, match="required"):
        Identifier[str](type="id", value="abc")
        Identifier[str](value="abc")

    with pytest.raises(ValidationError, match="_name"):
        Identifier[str](type="id", value="abc", _name=None)

    with pytest.raises(ValidationError, match="type"):
        Identifier[str](type=None, value="abc", _name="abc")

    with pytest.raises(ValidationError, match="comment"):
        Identifier[str](type="id", value="abc", _name="id#abc", comment="OK")


def test_log():
    """Validate data with Log model."""
    Log(agent="CXS", type="annotation", date=datetime.now())

    Log(
        task="run 3",
        agent="CXS",
        type="annotation",
        comment="UCMI 3.10",
        date="2021-11-03T04:42:54.844631+00:00",
    )
    data = Log(
        task=None, agent="CXS", type="parsing", date="2021-11-03T04:42:54.844631+00:00"
    )

    gold_dict = {
        "agent": "CXS",
        "type": "parsing",
        "date": "2021-11-03T04:42:54.844631+00:00",
    }
    # None values will be exported, use exclude_none=True to export clean
    assert data.model_dump() != gold_dict
    assert data.model_dump(exclude_none=True, by_alias=True) == gold_dict
    # Optional unset parameters will be exported as null, use exclude_unset=True
    assert Log(**gold_dict).model_dump() != gold_dict
    assert Log(**gold_dict).model_dump(exclude_unset=True, by_alias=True) == gold_dict
    # Models that inherit from AliasModel will generate data with alias field names
    assert Log(**gold_dict).model_dump(exclude_unset=True) == gold_dict
    # ***Best practice***: exclude_unset=True, exclude_none=True, by_alias=True
    assert (
        Log(**gold_dict).model_dump(
            exclude_unset=True, exclude_none=True, by_alias=True
        )
        == gold_dict
    )

    with open("test/data/json_schemas/base_log.json", encoding="utf-8") as tf:
        gold_json_schema = json.load(tf)
    assert Log.model_json_schema() == gold_json_schema

    with pytest.raises(
        ValidationError, match="Value type must be a datetime or a non-numeric string"
    ):
        Log(agent="CXS", type="annotation", date=123456789)


def test_file_info_object():
    """Validate data with FileInfoObject model."""
    gold_dict = {
        "filename": "document.pdf",
        "filename-prov": "http:www.ibm.com",
        "document-hash": "PnNF3Fhr22nJH4a",
    }
    data = FileInfoObject(**gold_dict)
    # dictionaries and JSON exports need to explicitly use aliases, but children from AliasModel don't.
    assert data.model_dump(by_alias=True) == gold_dict
    assert data.model_dump() == gold_dict

    gold_dict.pop("filename-prov")
    gold_json = json.dumps(gold_dict)
    FileInfoObject(**gold_dict).model_dump_json(
        exclude_unset=True, exclude_none=True
    ) == gold_json

    # creating an instance with input variables requires the use of field names. Since
    # document-hash is an invalid function parameter name, 'populate_by_name' needs to
    # be set to True in model definition. For convenience, inherit from AliasModel.
    FileInfoObject(filename="document.pdf", document_hash="PnNF3Fhr22nJH4a")


def test_collection_info():
    """Validate data with CollectionInfo model."""

    # Test 1
    gold_dict = {
        "name": "patent USPTO",
        "type": "Document",
        "version": "3.2.0",
        "alias": ["patent"],
    }
    data = CollectionInfo(**gold_dict)
    assert data.model_dump(exclude_unset=True, exclude_none=True) == gold_dict

    # Test 2
    gold_dict = {
        "name": "patent USPTO",
        "type": "experiment",
        "version": "3.2.0",
        "alias": ["simulation"],
    }
    with pytest.raises(ValidationError, match="type"):
        CollectionInfo(**gold_dict)

    # Test 3
    input_dict = {
        "name": "patent USPTO",
        "type": "Document",
        "version": "3.2.0",
        "alias": None,
    }
    clean_dict = {"name": "patent USPTO", "type": "Document", "version": "3.2.0"}
    data = CollectionInfo(**input_dict)
    assert (
        data.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
        != input_dict
    )
    assert (
        data.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
        == clean_dict
    )
    data = CollectionInfo(**clean_dict)
    assert (
        data.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
        == clean_dict
    )


def test_collection_document_info():
    """Validate data with CollectionDocumentInfo model."""
    gold_dict = {
        "name": "patent USPTO",
        "type": "Document",
        "version": "3.2.0",
        "alias": ["patent"],
    }
    data = CollectionDocumentInfo(**gold_dict)
    assert (
        data.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
        == gold_dict
    )

    # within dictionary
    desc_dict = {
        "logs": [
            {
                "date": "2021-11-03T04:42:54.844631+00:00",
                "agent": "CXS",
                "type": "parsing",
            }
        ],
        "collection": {
            "name": "patent USPTO",
            "type": "Document",
            "version": "3.2.0",
            "alias": ["patent"],
        },
    }
    CCSDocumentDescription(**desc_dict)

    desc_dict["collection"]["type"] = "Record"
    with pytest.raises(ValidationError, match="collection.type"):
        CCSDocumentDescription(**desc_dict)


def test_collection_record_info():
    """Validate data with CollectionRecordInfo model."""
    gold_dict = {
        "name": "PubChem",
        "type": "Record",
        "version": "3.2.0",
        "alias": ["chemical", "Material Sciences"],
    }
    data = CollectionRecordInfo(**gold_dict)
    assert (
        data.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)
        == gold_dict
    )

    # within dictionary
    desc_dict = {
        "logs": [
            {
                "date": "2021-11-03T04:42:54.844631+00:00",
                "agent": "CXS",
                "type": "parsing",
            }
        ],
        "collection": {
            "name": "PubChem",
            "type": "Record",
            "version": "3.2.0",
            "alias": ["chemical", "Material Sciences"],
        },
    }
    RecordDescription(**desc_dict)

    desc_dict["collection"]["type"] = "Document"
    with pytest.raises(ValidationError, match="collection.type"):
        RecordDescription(**desc_dict)

    desc_dict["collection"]["type"] = "record"
    with pytest.raises(ValidationError, match="collection.type"):
        RecordDescription(**desc_dict)


def test_strict_date_time():
    """Validate data with StrictDateTime model."""

    class Model(BaseModel):
        published: StrictDateTime

    # allowed formats
    Model(published=datetime.now(tz=timezone.utc))

    data = Model(published="2022-12-01T03:49:20.724435+00:00")
    assert data.published.isoformat() == "2022-12-01T03:49:20.724435+00:00"

    data = Model(published="2022-12-01T03:49:20.724435+03:00")
    assert data.published.isoformat() == "2022-12-01T03:49:20.724435+03:00"

    data = Model(published="2022-12-01T03:49:20.724435Z")
    assert data.published.isoformat() == "2022-12-01T03:49:20.724435+00:00"

    data = Model(published="2022-12-01T03:49:20")
    assert data.published.isoformat() == "2022-12-01T03:49:20"

    data = Model(published="2022-12-01")
    assert data.published.isoformat() == "2022-12-01T00:00:00"

    # invalid formats
    with pytest.raises(ValidationError, match="published"):
        Model(published="03:49:20")

    with pytest.raises(ValidationError, match="published"):
        Model(published=1679616000.0)
