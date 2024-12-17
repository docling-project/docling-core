#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module data_types.ccs."""
import glob
import json
from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError

from docling_core.types.base import (
    CollectionNameTypeT,
    DescriptionAnalyticsT,
    IdentifierTypeT,
    LanguageT,
)
from docling_core.types.legacy_doc.document import (
    CCSDocument,
    CCSDocumentDescription,
    Publication,
)


def test_ccs_document():
    """Validate data with CCSDocument schema."""
    for filename in glob.glob("test/data/legacy_doc/doc-*.json"):
        with open(filename, encoding="utf-8") as file_obj:
            file_json = file_obj.read()
        try:
            # do not pass strict=True, since date input values are not an instance of datetime.
            CCSDocument.model_validate_json(file_json)
            # try as well as dictionary
            doc = json.loads(file_json)
            CCSDocument.model_validate(doc)
        except ValidationError as e:
            print(f"Validation error in file {filename}:\n{e.json()}")
            raise

    # check doc-error-1 is invalid in logs
    try:
        with open("test/data/legacy_doc/error-1.json", encoding="utf-8") as file_obj:
            file_json = file_obj.read()
        CCSDocument.model_validate_json(file_json)
        assert False, f"Data in file {filename} should be invalid for CCSDocument model"
    except ValidationError as e:
        for error in e.errors():
            print(type(error))
            assert all(
                item in error["loc"] for item in ("description", "logs")
            ), f"Data in file {filename} should fail in logs"

    # check doc-error-2 is invalid for missing page-hashes
    with (
        pytest.raises(ValidationError, match="page-hashes"),
        open("test/data/legacy_doc/error-2.json", encoding="utf-8") as file_obj,
    ):
        file_json = file_obj.read()
        CCSDocument.model_validate_json(file_json)

    # check doc-error-3 is invalid for wrong types in citation_count and reference_count
    with (
        pytest.raises(ValidationError, match="count"),
        open("test/data/legacy_doc/error-3.json", encoding="utf-8") as file_obj,
    ):
        file_json = file_obj.read()
        CCSDocument.model_validate_json(file_json)


def test_publication_journal():
    """ "Validate data with Publication model."""
    for filename in glob.glob("test/data/legacy_doc/intermediates/publication_*.json"):
        with open(filename, encoding="utf-8") as file_obj:
            file_json = file_obj.read()
            file_dict = json.loads(file_json)
        try:
            Publication.model_validate_json(file_json)
            Publication.model_validate(file_dict)
        except ValidationError as e:
            assert False, f"Validation error in file {filename}:\n{e.json()}"


def test_description_advanced_t():
    """Validate data with different DescriptionAdvancedT instances."""
    # without description.advanced
    with open("test/data/legacy_doc/doc-5.json", encoding="utf-8") as file_obj:
        desc = json.load(file_obj)["description"]

    # without advanced
    CCSDocumentDescription.model_validate(desc)

    # any dictionary is valid, since it is not parametrized
    CCSDocumentDescription(**desc, advanced={"serial": "CXS12345"})
    CCSDocumentDescription(**desc, advanced={0: "CXS12345"})
    with pytest.raises(
        ValidationError, match="should be a valid dictionary or instance of BaseModel"
    ):
        CCSDocumentDescription(**desc, advanced=False)

    class MyAdvanced(BaseModel):
        serial: str
        comment: Optional[str] = None

    # with a model and bound specification
    adv_inst = MyAdvanced(serial="CXS12345", comment="public document")
    CCSDocumentDescription(**desc, advanced=adv_inst)
    with pytest.raises(ValidationError, match="Field required"):
        CCSDocumentDescription(**desc, advanced=MyAdvanced(comment="public document"))

    # with a model and generic type specification
    advanced = MyAdvanced(serial="CXS12345", comment="public document")
    CCSDocumentDescription[
        MyAdvanced,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ](**desc)
    CCSDocumentDescription[
        MyAdvanced,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ](**desc, advanced=adv_inst)
    with pytest.raises(ValidationError, match="Field required"):
        CCSDocumentDescription[
            MyAdvanced,
            DescriptionAnalyticsT,
            IdentifierTypeT,
            LanguageT,
            CollectionNameTypeT,
        ](**desc, advanced={})

    # deriving a new type
    MyDocument = CCSDocumentDescription[
        MyAdvanced,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ]
    MyDocument.model_validate(desc)
    desc["advanced"] = advanced
    MyDocument.model_validate(desc)
