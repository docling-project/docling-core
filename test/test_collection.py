#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module types."""
import glob

import pytest
from pydantic import ValidationError

from docling_core.types import Document, Generic, Record


def test_generic():
    """Test the Generic model."""
    input_generic_0 = {
        "file-info": {
            "filename": "abc.xml",
            "filename-prov": "abc.xml.zip",
            "document-hash": "123457889",
        },
        "_name": "The ABC doc",
        "custom": ["The custom ABC content 1.", "The custom ABC content 2."],
    }
    Generic.model_validate(input_generic_0)

    input_generic_1 = {
        "file-info": {"filename": "abc.xml", "document-hash": "123457889"},
        "_name": "The ABC doc",
    }
    Generic.model_validate(input_generic_1)

    input_generic_2 = {
        "_name": "The ABC doc",
        "custom": ["The custom ABC content 1.", "The custom ABC content 2."],
    }
    with pytest.raises(ValidationError):
        Generic.model_validate(input_generic_2)


def test_document():
    """Test the Document model."""
    for filename in glob.glob("test/data/doc/doc-*.json"):
        with open(filename) as file_obj:
            file_json = file_obj.read()
        Document.model_validate_json(file_json)


def test_document_md_export():
    """Test the Document Markdown export."""
    with open("test/data/doc/md-export.json") as src_obj:
        src_data = src_obj.read()
    doc = Document.model_validate_json(src_data)
    md = doc.export_to_markdown()

    with open("test/data/doc/md-export.md") as gold_obj:
        gold_data = gold_obj.read().strip()

    assert md == gold_data


def test_record():
    """Test the Document model."""
    for filename in glob.glob("test/data/rec/record-*.json"):
        with open(filename) as file_obj:
            file_json = file_obj.read()
        Record.model_validate_json(file_json)
