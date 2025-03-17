#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module types."""
import glob

import pytest
from pydantic import ValidationError

from docling_core.types import Generic, Record
from docling_core.types.legacy_doc.document import ExportedCCSDocument as Document

GENERATE = False


def test_generic():
    """Test the Generic model."""
    input_generic_0 = {
        "file-info": {
            "filename": "abc.xml",
            "filename-prov": "abc.xml.zip",
            "document-hash": "123457889",
        },
        "_name": "The ABC legacy_doc",
        "custom": ["The custom ABC content 1.", "The custom ABC content 2."],
    }
    Generic.model_validate(input_generic_0)

    input_generic_1 = {
        "file-info": {"filename": "abc.xml", "document-hash": "123457889"},
        "_name": "The ABC legacy_doc",
    }
    Generic.model_validate(input_generic_1)

    input_generic_2 = {
        "_name": "The ABC legacy_doc",
        "custom": ["The custom ABC content 1.", "The custom ABC content 2."],
    }
    with pytest.raises(ValidationError):
        Generic.model_validate(input_generic_2)


def test_document():
    """Test the Document model."""
    for filename in glob.glob("test/data/legacy_doc/doc-*.json"):
        with open(filename, encoding="utf-8") as file_obj:
            file_json = file_obj.read()
        Document.model_validate_json(file_json)


def test_table_export_to_tokens():
    """Test the Table Tokens export."""

    for filename in glob.glob("test/data/legacy_doc/doc-*.json"):
        with open(filename, encoding="utf-8") as file_obj:
            file_json = file_obj.read()

        doc = Document.model_validate_json(file_json)

        if doc.tables is not None and doc.page_dimensions is not None:

            pagedims = doc.get_map_to_page_dimensions()

            if doc.tables is not None:
                for i, table in enumerate(doc.tables):
                    page = table.prov[0].page
                    out = table.export_to_document_tokens(
                        page_w=pagedims[page][0], page_h=pagedims[page][1]
                    )

                    fname = f"{filename}_table_{i}.dt.txt"
                    if GENERATE:
                        print(f"writing {fname}")
                        with open(fname, "w", encoding="utf-8") as gold_obj:
                            gold_obj.write(out)

                    with open(fname, "r", encoding="utf-8") as gold_obj:
                        gold_data = gold_obj.read()

                    assert out == gold_data

                    # we only test on the first table
                    break

        elif doc.tables is not None and doc.page_dimensions is None:

            if doc.tables is not None:
                for i, table in enumerate(doc.tables):
                    page = table.prov[0].page
                    out = table.export_to_document_tokens(
                        add_table_location=False, add_cell_location=False
                    )

                    fname = f"{filename}_table_{i}.dt.txt"
                    if GENERATE:
                        print(f"writing {fname}")
                        with open(fname, "w", encoding="utf-8") as gold_obj:
                            gold_obj.write(out)

                    with open(fname, "r", encoding="utf-8") as gold_obj:
                        gold_data = gold_obj.read()

                    assert out == gold_data

                    # we only test on the first table
                    break


def test_document_export_to_md():
    """Test the Document Markdown export."""
    with open("test/data/legacy_doc/doc-export.json", encoding="utf-8") as src_obj:
        src_data = src_obj.read()
    doc = Document.model_validate_json(src_data)

    md = doc.export_to_markdown()

    if GENERATE:
        with open(
            "test/data/legacy_doc/doc-export.md", "w", encoding="utf-8"
        ) as gold_obj:
            gold_obj.write(md)

    with open("test/data/legacy_doc/doc-export.md", encoding="utf-8") as gold_obj:
        gold_data = gold_obj.read().strip()

    assert md == gold_data


def test_document_export_to_tokens():
    """Test the Document Tokens export."""
    with open("test/data/legacy_doc/doc-export.json", encoding="utf-8") as src_obj:
        src_data = src_obj.read()

    doc = Document.model_validate_json(src_data)
    xml = doc.export_to_document_tokens(delim=True)

    if GENERATE:
        with open(
            "test/data/legacy_doc/doc-export.dt.txt", "w", encoding="utf-8"
        ) as gold_obj:
            gold_obj.write(xml)

    with open(
        "test/data/legacy_doc/doc-export.dt.txt", "r", encoding="utf-8"
    ) as gold_obj:
        gold_data = gold_obj.read().strip()

    assert xml == gold_data


def test_record():
    """Test the Document model."""
    for filename in glob.glob("test/data/rec/record-*.json"):
        with open(filename, encoding="utf-8") as file_obj:
            file_json = file_obj.read()
        Record.model_validate_json(file_json)
