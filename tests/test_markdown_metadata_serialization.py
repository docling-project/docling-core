"""Test the optional document-level Markdown metadata hook."""

from pathlib import Path

import pytest
import yaml

from docling_core.transforms.serializer.base import BaseMetadataSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    YAMLMetadataSerializer,
)
from docling_core.types.doc import DocItemLabel, DoclingDocument

from .test_publication_serialization import _book_document


def test_frontmatter_is_disabled_by_default():
    doc = _book_document()
    expected = Path("tests/data/serialization/publication.md").read_bytes().split(b"---\n", maxsplit=2)[2][1:]
    implicit = MarkdownDocSerializer(doc=doc).serialize()
    explicit = MarkdownDocSerializer(doc=doc, metadata_serializer=None).serialize()
    assert implicit.text.encode("utf-8") == expected
    assert explicit == implicit
    assert doc.export_to_markdown().encode("utf-8") == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "title": "Dîner: 'Oxford' # 1\n---\nnext line",
            "authors": ["Alice", "Bob"],
            "published": "2026-08-24",
            "custom": {"enabled": True, "count": 3, "missing": None},
        },
    ],
)
def test_arbitrary_yaml_metadata_roundtrips(metadata):
    doc = _book_document()
    plain = MarkdownDocSerializer(doc=doc).serialize()
    serializer = MarkdownDocSerializer(doc=doc, metadata_serializer=YAMLMetadataSerializer(metadata=metadata))
    result = serializer.serialize()
    frontmatter, body = result.text[4:].split("\n---\n\n", maxsplit=1)
    assert yaml.safe_load(frontmatter) == metadata
    assert body.encode("utf-8") == plain.text.encode("utf-8")
    assert result.spans == plain.spans
    assert serializer.serialize() == result


@pytest.mark.parametrize("page_break", [None, "", "<PB>\n"])
def test_custom_hook_gets_finalized_parts_and_context(page_break):
    """Plugins receive finalized text without mutating the source results."""
    doc = DoclingDocument(name="custom")
    marker = "#_#_DOCLING_DOC_PAGE_BREAK_1_2_#_#"
    original_parts = [SerializationResult(text=f"before{marker}after"), SerializationResult()]
    params = MarkdownParams(page_break_placeholder=page_break)
    calls = []

    class CustomMetadataSerializer(BaseMetadataSerializer):
        def serialize(self, *, doc, doc_serializer, parts, **kwargs):
            calls.append((doc, doc_serializer, parts, kwargs))
            return create_ser_result(text=f"---\nname: {doc.name}\n---\n\n")

    serializer = MarkdownDocSerializer(doc=doc, metadata_serializer=CustomMetadataSerializer(), params=params)
    result = serializer.serialize_doc(parts=original_parts, custom_option=42)
    expected_body = f"before{marker if page_break is None else page_break}after"
    assert result.text == f"---\nname: custom\n---\n\n{expected_body}"
    assert len(calls) == 1
    passed_doc, passed_serializer, passed_parts, kwargs = calls[0]
    assert passed_doc is doc
    assert passed_serializer is serializer
    assert passed_parts[0].text == expected_body
    assert kwargs["custom_option"] == 42
    assert original_parts[0].text == f"before{marker}after"


def test_metadata_only_document_and_item_serialization():
    doc = DoclingDocument(name="empty")
    serializer = MarkdownDocSerializer(doc=doc, metadata_serializer=YAMLMetadataSerializer())
    assert serializer.serialize().text == "---\n{}\n---\n\n"
    item = doc.add_text(label=DocItemLabel.TEXT, text="Just the item")
    assert serializer.serialize(item=item).text == "Just the item"


def test_hook_can_return_no_frontmatter():
    class EmptyMetadataSerializer(BaseMetadataSerializer):
        def serialize(self, **kwargs):
            return SerializationResult()

    doc = _book_document()
    assert MarkdownDocSerializer(doc=doc, metadata_serializer=EmptyMetadataSerializer()).serialize() == (
        MarkdownDocSerializer(doc=doc).serialize()
    )
