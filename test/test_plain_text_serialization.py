"""Test plain-text serialization."""

from pathlib import Path

from docling_core.transforms.serializer.plain_text import (
    PlainTextDocSerializer,
    PlainTextParams,
)
from docling_core.types.doc.document import DoclingDocument

from .test_serialization import verify


def test_plain_text_constructed(sample_doc):
    """Serializing the constructed sample doc produces clean plain text."""
    exp_file = Path("./test/data/doc/constructed.gt.txt")
    ser = PlainTextDocSerializer(doc=sample_doc, params=PlainTextParams())
    verify(exp_file=exp_file, actual=ser.serialize().text)


def test_plain_text_export_method(sample_doc):
    """export_to_text() is consistent with PlainTextDocSerializer."""
    ser = PlainTextDocSerializer(doc=sample_doc, params=PlainTextParams())
    assert sample_doc.export_to_text() == ser.serialize().text


def test_plain_text_no_heading_markers(sample_doc):
    """Heading and title items must not contain '#' markers."""
    result = sample_doc.export_to_text()
    for line in result.splitlines():
        assert not line.startswith("#"), f"Unexpected heading marker in: {line!r}"


def test_plain_text_no_inline_markers(sample_doc):
    """Bold, italic, and strikethrough markers must be absent."""
    result = sample_doc.export_to_text()
    assert "**" not in result
    assert "~~" not in result
    # single '*' could appear in list markers, so check for italic wrapping pattern
    import re
    assert not re.search(r"\*[^*\n]+\*", result), "Unexpected italic markers found"


def test_plain_text_no_hyperlink_syntax(sample_doc):
    """Hyperlink markdown syntax must not appear; only the label is kept."""
    result = sample_doc.export_to_text()
    assert "](" not in result


def test_plain_text_deprecated_delim(sample_doc, caplog):
    """Passing a custom delim emits a deprecation warning and is ignored."""
    import logging

    with caplog.at_level(logging.WARNING):
        sample_doc.export_to_text(delim="\n")
    assert "delim" in caplog.text


def test_plain_text_from_json():
    """Serializing a real document from JSON produces clean plain text."""
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)
    exp_file = src.parent / f"{src.stem}.gt.txt"
    ser = PlainTextDocSerializer(doc=doc, params=PlainTextParams())
    verify(exp_file=exp_file, actual=ser.serialize().text)
