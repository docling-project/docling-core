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
    exp_file = Path("./tests/data/doc/constructed.gt.txt")
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
    src = Path("./tests/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)
    exp_file = src.parent / f"{src.stem}.gt.txt"
    ser = PlainTextDocSerializer(doc=doc, params=PlainTextParams())
    verify(exp_file=exp_file, actual=ser.serialize().text)


def test_plain_text_keeps_a_pipe_in_a_table_cell():
    """A pipe in a table cell is text, not the `&#124;` Markdown escape (#1076)."""
    from docling_core.types.doc import DocItemLabel, TableCell, TableData

    doc = DoclingDocument(name="x")
    doc.add_text(label=DocItemLabel.TEXT, text="a | b")
    rows = [("Range", "Note"), ("0|1", "x<y & z")]
    doc.add_table(
        data=TableData(
            num_rows=2,
            num_cols=2,
            table_cells=[
                TableCell(
                    text=text,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=row_idx == 0,
                )
                for row_idx, row in enumerate(rows)
                for col_idx, text in enumerate(row)
            ],
        )
    )

    text = doc.export_to_text()

    assert "0|1" in text
    assert "&#124;" not in text
    # Markdown keeps the escape, for the table syntax.
    assert "&#124;" in doc.export_to_markdown()
