"""Tests for MarkdownSummarySerializer (document outline)."""

from pathlib import Path

import pytest

from docling_core.transforms.serializer.markdown_summary import (
    MarkdownSummaryParams,
    MarkdownSummarySerializer,
)

from .test_docling_doc import _construct_doc


@pytest.mark.parametrize("use_md_headers", [False, True])
def test_markdown_summary_outline(use_md_headers: bool):
    # Build a representative document with title, headers, text, lists, table, and pictures
    doc = _construct_doc()

    ser = MarkdownSummarySerializer(
        doc=doc,
        params=MarkdownSummaryParams(use_markdown_headers=use_md_headers),
    )

    outline = ser.serialize().text

    print(outline)
    
    # Leading list items should not appear in the outline
    assert "item of leading list" not in outline

    # Captions should be excluded from outline
    assert "This is the caption of table 1." not in outline
    assert "This is the caption of figure 1." not in outline
    assert "This is the caption of figure 2." not in outline

    # Title and section header formatting based on params
    if use_md_headers:
        # Markdown-style headers
        assert "# Title of the Document" in outline
        assert "## 1. Introduction" in outline
        # Ensure we don't get the verbose label style when using MD headers
        assert "title (reference=" not in outline.splitlines()[0]
    else:
        # Verbose outline lines with references
        first_line = outline.splitlines()[0]
        assert first_line.startswith("title (reference=") and first_line.endswith(
            "): Title of the Document"
        )
        # Section header line contains level and reference
        assert "section-header (level=1, reference=" in outline

    # Tables and pictures should be numbered and listed with references
    assert "table 1 (reference=" in outline
    assert "picture 1 (reference=" in outline

