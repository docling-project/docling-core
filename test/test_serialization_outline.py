import json
from pathlib import Path

from docling_core.experimental.serializer.outline import (
    OutlineDocSerializer,
    OutlineFormat,
    OutlineItemData,
    OutlineMode,
    OutlineParams,
    _format_indented_text_line,
)
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from .test_utils import assert_or_generate_ground_truth


def test_outline_serializer_mode_toc():
    """Test TABLE_OF_CONTENTS mode only includes titles and section headers."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".toc.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    params = OutlineParams(include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert "\\[ref=#/texts/" in result.text

    reference_count = result.text.count("\\[ref=")
    assert reference_count > 0, "TABLE_OF_CONTENTS mode should include section headers"

    assert_or_generate_ground_truth(result.text, exp_path, "Unexpected TOC serialization")

    # with heading hierachy
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    exp_path = doc_path.with_suffix(".toc.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert_or_generate_ground_truth(result.text, exp_path, "Unexpected TOC serialization")


def test_outline_serializer_mode_toc_custom():
    """Test TABLE_OF_CONTENTS mode with custom item labels."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".custom.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    # params = OutlineParams(include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS)
    params = OutlineParams(include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS, labels={DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER, DocItemLabel.TABLE})
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0

    assert_or_generate_ground_truth(result.text, exp_path, "Unexpected outline serialization")


def test_outline_serializer_mode_outline():
    """Test OUTLINE mode includes all document items."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".outline.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    params = OutlineParams(include_non_meta=True, mode=OutlineMode.OUTLINE)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0
    reference_count_outline = result.text.count("\\[ref=")

    assert_or_generate_ground_truth(result.text, exp_path, "Unexpected outline serialization")

    # Compare with TABLE_OF_CONTENTS mode
    params_toc = OutlineParams(include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS)
    ser_toc = OutlineDocSerializer(doc=doc, params=params_toc)
    result_toc = ser_toc.serialize()
    reference_count_toc = result_toc.text.count("\\[ref=")
    assert reference_count_outline > reference_count_toc, (
        f"OUTLINE mode should include more items ({reference_count_outline}) "
        f"than TABLE_OF_CONTENTS mode ({reference_count_toc})"
    )


def test_outline_serializer_include_non_meta_false():
    """Test that include_non_meta=False still outputs structure and summaries.

    When include_non_meta=False, the outline should still show:
    - References (document structure)
    - Summaries (metadata)
    But exclude the actual text content (prepend).
    """
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".mtoc.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    params = OutlineParams(include_non_meta=False, mode=OutlineMode.TABLE_OF_CONTENTS)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0, (
        "Outline should show structure (references) and metadata (summaries) "
        "even when include_non_meta=False"
    )
    assert "\\[ref=" in result.text, "Should include references"

    assert_or_generate_ground_truth(result.text, exp_path)


def test_outline_serializer_empty_document():
    """Test serializer handles documents without relevant items gracefully."""
    # Create a minimal document
    doc = DoclingDocument(name="test_doc")

    params = OutlineParams(
        include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS
    )
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    # Should return empty or minimal result, not crash
    assert isinstance(result.text, str)


def test_outline_serializer_json_format():
    """Test JSON format output for TABLE_OF_CONTENTS mode."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".mtoc.gt.json")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    # Test with include_non_meta=True (includes titles)
    params = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON
    )
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0

    # Parse the JSON to verify structure
    data = json.loads(result.text)
    assert isinstance(data, list)
    assert len(data) > 0

    # Check first item structure (should be document-level metadata)
    first_item = data[0]
    assert isinstance(first_item, dict)
    assert first_item["ref"] == "#/body", "First item should be document-level metadata"
    assert first_item.keys() == {"ref", "item", "title", "summary", "level"}
    assert isinstance(first_item["level"], int)

    # Check second item (first text item)
    if len(data) > 1:
        second_item = data[1]
        assert second_item["ref"].startswith("#/texts/")

    # When include_non_meta=True, titles should be present
    # (at least for some items that have text)
    has_title = any("title" in item for item in data)
    assert has_title, "At least some items should have titles when include_non_meta=True"

    # All items with summaries should have them
    for item in data:
        if "summary" in item:
            assert isinstance(item["summary"], str)
            assert len(item["summary"]) > 0

    assert_or_generate_ground_truth(result.text, exp_path, is_json=True)

    # Hierarchical document with extra fields
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    exp_path_hier = doc_path.with_suffix(".mtoc.gt.json")

    doc = DoclingDocument.load_from_json(filename=doc_path)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()
    data = json.loads(result.text)
    first_item = data[0]
    # Document-level metadata should have the custom field from the document's meta
    assert first_item["ref"] == "#/body"
    assert first_item.keys() == {"ref", "item", "title", "summary", "level", "mellea__original_char_count"}
    assert first_item["mellea__original_char_count"] == 382  # Document-level summary char count
    assert isinstance(first_item["level"], int)

    assert_or_generate_ground_truth(result.text, exp_path_hier, is_json=True)

    # Outline mode with title
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path_hier = doc_path.with_suffix(".outline.gt.json")
    doc = DoclingDocument.load_from_json(filename=doc_path)
    params = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.OUTLINE,
        format=OutlineFormat.JSON
    )
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0
    data = json.loads(result.text)
    assert isinstance(data, list)
    assert len(data) > 0
    has_title = any("title" in item for item in data)
    assert has_title, "At least some items should have titles when include_non_meta=True"
    has_item = all("item" in item for item in data)
    assert has_item, "All data points should have the item field"
    has_picture = any(item["item"] == "picture" for item in data)
    assert has_picture, f"In document {doc_path.name} at least some items should be of type 'picture' in outline mode"
    has_table = any(item["item"] == "table" for item in data)
    assert has_table, f"In document {doc_path.name} at least some items should be of type 'table' in outline mode"
    has_table_summary = any(item["item"] == "table" and "summary" in item for item in data)
    assert has_table_summary, f"In document {doc_path.name} at least a table has a summary and should appear in outline mode"

    assert_or_generate_ground_truth(result.text, exp_path_hier, is_json=True)


def test_outline_serializer_json_format_without_non_meta():
    """Test JSON format output without non-meta content."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    # Test with include_non_meta=False (no titles, only refs and summaries)
    params = OutlineParams(
        include_non_meta=False,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON
    )
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0

    # Parse the JSON to verify structure
    data = json.loads(result.text)
    assert isinstance(data, list)
    assert len(data) > 0

    # Check that no titles are present when include_non_meta=False
    for item in data:
        assert "ref" in item
        assert "title" not in item, "Titles should not be present when include_non_meta=False"
        # Summaries should still be present
        if "summary" in item:
            assert isinstance(item["summary"], str)


def test_outline_serializer_itxt_format():
    """Test ITXT format output for TABLE_OF_CONTENTS mode."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".mtoc.gt.itxt")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    # Test with include_non_meta=True (includes titles)
    params = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.ITXT
    )
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0

    # Verify structure - should be indented text lines
    lines = result.text.split("\n")
    assert len(lines) > 0

    # Check first line structure (level 1, no indentation)
    first_line = lines[0]
    assert first_line.startswith("[ref=")
    assert "[" in first_line and "]" in first_line

    # Verify against ground truth file
    assert_or_generate_ground_truth(result.text, exp_path, "Serialized ITXT should match expected output")

    # Hierarchical document with extra fields
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    exp_path_hier = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.toc.gt.itxt")

    doc = DoclingDocument.load_from_json(filename=doc_path)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    lines = result.text.split("\n")
    assert len(lines) > 0

    # Check that we have indented lines (level 2+)
    has_indented = any(line.startswith("   ") for line in lines)
    assert has_indented, "Should have indented lines for hierarchical structure"

    # Verify against ground truth file
    assert_or_generate_ground_truth(result.text, exp_path_hier, "Hierarchical ITXT should match expected output")


def test_outline_serializer_itxt_format_without_non_meta():
    """Test ITXT format output without non-meta content."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    # Test with include_non_meta=False (no titles, only refs and summaries)
    params = OutlineParams(
        include_non_meta=False,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.ITXT
    )
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0

    # Verify structure - should be indented text lines
    lines = result.text.split("\n")
    assert len(lines) > 0

    # Check that no titles are present when include_non_meta=False
    # (titles appear in brackets like [Title Text])
    for line in lines:
        if line.strip():  # Skip empty lines
            assert "[ref=" in line, "Each line should have a ref"
            # Count brackets - should have ref brackets but no title brackets when include_non_meta=False
            # Format without title: [ref=...] summary
            # Format with title: [ref=...] [Title] summary
            bracket_count = line.count("[")
            assert bracket_count == 1, f"Should only have ref bracket when include_non_meta=False, got {bracket_count} in: {line}"


def test_format_indented_text_line():
    """Test _format_indented_text_line function with various inputs."""

    # Test with short summary (should not be truncated)
    item_short = OutlineItemData(
        ref="#/texts/0",
        item="section_header",
        title="Introduction",
        summary="This is a short summary.",
        level=1
    )
    result = _format_indented_text_line(item_short, indent_size=2, max_summary_length=100)
    assert result == "  [ref=#/texts/0] [Introduction] This is a short summary."
    assert "..." not in result, "Short summary should not be truncated"

    # Test with long summary (should be truncated)
    long_summary = "A" * 150  # 150 characters
    item_long = OutlineItemData(
        ref="#/texts/1",
        item="section_header",
        title="Long Section",
        summary=long_summary,
        level=2
    )
    result = _format_indented_text_line(item_long, indent_size=2, max_summary_length=50)
    assert result.startswith("    [ref=#/texts/1] [Long Section] ")
    assert result.endswith("...")
    assert len(result.split("] ")[-1]) == 50, "Truncated summary should be exactly max_summary_length"

    # Test without title
    item_no_title = OutlineItemData(
        ref="#/texts/2",
        item="paragraph",
        summary="Summary without title",
        level=0
    )
    result = _format_indented_text_line(item_no_title, indent_size=2, max_summary_length=100)
    assert result == "[ref=#/texts/2] Summary without title"
    assert "[" not in result.split("] ", 1)[1], "Should not have title brackets"

    # Test without summary
    item_no_summary = OutlineItemData(
        ref="#/texts/3",
        item="title",
        title="Title Only",
        level=1
    )
    result = _format_indented_text_line(item_no_summary, indent_size=2, max_summary_length=100)
    assert result == "  [ref=#/texts/3] [Title Only]"

    # Test with different indent sizes
    item_indent = OutlineItemData(
        ref="#/texts/4",
        item="section_header",
        title="Nested",
        summary="Nested content",
        level=3
    )
    result = _format_indented_text_line(item_indent, indent_size=3, max_summary_length=100)
    assert result.startswith(" " * 9)  # 3 spaces * level 3
    assert "[ref=#/texts/4] [Nested] Nested content" in result
