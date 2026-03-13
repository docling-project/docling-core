import json
from pathlib import Path

from docling_core.experimental.serializer.outline import (
    OutlineDocSerializer,
    OutlineFormat,
    OutlineMode,
    OutlineParams,
)
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel


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

    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Unexpected TOC serialization "

    # with heading hierachy
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    exp_path = doc_path.with_suffix(".toc.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()
    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Unexpected TOC serialization "


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
    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Unexpected outline serialization "


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
    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Unexpected outline serialization "

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

    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Serialized text should match expected output"


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

    # Check first item structure
    first_item = data[0]
    assert isinstance(first_item, dict)
    assert first_item.keys() == {"ref", "title", "summary", "level"}
    assert first_item["ref"].startswith("#/texts/")
    assert isinstance(first_item["level"], int)

    # When include_non_meta=True, titles should be present
    # (at least for some items that have text)
    has_title = any("title" in item for item in data)
    assert has_title, "At least some items should have titles when include_non_meta=True"

    # All items with summaries should have them
    for item in data:
        if "summary" in item:
            assert isinstance(item["summary"], str)
            assert len(item["summary"]) > 0
    with open(exp_path) as f:
        expected = json.load(f)
    assert json.loads(result.text) == expected, "Serialized text should match expected output"

    # Hierarchical document with extra fields
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    doc = DoclingDocument.load_from_json(filename=doc_path)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()
    data = json.loads(result.text)
    first_item = data[0]
    assert first_item.keys() == {"ref", "title", "summary", "level", "mellea__original_char_count"}
    assert first_item["mellea__original_char_count"] == 809
    assert isinstance(first_item["level"], int)


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
    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Serialized ITXT should match expected output"

    # Hierarchical document with extra fields
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    exp_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.toc.gt.itxt")

    doc = DoclingDocument.load_from_json(filename=doc_path)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    lines = result.text.split("\n")
    assert len(lines) > 0

    # Check that we have indented lines (level 2+)
    has_indented = any(line.startswith("   ") for line in lines)
    assert has_indented, "Should have indented lines for hierarchical structure"

    # Verify against ground truth file
    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Hierarchical ITXT should match expected output"


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
