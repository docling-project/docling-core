import json
from pathlib import Path

import pytest

from docling_core.experimental.serializer.outline import (
    OutlineDocSerializer,
    OutlineFormat,
    OutlineItemData,
    OutlineMode,
    OutlineParams,
    _format_indented_text_line,
)
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import RefItem
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
    for line in lines:
        if line.strip():
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

@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_outline_serialization_from_item():
    """Test the outline serialization starting from different node item."""

    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    doc = DoclingDocument.load_from_json(filename=doc_path)
    params = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON
    )

    ser = OutlineDocSerializer(doc=doc, params=params)
    doc_out = ser.serialize()
    root_data = json.loads(doc_out.text)

    # Test 1: Serialize from a nested item using start_item parameter
    nested_item = RefItem(cref="#/texts/25").resolve(doc)
    params_with_start = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON,
        start_item=nested_item
    )
    ser_nested = OutlineDocSerializer(doc=doc, params=params_with_start)
    nested_out = ser_nested.serialize()

    # nested_out should be a properly formatted JSON array
    assert nested_out.text, "Nested serialization should produce output"
    nested_data = json.loads(nested_out.text)
    assert isinstance(nested_data, list), "Nested output should be a JSON array"

    # The nested output should be a subset of the root output
    assert len(nested_data) <= len(root_data), "Nested outline should have fewer or equal items than root"
    assert len(nested_data) > 0, "Nested outline should have at least one item"

    # Check that the nested item is included
    nested_ref = nested_item.self_ref
    assert any(item_data["ref"] == nested_ref for item_data in nested_data), \
        f"Nested item {nested_ref} should be in the output"

    # Verify that all items in nested_data are either the nested_item or its descendants
    # Expected: #/texts/25 (level 2) and its 7 children (all level 3)
    assert len(nested_data) == 8, f"Expected 8 items (1 parent + 7 children), got {len(nested_data)}"

    # First item should be the starting item
    assert nested_data[0]["ref"] == "#/texts/25"
    assert nested_data[0]["level"] == 2

    # All other items should be level 3 (children of level 2)
    for item_data in nested_data[1:]:
        assert item_data["level"] == 3, f"Child item {item_data['ref']} should be level 3"

    # Test 2: Serialize with maximum level
    params_with_max_level = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON,
        max_level=2
    )
    ser_max_level = OutlineDocSerializer(doc=doc, params=params_with_max_level)
    max_level_out = ser_max_level.serialize()
    max_level_data = json.loads(max_level_out.text)

    # The max_level output should be a subset of the root output
    assert len(max_level_data) <= len(root_data), "Max level outline should have fewer or equal items than root"

    # Check that all section headers have level <= 2
    for item_data in max_level_data:
        if "level" in item_data and item_data["item"] == "section_header":
            assert item_data["level"] <= 2, f"Section header level {item_data['level']} should be <= 2"

    # Test 3: Combine start_item and max_level
    # When start_item is level 2 and max_level is 2, we should get the item and its level-3 children excluded
    params_combined = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON,
        start_item=nested_item,
        max_level=2
    )
    ser_combined = OutlineDocSerializer(doc=doc, params=params_combined)
    combined_out = ser_combined.serialize()
    combined_data = json.loads(combined_out.text)
    assert isinstance(combined_data, list), "Combined output should be a JSON array"

    # Should only include the nested item itself (level 2), not its level-3 children
    assert len(combined_data) == 1, f"Expected 1 item (only level 2, no level 3 children), got {len(combined_data)}"
    assert combined_data[0]["ref"] == "#/texts/25"
    assert combined_data[0]["level"] == 2

    # Test 4: Test with OUTLINE mode (includes all items, not just headings)
    params_outline_max = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.OUTLINE,
        format=OutlineFormat.JSON,
        max_level=2
    )
    ser_outline_max = OutlineDocSerializer(doc=doc, params=params_outline_max)
    outline_max_out = ser_outline_max.serialize()
    outline_max_data = json.loads(outline_max_out.text)

    # In OUTLINE mode with max_level, we should still get non-heading items
    # but only those that are children of headings with level <= max_level
    has_non_heading = any(item["item"] not in ["section_header", "title"] for item in outline_max_data)
    assert has_non_heading, "OUTLINE mode should include non-heading items"

    # Test 5: Test with Markdown format
    params_md_max = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.MARKDOWN,
        max_level=2
    )
    ser_md_max = OutlineDocSerializer(doc=doc, params=params_md_max)
    md_max_out = ser_md_max.serialize()
    assert isinstance(md_max_out.text, str)
    assert "# 2408.09869v5\n\\[ref=#/body\\]" in md_max_out.text
    assert "## Docling Technical Report\n\\[ref=#/texts/1\\]" in md_max_out.text
    assert "### 4 Performance\n\\[ref=#/texts/66\\]" in md_max_out.text
    assert "### References\n\\[ref=#/texts/78\\]" in md_max_out.text
    assert "#### OCR\n\\[ref=#/texts/58\\]" not in md_max_out.text

    params_md_start = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.MARKDOWN,
        start_item=nested_item
    )
    ser_md_start = OutlineDocSerializer(doc=doc, params=params_md_start)
    md_start_out = ser_md_start.serialize()
    assert md_start_out.text.startswith("### 3 Processing pipeline\n\\[ref=#/texts/25\\]")
    # check parents are not included
    assert "\\[ref=#/texts/19\\]" not in md_start_out.text
    assert "\\[ref=#/texts/1\\]" not in md_start_out.text
    assert "\\[ref=#/body\\]" not in md_start_out.text
    # check siblings are not included
    assert "\\[ref=#/texts/19\\]" not in md_start_out.text
    assert "\\[ref=#/texts/66\\]" not in md_start_out.text

    # Test 6: Test with ITXT format
    params_itxt_max = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.ITXT,
        max_level=2
    )
    ser_itxt_max = OutlineDocSerializer(doc=doc, params=params_itxt_max)
    itxt_max_out = ser_itxt_max.serialize()
    assert isinstance(itxt_max_out.text, str)
    assert "[ref=#/body]" in itxt_max_out.text
    assert "[ref=#/texts/1]" in itxt_max_out.text
    assert "[ref=#/texts/66]" in itxt_max_out.text
    assert "[ref=#/texts/78]" in itxt_max_out.text
    assert "[ref=#/texts/58]" not in itxt_max_out.text

    params_itxt_start = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.ITXT,
        start_item=nested_item
    )
    ser_itxt_start = OutlineDocSerializer(doc=doc, params=params_itxt_start)
    itxt_start_out = ser_itxt_start.serialize()
    assert isinstance(itxt_start_out.text, str)
    assert len(itxt_start_out.text) > 0
    # check parents are not included
    assert "[ref=#/texts/19]" not in itxt_start_out.text
    assert "[ref=#/texts/1]" not in itxt_start_out.text
    assert "[ref=#/body]" not in itxt_start_out.text
    # check siblings are not included
    assert "[ref=#/texts/19]" not in itxt_start_out.text
    assert "[ref=#/texts/66]" not in itxt_start_out.text

    # Check that indentation is normalized (first line should have no leading spaces)
    itxt_lines = itxt_start_out.text.split("\n")
    assert len(itxt_lines) > 0, "ITXT output should have at least one line"
    first_line = itxt_lines[0]
    assert not first_line.startswith(" "), f"First line should have no leading spaces, got: '{first_line}'"
    assert first_line.startswith("[ref=#/texts/25]"), "First line should be the start_item"



@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_outline_serialization_spans():
    """Test that serialization results preserve spans for get_unique_doc_items()."""
    doc_path = Path("test/data/doc/2408.09869v5_hierarchical_enriched_summary.json")
    doc = DoclingDocument.load_from_json(filename=doc_path)
    
    # Get a nested item to use as start_item
    nested_item = RefItem(cref="#/texts/25").resolve(doc)
    
    # Test 1: JSON format with start_item
    params_json = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON,
        start_item=nested_item
    )
    ser_json = OutlineDocSerializer(doc=doc, params=params_json)
    result_json = ser_json.serialize()
    
    # Verify spans are present
    assert len(result_json.spans) > 0, "JSON format should have spans"
    
    # Get unique doc items
    doc_items_json = result_json.get_unique_doc_items()
    assert len(doc_items_json) > 0, "Should have doc items from spans"
    
    # Verify the items match what's in the JSON output
    json_data = json.loads(result_json.text)
    json_refs = {item["ref"] for item in json_data}
    span_refs = {item.self_ref for item in doc_items_json}
    
    # All items in JSON should have corresponding spans
    assert json_refs == span_refs, f"JSON refs {json_refs} should match span refs {span_refs}"
    
    # Test 2: Markdown format with max_level
    params_md = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.MARKDOWN,
        max_level=2
    )
    ser_md = OutlineDocSerializer(doc=doc, params=params_md)
    result_md = ser_md.serialize()
    
    # Verify spans are present
    assert len(result_md.spans) > 0, "Markdown format should have spans"
    
    # Get unique doc items
    doc_items_md = result_md.get_unique_doc_items()
    assert len(doc_items_md) > 0, "Should have doc items from spans"
    
    # Verify items are in the markdown text
    for item in doc_items_md:
        assert f"\\[ref={item.self_ref}\\]" in result_md.text, \
            f"Item {item.self_ref} should be in markdown text"
    
    # Test 3: ITXT format with start_item
    params_itxt = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.ITXT,
        start_item=nested_item
    )
    ser_itxt = OutlineDocSerializer(doc=doc, params=params_itxt)
    result_itxt = ser_itxt.serialize()
    
    # Verify spans are present
    assert len(result_itxt.spans) > 0, "ITXT format should have spans"
    
    # Get unique doc items
    doc_items_itxt = result_itxt.get_unique_doc_items()
    assert len(doc_items_itxt) > 0, "Should have doc items from spans"
    
    # Verify items are in the ITXT text
    for item in doc_items_itxt:
        assert f"[ref={item.self_ref}]" in result_itxt.text, \
            f"Item {item.self_ref} should be in ITXT text"
    
    # Test 4: Verify filtering consistency across formats
    # All three formats with the same start_item should have the same doc items
    params_json_filtered = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.JSON,
        start_item=nested_item
    )
    params_md_filtered = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.MARKDOWN,
        start_item=nested_item
    )
    params_itxt_filtered = OutlineParams(
        include_non_meta=True,
        mode=OutlineMode.TABLE_OF_CONTENTS,
        format=OutlineFormat.ITXT,
        start_item=nested_item
    )
    
    ser_json_f = OutlineDocSerializer(doc=doc, params=params_json_filtered)
    ser_md_f = OutlineDocSerializer(doc=doc, params=params_md_filtered)
    ser_itxt_f = OutlineDocSerializer(doc=doc, params=params_itxt_filtered)
    
    result_json_f = ser_json_f.serialize()
    result_md_f = ser_md_f.serialize()
    result_itxt_f = ser_itxt_f.serialize()
    
    items_json_f = result_json_f.get_unique_doc_items()
    items_md_f = result_md_f.get_unique_doc_items()
    items_itxt_f = result_itxt_f.get_unique_doc_items()
    
    refs_json_f = {item.self_ref for item in items_json_f}
    refs_md_f = {item.self_ref for item in items_md_f}
    refs_itxt_f = {item.self_ref for item in items_itxt_f}
    
    # All formats should have the same set of item references
    assert refs_json_f == refs_md_f == refs_itxt_f, \
        f"All formats should have same items: JSON={refs_json_f}, MD={refs_md_f}, ITXT={refs_itxt_f}"
    
    # Verify the expected items are present (nested_item and its 7 children)
    assert len(items_json_f) == 8, f"Expected 8 items, got {len(items_json_f)}"
    assert nested_item.self_ref in refs_json_f, "Start item should be included"
