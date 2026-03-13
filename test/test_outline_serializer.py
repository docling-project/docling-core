import json
from pathlib import Path

from docling_core.experimental.serializer.outline import (
    OutlineDocSerializer,
    OutlineFormat,
    OutlineMode,
    OutlineParams,
)
from docling_core.types.doc import DoclingDocument


def test_outline_serializer_mode_table_of_contents():
    """Test TABLE_OF_CONTENTS mode only includes titles and section headers."""
    doc_path = Path("test/data/doc/2408.09869v5_enriched_summary.json")
    exp_path = doc_path.with_suffix(".toc.gt.md")

    doc = DoclingDocument.load_from_json(filename=doc_path)

    params = OutlineParams(include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS)
    ser = OutlineDocSerializer(doc=doc, params=params)
    result = ser.serialize()

    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert "[reference=#/texts/" in result.text

    reference_count = result.text.count("[reference=")
    assert reference_count > 0, "TABLE_OF_CONTENTS mode should include section headers"
    assert "(summary=" in result.text, "Summaries should be included when include_non_meta=True"

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
    reference_count_outline = result.text.count("[reference=")
    with open(exp_path) as f:
        expected = f.read()
    assert result.text == expected, "Unexpected outline serialization "

    # Compare with TABLE_OF_CONTENTS mode
    params_toc = OutlineParams(include_non_meta=True, mode=OutlineMode.TABLE_OF_CONTENTS)
    ser_toc = OutlineDocSerializer(doc=doc, params=params_toc)
    result_toc = ser_toc.serialize()
    reference_count_toc = result_toc.text.count("[reference=")
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
    assert "[reference=" in result.text, "Should include references"
    assert "(summary=" in result.text, "Should include summaries"

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
    assert "ref" in first_item
    assert first_item["ref"].startswith("#/texts/")

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

