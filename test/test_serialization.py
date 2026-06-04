"""Test serialization."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from docling_core.transforms.serializer.common import (
    _DEFAULT_LABELS,
    InlineBoundary,
    _classify_ambiguous_word_boundary,
    _classify_inline_boundary,
    _classify_provenance_boundary,
    _classify_text_boundary,
    _is_semantic_inline_atom,
    _is_styled_text,
    _join_inline_parts,
    create_ser_result,
)
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLMetaSerializer,
    HTMLOutputStyle,
    HTMLParams,
    HTMLTableSerializer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
    OrigListItemMarkerMode,
    _cell_content_has_table,
)
from docling_core.transforms.serializer.webvtt import WebVTTDocSerializer, WebVTTParams
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import BoundingBox, ImageRefMode
from docling_core.types.doc.document import (
    BaseMeta,
    CharSpan,
    DescriptionAnnotation,
    DoclingDocument,
    EntitiesMetaField,
    EntityMention,
    Formatting,
    LanguageMetaField,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    RefItem,
    RichTableCell,
    SummaryMetaField,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA


def verify(exp_file: Path, actual: str):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{actual}\n")
    else:
        with open(exp_file, encoding="utf-8") as f:
            expected = f.read().rstrip()

        # Normalize platform-dependent quote escaping for DocTags outputs
        name = exp_file.name
        if name.endswith((".dt", ".idt", ".idt.xml")):

            def _normalize_quotes(s: str) -> str:
                return s.replace("&quot;", '"').replace("&#34;", '"')

            expected = _normalize_quotes(expected)
            actual = _normalize_quotes(actual)

        assert actual == expected


def _make_inline_doc() -> tuple[DoclingDocument, object]:
    doc = DoclingDocument(name="test")
    return doc, doc.add_inline_group()


def _add_inline_text(
    doc: DoclingDocument,
    group,
    text: str,
    *,
    label: DocItemLabel = DocItemLabel.TEXT,
    **kwargs,
):
    return doc.add_text(label=label, parent=group, text=text, **kwargs)


def _extract_body_content(html: str) -> str:
    start = html.find("<body>") + 6
    end = html.find("</body>")
    return html[start:end].strip()


def _make_prov(start: int, end: int, *, page_no: int = 1) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=0.0, t=0.0, r=1.0, b=1.0),
        charspan=(start, end),
    )


# ===============================
# Markdown tests
# ===============================


def test_md_cross_page_list_page_break():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            page_break_placeholder="<!-- page break -->",
            labels=_DEFAULT_LABELS - {DocItemLabel.PICTURE},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.md"), actual=actual)


def test_md_checkboxes():
    src = Path("./test/data/doc/checkboxes.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            page_break_placeholder="<!-- page break -->",
            labels=_DEFAULT_LABELS - {DocItemLabel.PICTURE},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.gt.md", actual=actual)


def test_md_cross_page_list_page_break_none():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            page_break_placeholder=None,
            labels=_DEFAULT_LABELS - {DocItemLabel.PICTURE},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_pb_none.gt.md", actual=actual)


def test_md_cross_page_list_page_break_empty():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            page_break_placeholder="",
            labels=_DEFAULT_LABELS - {DocItemLabel.PICTURE},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_pb_empty.gt.md", actual=actual)


def test_md_cross_page_list_page_break_non_empty():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            page_break_placeholder="<!-- page-break -->",
            labels=_DEFAULT_LABELS - {DocItemLabel.PICTURE},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_pb_non_empty.gt.md", actual=actual)


def test_md_cross_page_list_page_break_p2():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            page_break_placeholder=None,
            pages={2},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_p2.gt.md", actual=actual)


def test_md_charts():
    src = Path("./test/data/doc/barchart.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.md"), actual=actual)


def test_md_inline_and_formatting():
    src = Path("./test/data/doc/inline_and_formatting.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.md"), actual=actual)


def test_md_pb_placeholder_and_page_filter():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    # NOTE ambiguous case
    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            page_break_placeholder="<!-- page break -->",
            pages={3, 4, 6},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.md"), actual=actual)


def test_md_list_item_markers(sample_doc):
    root_dir = Path("./test/data/doc")
    for mode in OrigListItemMarkerMode:
        for valid in [False, True]:
            ser = MarkdownDocSerializer(
                doc=sample_doc,
                params=MarkdownParams(
                    orig_list_item_marker_mode=mode,
                    ensure_valid_list_item_marker=valid,
                ),
            )
            actual = ser.serialize().text
            verify(
                root_dir / f"constructed_mode_{str(mode.value).lower()}_valid_{str(valid).lower()}.gt.md",
                actual=actual,
            )


def test_md_mark_meta_true():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            mark_meta=True,
            pages={1, 5},
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_mark_meta_true.gt.md",
        actual=actual,
    )


def test_md_mark_meta_false():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            mark_meta=False,
            pages={1, 5},
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_mark_meta_false.gt.md",
        actual=actual,
    )


def test_md_legacy_annotations_mark_true(sample_doc):
    exp_file = Path("./test/data/doc/constructed_legacy_annot_mark_true.gt.md")
    with pytest.warns(DeprecationWarning):
        sample_doc.tables[0].annotations.append(
            DescriptionAnnotation(text="This is a description of table 1.", provenance="foo")
        )
        ser = MarkdownDocSerializer(
            doc=sample_doc,
            params=MarkdownParams(
                mark_annotations=True,
            ),
        )
        actual = ser.serialize().text
    verify(
        exp_file=exp_file,
        actual=actual,
    )


def test_md_legacy_annotations_mark_false(sample_doc):
    exp_file = Path("./test/data/doc/constructed_legacy_annot_mark_false.gt.md")
    with pytest.warns(DeprecationWarning):
        sample_doc.tables[0].annotations.append(
            DescriptionAnnotation(text="This is a description of table 1.", provenance="foo")
        )
        ser = MarkdownDocSerializer(
            doc=sample_doc,
            params=MarkdownParams(
                mark_annotations=False,
            ),
        )
        actual = ser.serialize().text
    verify(
        exp_file=exp_file,
        actual=actual,
    )


def test_md_nested_lists():
    src = Path("./test/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.md"), actual=actual)


def test_md_rich_table(rich_table_doc):
    exp_file = Path("./test/data/doc/rich_table.gt.md")

    ser = MarkdownDocSerializer(doc=rich_table_doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def test_md_single_row_table():
    exp_file = Path("./test/data/doc/single_row_table.gt.md")
    words = ["foo", "bar"]
    doc = DoclingDocument(name="")
    row_idx = 0
    table = doc.add_table(data=TableData(num_rows=1, num_cols=len(words)))
    for col_idx, word in enumerate(words):
        doc.add_table_cell(
            table_item=table,
            cell=TableCell(
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + 1,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + 1,
                text=word,
            ),
        )

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def test_md_pipe_in_table():
    doc = DoclingDocument(name="Pipe in Table")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    # TODO: properly handle nested tables, for now just escape the pipe
    doc.add_table_cell(
        table,
        TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Fruits | Veggies",
        ),
    )
    ser = doc.export_to_markdown()
    assert ser == "| Fruits &#124; Veggies   |\n|-------------------------|"


def test_cell_content_has_table_detects_descendant_table():
    """Ensure nested tables are detected through non-table parent nodes."""
    doc = DoclingDocument(name="descendant_table")
    wrapper = doc.add_group()
    nested_table = doc.add_table(data=TableData(num_rows=1, num_cols=1), parent=wrapper)
    doc.add_table_cell(
        nested_table,
        TableCell(
            text="inner",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        ),
    )

    assert _cell_content_has_table(wrapper, doc)


def _build_nested_rich_table_doc(depth: int) -> DoclingDocument:
    """Build a document with `depth` levels of nested RichTableCell tables.

    Each level is a 1x2 table whose first cell is a RichTableCell referencing
    the next-level table, and whose second cell is a plain TableCell.
    This is the structure produced by the HTML backend for Wikipedia clade tables.
    """
    doc = DoclingDocument(name="nested_tables")

    def _add_level(parent, remaining: int):
        table = doc.add_table(data=TableData(num_rows=1, num_cols=2), parent=parent)
        if remaining > 0:
            nested = _add_level(table, remaining - 1)
            rich_cell: TableCell = RichTableCell(
                ref=nested.get_ref(),
                text="rich",
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            )
        else:
            rich_cell = TableCell(
                text="leaf",
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            )
        doc.add_table_cell(table, rich_cell)
        doc.add_table_cell(
            table,
            TableCell(
                text="plain",
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=1,
                end_col_offset_idx=2,
            ),
        )
        return table

    _add_level(doc.body, depth)
    return doc


def test_md_nested_rich_table_no_hang():
    """Regression: export_to_markdown() must not hang on nested RichTableCells.

    When a RichTableCell's content contains a nested table, the
    ``_nested_in_table`` flag passed through kwargs causes
    MarkdownTableSerializer to flatten the inner table instead of
    re-entering the full table serializer recursively.  Without this
    guard every level of nesting re-enters the table serializer, causing
    exponential string growth and an indefinite hang.
    """
    doc = _build_nested_rich_table_doc(depth=5)

    result: list[str] = []

    def _run() -> None:
        result.append(doc.export_to_markdown())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive(), "export_to_markdown() hung on a document with nested RichTableCells."
    assert result, "export_to_markdown() produced no output"

    # The outer table must be a valid 2-column markdown table.
    # Without the pipe-escaping fix, inner-table pipes would leak into the outer
    # table and produce dozens of phantom columns.
    table_rows = [line for line in result[0].splitlines() if line.startswith("|")]
    assert table_rows, "Expected at least one markdown table row in output"
    col_counts = {line.count("|") - 1 for line in table_rows}
    assert col_counts == {2}, f"Outer table must have exactly 2 columns throughout; got column counts: {col_counts}"


def test_md_compact_table():
    """Test compact table format removes padding and uses minimal separators."""
    # Test the _compact_table method directly
    padded_table = """| item   | qty   | description           |
| ------ | ----: | :-------------------: |
| spam   | 42    | A canned meat product |
| eggs   | 451   | Fresh farm eggs       |
| bacon  | 0     | Out of stock          |"""

    expected_compact = """| item | qty | description |
| - | -: | :-: |
| spam | 42 | A canned meat product |
| eggs | 451 | Fresh farm eggs |
| bacon | 0 | Out of stock |"""

    compact_result = MarkdownTableSerializer._compact_table(padded_table)
    assert compact_result == expected_compact

    # Verify space savings
    assert len(compact_result) < len(padded_table)


def test_md_numeric_precision_preserved():
    """Test that numeric values in tables preserve their full precision.

    Regression test for issue where tabulate's numparse would silently
    truncate numeric strings to ~6 significant figures.
    """
    doc = DoclingDocument(name="Numeric Precision Test")
    precise_values = [
        "225.8183",
        "24797.34",
        "20896.7184",
        "17358.138",
        "123.456789",
    ]
    table = doc.add_table(data=TableData(num_rows=len(precise_values) + 1, num_cols=2))

    # Add header row
    doc.add_table_cell(
        table_item=table,
        cell=TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Description",
        ),
    )
    doc.add_table_cell(
        table_item=table,
        cell=TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="Value",
        ),
    )

    # Add data rows with precise numeric values
    for row_idx, value in enumerate(precise_values, start=1):
        doc.add_table_cell(
            table_item=table,
            cell=TableCell(
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + 1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                text=f"Item {row_idx}",
            ),
        )
        doc.add_table_cell(
            table_item=table,
            cell=TableCell(
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + 1,
                start_col_offset_idx=1,
                end_col_offset_idx=2,
                text=value,
            ),
        )

    markdown_output = doc.export_to_markdown()
    for value in precise_values:
        assert value in markdown_output, (
            f"Numeric value '{value}' was not preserved in markdown output. "
            "This indicates precision loss during table serialization."
        )


def test_md_traverse_pictures():
    """Test traverse_pictures parameter to include text inside PictureItems."""
    doc = DoclingDocument(name="Test Document")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Text before picture")
    picture = doc.add_picture()

    # Manually add a text item as child of picture
    text_in_picture = TextItem(
        self_ref=f"#/texts/{len(doc.texts)}",
        parent=RefItem(cref=picture.self_ref),
        label=DocItemLabel.PARAGRAPH,
        text="Text inside picture",
        orig="Text inside picture",
    )
    doc.texts.append(text_in_picture)
    picture.children.append(RefItem(cref=text_in_picture.self_ref))
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Text after picture")

    # Test with traverse_pictures=False (default)
    ser_no_traverse = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            traverse_pictures=False,
        ),
    )
    result_no_traverse = ser_no_traverse.serialize().text

    # Should NOT contain text inside picture
    assert "Text before picture" in result_no_traverse
    assert "Text after picture" in result_no_traverse
    assert "Text inside picture" not in result_no_traverse
    assert "<!-- image -->" in result_no_traverse

    # Test with traverse_pictures=True
    ser_with_traverse = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="<!-- image -->",
            traverse_pictures=True,
        ),
    )
    result_with_traverse = ser_with_traverse.serialize().text

    # Should contain text inside picture
    assert "Text before picture" in result_with_traverse
    assert "Text after picture" in result_with_traverse
    assert "Text inside picture" in result_with_traverse
    assert "<!-- image -->" in result_with_traverse


# ===============================
# HTML tests
# ===============================


def test_html_table_serializer_get_header_and_body_lines():
    """Test HTMLTableSerializer.get_header_and_body_lines() method."""
    serializer = HTMLTableSerializer()

    # Test 1: Valid HTML with headers and data
    valid_html = "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Data1</td><td>Data2</td></tr></table>"
    headers, body = serializer.get_header_and_body_lines(table_text=valid_html)
    assert len(headers) > 0, "Should have headers"
    assert len(body) > 0, "Should have body rows"

    # Test 2: Row without closing </tr> tag
    # Parser will find the row, but when we search for </tr> it won't be found
    no_close_tr = "<tr><th>Header</th></tr><tr><td>Data1"
    headers, body = serializer.get_header_and_body_lines(table_text=no_close_tr)
    assert isinstance(headers, list)
    assert isinstance(body, list)

    # Test 3: Data rows with incomplete closing tags
    # When collecting remaining rows, some </tr> tags are missing
    incomplete_data = "<tr><th>H1</th></tr><tr><td>D1</td></tr><tr><td>D2"
    headers, body = serializer.get_header_and_body_lines(table_text=incomplete_data)
    assert isinstance(headers, list)
    assert isinstance(body, list)

    # Test 4: Force exception in parser
    with patch("docling_core.transforms.serializer.html._SimpleHTMLTableParser") as mock_parser_class:
        mock_parser = MagicMock()
        mock_parser.feed.side_effect = Exception("Parser error")
        mock_parser_class.return_value = mock_parser

        broken_html = "<tr><th>Header</th></tr><tr><td>Data</td></tr>"
        headers, body = serializer.get_header_and_body_lines(table_text=broken_html)
        # Should use fallback logic
        assert isinstance(headers, list)
        assert isinstance(body, list)

    # Test 5: Parser returns more rows than exist in HTML
    # Mock parser to return extra rows that don't exist in the HTML
    with patch("docling_core.transforms.serializer.html._SimpleHTMLTableParser") as mock_parser_class:
        mock_parser = MagicMock()
        # Create fake row data - more rows than actually exist in HTML
        mock_parser.rows = [
            {"th_cells": ["H1"], "td_cells": []},
            {"th_cells": ["H2"], "td_cells": []},
            {"th_cells": ["H3"], "td_cells": []},  # This row doesn't exist in HTML
            {"th_cells": [], "td_cells": ["D1"]},
        ]
        mock_parser_class.return_value = mock_parser

        # HTML with only 2 rows, but parser claims 4
        limited_html = "<tr><th>H1</th></tr><tr><th>H2</th></tr>"
        headers, body = serializer.get_header_and_body_lines(table_text=limited_html)
        assert isinstance(headers, list)
        assert isinstance(body, list)

    # Test 6: Specific case for line 485 - row_start found but row_end not found
    # Create HTML where parser finds a row, but the actual HTML has <tr without </tr>
    with patch("docling_core.transforms.serializer.html._SimpleHTMLTableParser") as mock_parser_class:
        mock_parser = MagicMock()
        # Parser reports a header row exists
        mock_parser.rows = [
            {"th_cells": ["Header"], "td_cells": []},
        ]
        mock_parser_class.return_value = mock_parser

        # But the HTML has <tr without matching </tr>
        html_no_close = "<tr><th>Header"
        headers, body = serializer.get_header_and_body_lines(table_text=html_no_close)
        assert isinstance(headers, list)
        assert isinstance(body, list)

    # Test 7: Specific case for line 504 - data collection finds <tr but no </tr>
    # Create HTML where we start collecting data rows but encounter incomplete row
    with patch("docling_core.transforms.serializer.html._SimpleHTMLTableParser") as mock_parser_class:
        mock_parser = MagicMock()
        # Parser reports header then data rows
        mock_parser.rows = [
            {"th_cells": ["H"], "td_cells": []},
            {"th_cells": [], "td_cells": ["D1"]},  # This triggers data collection
        ]
        mock_parser_class.return_value = mock_parser

        # HTML has complete header but incomplete data row
        html_incomplete_data = "<tr><th>H</th></tr><tr><td>D1</td></tr><tr><td>D2"
        headers, body = serializer.get_header_and_body_lines(table_text=html_incomplete_data)
        assert isinstance(headers, list)
        assert isinstance(body, list)

    # Test 8: Table with footer content
    with_footer = "<tr><th>H</th></tr><tr><td>D</td></tr>Footer content"
    headers, body = serializer.get_header_and_body_lines(table_text=with_footer)
    assert isinstance(headers, list)
    assert isinstance(body, list)
    # Footer should be in body
    assert "Footer" in str(body)


def test_html_charts():
    src = Path("./test/data/doc/barchart.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.PLACEHOLDER,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.html"), actual=actual)


def test_html_cross_page_list_page_break():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.PLACEHOLDER,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.html"), actual=actual)


def test_html_cross_page_list_page_break_p1():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            pages={1},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_p1.gt.html", actual=actual)


def test_html_cross_page_list_page_break_p2():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            pages={2},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_p2.gt.html", actual=actual)


def test_html_split_page():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.EMBEDDED,
            output_style=HTMLOutputStyle.SPLIT_PAGE,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_split.gt.html", actual=actual)


def test_html_split_page_p2():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.EMBEDDED,
            output_style=HTMLOutputStyle.SPLIT_PAGE,
            pages={2},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_split_p2.gt.html", actual=actual)


def test_html_split_page_p2_with_visualizer():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.EMBEDDED,
            output_style=HTMLOutputStyle.SPLIT_PAGE,
            pages={2},
        ),
    )
    ser_res = ser.serialize(
        visualizer=LayoutVisualizer(),
    )
    actual = ser_res.text

    # pinning the result with visualizer appeared flaky, so at least ensure it contains
    # a figure (for the page) and that it is different than without visualizer:
    assert '<figure><img src="data:image/png;base64' in actual
    file_without_viz = src.parent / f"{src.stem}_split_p2.gt.html"
    with open(file_without_viz) as f:
        data_without_viz = f.read()
    assert actual.strip() != data_without_viz.strip()


def test_html_split_page_no_page_breaks():
    src = Path("./test/data/doc/2408.09869_p1.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.EMBEDDED,
            output_style=HTMLOutputStyle.SPLIT_PAGE,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_split.gt.html", actual=actual)


def test_html_include_annotations_false():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            include_annotations=False,
            pages={1},
            html_head="<head></head>",  # keeping test output minimal
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_include_annotations_false.gt.html",
        actual=actual,
    )


def test_html_include_annotations_true():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            include_annotations=True,
            pages={1},
            html_head="<head></head>",  # keeping test output minimal
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_include_annotations_true.gt.html",
        actual=actual,
    )


def test_html_list_item_markers(sample_doc):
    root_dir = Path("./test/data/doc")
    for orig in [False, True]:
        ser = HTMLDocSerializer(
            doc=sample_doc,
            params=HTMLParams(
                show_original_list_item_marker=orig,
            ),
        )
        actual = ser.serialize().text
        verify(
            root_dir / f"constructed_orig_{str(orig).lower()}.gt.html",
            actual=actual,
        )


def test_html_nested_lists():
    src = Path("./test/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.html"), actual=actual)


def test_html_rich_table(rich_table_doc):
    exp_file = Path("./test/data/doc/rich_table.gt.html")

    ser = HTMLDocSerializer(doc=rich_table_doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def test_html_rich_cell_textitem_ref_subtree_inside_and_not_outside():
    """Descendants of a RichTableCell's TextItem ref render inside the table.

    With HTMLTextSerializer recursing into its item's children, the parent text
    (CELL-TEXT) and its child (DEEP-LEAK) both render as siblings inside the
    rich cell. The outer document iteration must not re-emit either of them as
    standalone content after the table.
    """
    doc = DoclingDocument(name="rich_cell_textitem_subtree")

    table = doc.add_table(data=TableData(num_rows=1, num_cols=2))
    rich_ref = doc.add_text(label=DocItemLabel.TEXT, text="CELL-TEXT", parent=table)
    doc.add_text(label=DocItemLabel.TEXT, text="DEEP-LEAK", parent=rich_ref)

    doc.add_table_cell(
        table_item=table,
        cell=RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            ref=rich_ref.get_ref(),
            text="cell 0,0",
        ),
    )
    doc.add_table_cell(
        table_item=table,
        cell=TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="plain",
        ),
    )

    out = HTMLDocSerializer(doc=doc).serialize().text
    body = out[out.find("<body>") : out.find("</body>") + len("</body>")]
    table_end = body.find("</table>") + len("</table>")
    inside_table = body[:table_end]
    after_table = body[table_end:]

    assert "CELL-TEXT" in inside_table, f"RichTableCell ref content missing from the table:\n{body}"
    assert "DEEP-LEAK" in inside_table, f"RichTableCell ref descendant missing from the table:\n{body}"
    assert "CELL-TEXT" not in after_table, f"RichTableCell ref content leaked outside the table:\n{body}"
    assert "DEEP-LEAK" not in after_table, f"RichTableCell ref descendant leaked outside the table:\n{body}"


def test_html_textitem_with_children_at_document_level():
    """Doc-level parity: TextItem with TextItem child renders both exactly once.

    Once HTMLTextSerializer recurses into children, the child is rendered by
    the parent's serialize call instead of by the outer iteration. Both must
    still appear in document order, exactly once each.
    """
    doc = DoclingDocument(name="textitem_with_children")
    parent = doc.add_text(label=DocItemLabel.TEXT, text="PARENT-TEXT")
    doc.add_text(label=DocItemLabel.TEXT, text="CHILD-TEXT", parent=parent)

    out = HTMLDocSerializer(doc=doc).serialize().text
    body = out[out.find("<body>") : out.find("</body>") + len("</body>")]

    assert body.count("PARENT-TEXT") == 1, f"PARENT-TEXT should appear exactly once:\n{body}"
    assert body.count("CHILD-TEXT") == 1, f"CHILD-TEXT should appear exactly once:\n{body}"
    assert body.find("PARENT-TEXT") < body.find("CHILD-TEXT"), f"PARENT-TEXT should appear before CHILD-TEXT:\n{body}"


def test_html_inline_and_formatting():
    src = Path("./test/data/doc/inline_and_formatting.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = HTMLDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.html"), actual=actual)


# ===============================
# WebVTT tests
# ===============================


@pytest.mark.parametrize(
    "example_num",
    [1, 2, 3, 4, 5],
)
def test_webvtt(example_num):
    src = Path(f"test/data/doc/webvtt_example_{example_num:02d}.json")
    doc = DoclingDocument.load_from_json(src)

    ser = WebVTTDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.vtt"), actual=actual)


def test_webvtt_params():
    """Test WebVTT serialization with WebVTTParams."""
    src = Path("./test/data/doc/webvtt_example_01.json")
    doc = DoclingDocument.load_from_json(src)

    # Test with omit_hours_if_zero=True
    ser = WebVTTDocSerializer(doc=doc, params=WebVTTParams(omit_hours_if_zero=True))
    actual = ser.serialize().text
    assert "00:11.000 --> 00:13.000" in actual

    # Test with omit_voice_end=True
    ser = WebVTTDocSerializer(doc=doc, params=WebVTTParams(omit_voice_end=True))
    actual = ser.serialize().text
    assert "</v>" not in actual

    # Test with both parameters enabled
    ser = WebVTTDocSerializer(doc=doc, params=WebVTTParams(omit_hours_if_zero=True, omit_voice_end=True))
    actual = ser.serialize().text

    assert "00:11.000 --> 00:13.000" in actual
    assert "</v>" not in actual

    ser_default = WebVTTDocSerializer(doc=doc, params=WebVTTParams())
    actual_default = ser_default.serialize().text
    assert len(actual) <= len(actual_default) or actual != actual_default


def test_html_meta_emits_xhtml_compatible_attributes():
    """Test that metadata attributes are name=value pairs."""
    doc = DoclingDocument(name="x")

    plain = TableCell(
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        text="",
    )
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[plain]))

    pic = doc.add_picture(parent=table)
    pic.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name="other", confidence=1.0)]
        )
    )

    table.data.table_cells = [
        RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="",
            ref=pic.get_ref(),
        )
    ]

    text = doc.add_text(
        label=DocItemLabel.TEXT, text="Output of HTML serializer must be parseable by a strict XML parser"
    )
    text.meta = BaseMeta(
        summary=SummaryMetaField(text="XHTML-compliant"),
        language=LanguageMetaField(code="en"),
        entities=EntitiesMetaField(
            mentions=[
                EntityMention(
                    text="HTML serializer",
                    label="software",
                    charspan=CharSpan((10, 25)),
                ),
                EntityMention(
                    text="XML parser",
                    label="software",
                    charspan=CharSpan((56, 66)),
                ),
            ]
        ),
    )

    html_out = table.export_to_html(doc)
    # No bare valueless attribute like `data-meta-classification` (must be
    # followed by `=` to be XHTML-compliant).
    assert "data-meta-classification>" not in html_out
    assert 'data-meta-name="classification"' in html_out

    html_out = doc.export_to_html()
    print(html_out)
    assert 'data-meta-name="language"' in html_out
    assert 'data-meta-name="entities"' in html_out
    # Output must be parseable by a strict XML parser.
    ET.fromstring(html_out)


# ===============================
# Tests for inline group join behavior without spaces
# ===============================


def test_md_inline_group_no_spaces():
    """Test that inline groups join text parts without spaces for continuous text."""
    doc, group = _make_inline_doc()
    _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text="D",
        formatting=Formatting(
            bold=True,
            italic=False,
            underline=False,
            strikethrough=False,
            script="baseline",
        ),
    )
    _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text="ocling",
        formatting=Formatting(
            bold=False,
            italic=False,
            underline=False,
            strikethrough=False,
            script="baseline",
        ),
    )

    # This should serialize as "**D**ocling" without space
    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text.strip()

    expected = "**D**ocling"
    assert actual == expected


def test_html_inline_group_no_spaces():
    """Test that inline groups join text parts without spaces for continuous text."""
    doc, group = _make_inline_doc()
    _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text="Project",
        formatting=Formatting(
            bold=True,
            italic=False,
            underline=False,
            strikethrough=False,
            script="baseline",
        ),
    )
    _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text="ing",
        formatting=Formatting(
            bold=False,
            italic=False,
            underline=False,
            strikethrough=False,
            script="baseline",
        ),
    )

    # This should serialize as <strong>Project</strong>ing without space
    ser = HTMLDocSerializer(
        doc=doc, params=HTMLParams(html_head="<head></head>", prettify=False)
    )
    actual = ser.serialize().text

    body_content = _extract_body_content(actual)
    assert "<strong>Project</strong>ing" in body_content


def test_md_inline_group_mixed_formatting_mid_word():
    """Test inline group with different formatting mid-word."""
    doc, group = _make_inline_doc()
    _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="Pars")
    _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text="ing",
        formatting=Formatting(
            bold=False,
            italic=True,
            underline=False,
            strikethrough=False,
            script="baseline",
        ),
    )

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text.strip()

    expected = "Pars*ing*"
    assert actual == expected


def test_html_inline_group_mixed_formatting_mid_word():
    """Test inline group with different formatting mid-word."""
    doc, group = _make_inline_doc()
    _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="Pars")
    _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text="ing",
        formatting=Formatting(
            bold=False,
            italic=True,
            underline=False,
            strikethrough=False,
            script="baseline",
        ),
    )

    ser = HTMLDocSerializer(
        doc=doc, params=HTMLParams(html_head="<head></head>", prettify=False)
    )
    actual = ser.serialize().text

    body_content = _extract_body_content(actual)
    assert "Pars<em>ing</em>" in body_content.replace("\n", " ")


def test_md_inline_group_single_part():
    """Test inline group with single text part (no joining needed)."""
    doc, group = _make_inline_doc()
    _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="Single")

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text.strip()

    expected = "Single"
    assert actual == expected


def test_html_inline_group_single_part():
    """Test inline group with single text part (no joining needed)."""
    doc, group = _make_inline_doc()
    _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="Single")

    ser = HTMLDocSerializer(
        doc=doc, params=HTMLParams(html_head="<head></head>", prettify=False)
    )
    actual = ser.serialize().text

    body_content = _extract_body_content(actual)
    assert "Single" in body_content


def test_join_inline_parts_filters_empty_parts_and_inserts_spacing():
    doc, group = _make_inline_doc()
    text_item = _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="A hyperlink on")
    code_item = doc.add_code(text="code in a line", parent=group, hyperlink="#link")

    parts = [
        create_ser_result(text="", span_source=text_item),
        create_ser_result(text="A hyperlink on", span_source=text_item),
        create_ser_result(text="[`code in a line`](#link)", span_source=code_item),
    ]

    assert _join_inline_parts(parts) == "A hyperlink on [`code in a line`](#link)"


@pytest.mark.parametrize(
    ("prev_orig", "curr_orig", "prev_prov", "curr_prov", "expected"),
    [
        ("Hello ", "world", None, None, InlineBoundary.SPACE),
        ("Hello", "world", _make_prov(0, 5), _make_prov(5, 10), InlineBoundary.JOIN),
        ("Hello", "world", _make_prov(0, 5), _make_prov(6, 11), InlineBoundary.SPACE),
        ("Hello", "world", _make_prov(0, 5, page_no=1), _make_prov(0, 5, page_no=2), InlineBoundary.UNKNOWN),
    ],
)
def test_classify_source_boundary(prev_orig, curr_orig, prev_prov, curr_prov, expected):
    doc, group = _make_inline_doc()
    prev_item = _add_inline_text(
        doc, group, label=DocItemLabel.TEXT, text=prev_orig.strip(), orig=prev_orig, prov=prev_prov
    )
    curr_item = _add_inline_text(
        doc, group, label=DocItemLabel.TEXT, text=curr_orig.strip(), orig=curr_orig, prov=curr_prov
    )

    assert _classify_provenance_boundary(prev_item=prev_item, item=curr_item) == expected


def test_classify_inline_boundary_handles_missing_items_and_whitespace():
    assert (
        _classify_inline_boundary(prev_text="foo ", prev_item=None, text="bar", item=None)
        == InlineBoundary.JOIN
    )
    assert (
        _classify_inline_boundary(prev_text="foo", prev_item=None, text="bar", item=None)
        == InlineBoundary.UNKNOWN
    )


@pytest.mark.parametrize(
    ("prev_text", "prev_formatting", "prev_hyperlink", "curr_text", "curr_formatting", "curr_hyperlink", "expected"),
    [
        ("snippet:", None, None, "bold", Formatting(bold=True), None, InlineBoundary.SPACE),
        ("D", Formatting(bold=True), None, "ocling", None, None, InlineBoundary.JOIN),
        ("Pars", None, None, "ing", Formatting(italic=True), None, InlineBoundary.JOIN),
        ("Foo", None, None, "emphasis", Formatting(italic=True), None, InlineBoundary.SPACE),
        ("strong emphasis", Formatting(bold=True), None, "tail", None, None, InlineBoundary.SPACE),
        ("bold", Formatting(bold=True), None, "italic", Formatting(italic=True), None, InlineBoundary.SPACE),
        ("bold", Formatting(bold=True), None, "&", None, None, InlineBoundary.SPACE),
        ("hello-", None, None, "world", Formatting(italic=True), None, InlineBoundary.UNKNOWN),
    ],
)
def test_classify_text_boundary(
    prev_text,
    prev_formatting,
    prev_hyperlink,
    curr_text,
    curr_formatting,
    curr_hyperlink,
    expected,
):
    doc, group = _make_inline_doc()
    prev_item = _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text=prev_text,
        formatting=prev_formatting,
        hyperlink=prev_hyperlink,
    )
    curr_item = _add_inline_text(
        doc,
        group,
        label=DocItemLabel.TEXT,
        text=curr_text,
        formatting=curr_formatting,
        hyperlink=curr_hyperlink,
    )

    assert _classify_text_boundary(prev_item=prev_item, item=curr_item) == expected


def test_inline_boundary_separates_semantic_atoms_from_text():
    doc, group = _make_inline_doc()
    prev_item = _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="A hyperlink on")
    code_item = doc.add_code(text="code in a line", parent=group, hyperlink="#link")
    formula_item = doc.add_formula(text="E=mc^2", parent=group)
    next_item = _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="(inline)")

    assert (
        _classify_inline_boundary(
            prev_text="A hyperlink on",
            prev_item=prev_item,
            text="[`code in a line`](#link)",
            item=code_item,
        )
        == InlineBoundary.SPACE
    )
    assert (
        _classify_inline_boundary(
            prev_text="$E=mc^2$",
            prev_item=formula_item,
            text="(inline)",
            item=next_item,
        )
        == InlineBoundary.SPACE
    )


def test_common_inline_helper_flags():
    doc, group = _make_inline_doc()
    plain = _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="plain")
    bold = _add_inline_text(doc, group, label=DocItemLabel.TEXT, text="bold", formatting=Formatting(bold=True))
    linked = _add_inline_text(
        doc, group, label=DocItemLabel.TEXT, text="link", hyperlink="https://example.com"
    )
    code = doc.add_code(text="print()", parent=group)
    formula = doc.add_formula(text="E=mc^2", parent=group)

    assert _is_styled_text(plain) is False
    assert _is_styled_text(bold) is True
    assert _is_styled_text(linked) is True

    assert _is_semantic_inline_atom(plain) is False
    assert _is_semantic_inline_atom(linked) is True
    assert _is_semantic_inline_atom(code) is True
    assert _is_semantic_inline_atom(formula) is True


@pytest.mark.parametrize(
    ("curr_raw_text", "expected"),
    [
        ("ing", InlineBoundary.JOIN),
        ("emphasis", InlineBoundary.SPACE),
        ("XYZ", InlineBoundary.SPACE),
        ("42", InlineBoundary.SPACE),
    ],
)
def test_classify_ambiguous_word_boundary(curr_raw_text, expected):
    assert _classify_ambiguous_word_boundary(curr_raw_text=curr_raw_text) == expected
