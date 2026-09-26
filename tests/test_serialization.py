"""Test serialization."""

import threading
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TypeAlias, Union
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import pytest
from pydantic import AnyUrl

from docling_core.transforms.serializer.common import _DEFAULT_LABELS
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLMetaSerializer,
    HTMLOutputStyle,
    HTMLParams,
    HTMLTableSerializer,
)
from docling_core.transforms.serializer.markdown import (
    CaptionPlacement,
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
    OrigListItemMarkerMode,
    _cell_content_has_table,
)
from docling_core.transforms.serializer.webvtt import WebVTTDocSerializer, WebVTTParams
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.types.doc import DoclingDocument, ImageRef
from docling_core.types.doc.base import BoundingBox, CoordOrigin, ImageRefMode, Size
from docling_core.types.doc.document import (
    BaseMeta,
    CharSpan,
    DescriptionAnnotation,
    EntitiesMetaField,
    EntityMention,
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


# ===============================
# Markdown tests
# ===============================


def test_md_cross_page_list_page_break():
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/checkboxes.json")
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
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/activities.json")
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


_PAGE_BREAK = "<!-- page break -->"


# builds a document whose items carry the given (text, page number) provenance
def _make_paged_doc(entries: list[tuple[str, int]], *, as_list: bool, num_pages: int) -> DoclingDocument:
    doc = DoclingDocument(name="paged")
    for page_no in range(1, num_pages + 1):
        doc.add_page(page_no=page_no, size=Size(width=595, height=842), image=None)
    group = doc.add_list_group(name="list") if as_list else None
    for text, page_no in entries:
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=50, t=700, r=500, b=680, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, len(text)),
        )
        if group is not None:
            doc.add_list_item(text=text, parent=group, prov=prov)
        else:
            doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
    return doc


def test_md_cross_page_list_page_break_matches_per_page_export():
    doc = _make_paged_doc(
        [("item one", 1), ("item two", 1), ("item three", 2), ("item four", 2)],
        as_list=True,
        num_pages=2,
    )
    globally = doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK)
    per_page = f"\n\n{_PAGE_BREAK}\n\n".join(doc.export_to_markdown(page_no=n) for n in (1, 2))
    assert globally == per_page


def test_md_cross_page_text_page_break_matches_per_page_export():
    # the control: outside a list scope this already held, and must keep holding
    doc = _make_paged_doc(
        [("item one", 1), ("item two", 1), ("item three", 2), ("item four", 2)],
        as_list=False,
        num_pages=2,
    )
    globally = doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK)
    per_page = f"\n\n{_PAGE_BREAK}\n\n".join(doc.export_to_markdown(page_no=n) for n in (1, 2))
    assert globally == per_page


def test_md_page_break_spacing_independent_of_scope():
    entries = [("item one", 1), ("item two", 1), ("item three", 2)]
    in_list = _make_paged_doc(entries, as_list=True, num_pages=2).export_to_markdown(page_break_placeholder=_PAGE_BREAK)
    at_doc_scope = _make_paged_doc(entries, as_list=False, num_pages=2).export_to_markdown(
        page_break_placeholder=_PAGE_BREAK
    )
    for actual in (in_list, at_doc_scope):
        assert f"\n\n{_PAGE_BREAK}\n\n" in actual


def test_md_cross_page_list_page_break_empty_placeholder():
    doc = _make_paged_doc([("item one", 1), ("item two", 2)], as_list=True, num_pages=2)
    # an empty placeholder still occupies the blank lines the marker would have
    assert doc.export_to_markdown(page_break_placeholder="") == "- item one\n\n\n\n- item two"


def test_md_page_break_at_document_bounds():
    doc = _make_paged_doc([("item 1", 1), ("item 2", 2), ("item 3", 3)], as_list=True, num_pages=3)
    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(page_break_placeholder=_PAGE_BREAK, pages={2}),
    )
    # a break opening or closing the export gains no blank line on the outer side
    assert ser.serialize().text == f"{_PAGE_BREAK}\n\n- item 2\n\n{_PAGE_BREAK}"


def test_md_consecutive_page_breaks():
    doc = _make_paged_doc([(f"item {n}", n) for n in range(1, 7)], as_list=True, num_pages=6)
    ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(page_break_placeholder=_PAGE_BREAK, pages={3, 4, 6}),
    )
    actual = ser.serialize().text
    assert f"{_PAGE_BREAK}\n\n{_PAGE_BREAK}" in actual
    assert "\n\n\n" not in actual


def test_md_nested_list_page_break_keeps_indent():
    doc = DoclingDocument(name="nested")
    for page_no in (1, 2):
        doc.add_page(page_no=page_no, size=Size(width=595, height=842), image=None)

    def _prov(page_no: int, text: str) -> ProvenanceItem:
        return ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=50, t=700, r=500, b=680, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, len(text)),
        )

    outer = doc.add_list_group(name="outer")
    doc.add_list_item(text="outer one", parent=outer, prov=_prov(1, "outer one"))
    inner = doc.add_list_group(name="inner", parent=outer)
    doc.add_list_item(text="inner a", parent=inner, prov=_prov(1, "inner a"))
    doc.add_list_item(text="inner b", parent=inner, prov=_prov(2, "inner b"))
    doc.add_list_item(text="outer two", parent=outer, prov=_prov(2, "outer two"))

    # the marker keeps the indentation that holds it inside the outer item
    assert f"\n\n    {_PAGE_BREAK}\n\n" in doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK)


def test_md_page_break_between_list_and_table():
    doc = _make_paged_doc([("item one", 1), ("item two", 1)], as_list=True, num_pages=2)
    doc.add_table(
        data=TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                TableCell(
                    text="cell",
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                )
            ],
        ),
        prov=ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(l=50, t=700, r=500, b=680, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, 4),
        ),
    )
    actual = doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK)
    assert f"- item two\n\n{_PAGE_BREAK}\n\n|" in actual


def test_md_charts():
    src = Path("./tests/data/doc/barchart.json")
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
    src = Path("./tests/data/doc/inline_and_formatting.yaml")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    root_dir = Path("./tests/data/doc")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    exp_file = Path("./tests/data/doc/constructed_legacy_annot_mark_true.gt.md")
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
    exp_file = Path("./tests/data/doc/constructed_legacy_annot_mark_false.gt.md")
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
    src = Path("./tests/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.md"), actual=actual)


def test_md_rich_table(rich_table_doc):
    exp_file = Path("./tests/data/doc/rich_table.gt.md")

    ser = MarkdownDocSerializer(doc=rich_table_doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def test_md_single_row_table():
    exp_file = Path("./tests/data/doc/single_row_table.gt.md")
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


def test_md_field_region():
    exp_file = Path("./tests/data/doc/field_region.gt.md")

    doc = DoclingDocument(name="")
    field_region = doc.add_field_region()
    field_item = doc.add_field_item(parent=field_region)
    doc.add_field_key(text="Name:", parent=field_item)
    doc.add_field_value(text="John Doe", parent=field_item)

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def _table_with_rows(rows: list[list[str]], num_header_rows: int) -> DoclingDocument:
    doc = DoclingDocument(name="")
    table = doc.add_table(data=TableData(num_rows=len(rows), num_cols=len(rows[0])))
    for row_idx, row in enumerate(rows):
        for col_idx, text in enumerate(row):
            doc.add_table_cell(
                table_item=table,
                cell=TableCell(
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    text=text,
                    column_header=row_idx < num_header_rows,
                ),
            )
    return doc


def test_md_table_stacked_header_is_flattened():
    doc = _table_with_rows(
        [["Product", "Tier 1", "Tier 2"], ["", "0-99", "100+"], ["CAT-001", "10.00", "9.00"]],
        num_header_rows=2,
    )
    actual = MarkdownDocSerializer(doc=doc).serialize().text
    assert actual == (
        "| Product   |   Tier 1 - 0-99 |   Tier 2 - 100+ |\n"
        "|-----------|-----------------|-----------------|\n"
        "| CAT-001   |           10.00 |            9.00 |"
    )


def test_md_table_stacked_header_drops_repeated_span_text():
    doc = _table_with_rows(
        [["human", "MRCNN"], ["human", "R50"], ["84-89", "68.4"]],
        num_header_rows=2,
    )
    actual = MarkdownDocSerializer(doc=doc).serialize().text
    assert actual.splitlines()[0] == "| human   |   MRCNN - R50 |"


def test_md_table_keeps_rows_under_a_vertically_spanning_header():
    """A row-spanning header is repeated into each row it covers by `grid`.

    Only rows a header starts on are header rows, otherwise the data rows under
    the span get flattened into the header and disappear from the body.
    """
    doc = DoclingDocument(name="")
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2))
    for cell in (
        TableCell(
            text="Category",
            start_row_offset_idx=0,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            row_span=2,
            column_header=True,
        ),
        TableCell(
            text="Price",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            column_header=True,
        ),
        TableCell(
            text="10.00",
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
    ):
        doc.add_table_cell(table_item=table, cell=cell)

    actual = MarkdownDocSerializer(doc=doc).serialize().text
    assert actual == ("| Category   |   Price |\n|------------|---------|\n| Category   |   10.00 |")


def test_md_table_row_header_above_the_body_is_not_promoted():
    """A row header that starts on the first body row must stay in the body.

    The ``"2025"`` cell below is intentionally mis-flagged with
    ``column_header=True`` rather than ``row_header=True``. That reproduces the
    erroneous output of the HTML backend prior to
    ``docling-project/docling#4216``, and models documents already serialized
    with those old flags. It is not a canonical model of a pivot table; this
    test only verifies the defensive guard in the markdown serializer.
    """
    doc = DoclingDocument(name="")
    table = doc.add_table(data=TableData(num_rows=3, num_cols=2))
    for cell in (
        TableCell(
            text="Year",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            column_header=True,
        ),
        TableCell(
            text="Month",
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            column_header=True,
        ),
        TableCell(
            text="2025",
            start_row_offset_idx=1,
            end_row_offset_idx=3,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            row_span=2,
            column_header=True,
        ),
        TableCell(
            text="January",
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
        TableCell(
            text="February",
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
    ):
        doc.add_table_cell(table_item=table, cell=cell)

    actual = MarkdownDocSerializer(doc=doc).serialize().text
    lines = actual.splitlines()
    assert lines[0].replace(" ", "") == "|Year|Month|"
    assert lines[2].replace(" ", "") == "|2025|January|"
    assert lines[3].replace(" ", "") == "|2025|February|"


def test_md_table_with_header_flags_below_row_zero_keeps_every_row_as_data():
    """No leading header block, so nothing is promoted and no row is dropped."""
    doc = DoclingDocument(name="")
    rows = [["title", ""], ["Product", "Price"], ["CAT-001", "10.00"]]
    table = doc.add_table(data=TableData(num_rows=len(rows), num_cols=2))
    for row_idx, row in enumerate(rows):
        for col_idx, text in enumerate(row):
            doc.add_table_cell(
                table_item=table,
                cell=TableCell(
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    text=text,
                    column_header=row_idx == 1,
                ),
            )

    actual = MarkdownDocSerializer(doc=doc).serialize().text
    assert actual.splitlines()[0] == "|         |       |"
    assert "| title   |       |" in actual
    assert "| CAT-001 | 10.00 |" in actual


def test_md_table_without_header_flags_keeps_first_row_as_header():
    """Backends that never set column_header keep the pre-existing behavior."""
    doc = _table_with_rows([["foo", "bar"], ["baz", "qux"]], num_header_rows=0)
    actual = MarkdownDocSerializer(doc=doc).serialize().text
    assert actual.splitlines()[0] == "| foo   | bar   |"


def test_md_table_single_header_row_with_flags():
    """A single flagged header row is used as-is without joining or deduplication."""
    doc = _table_with_rows(
        [["Product", "Price"], ["CAT-001", "10.00"], ["CAT-002", "9.00"]],
        num_header_rows=1,
    )
    actual = MarkdownDocSerializer(doc=doc).serialize().text
    lines = actual.splitlines()
    assert lines[0] == "| Product   |   Price |"
    assert lines[2] == "| CAT-001   |   10.00 |"
    assert lines[3] == "| CAT-002   |    9.00 |"


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


def test_md_heading_in_rich_table_cell_renders_as_plain_text():
    """Test headings inside RichTableCell must not emit `#` markers.

    According to the Markdown spec, heading syntax is invalid inside tables.
    A SectionHeaderItem or TitleItem referenced by a RichTableCell must be
    serialized as plain text, not as `## Heading`.
    """
    doc = DoclingDocument(name="heading_in_table")
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2))

    section_header = doc.add_heading(text="Section Heading", level=1, parent=table)
    title = doc.add_title(text="Document Title", parent=table)

    doc.add_table_cell(
        table,
        RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            ref=section_header.get_ref(),
        ),
    )
    doc.add_table_cell(
        table,
        RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            ref=title.get_ref(),
        ),
    )
    doc.add_table_cell(
        table,
        TableCell(
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="value A",
        ),
    )
    doc.add_table_cell(
        table,
        TableCell(
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="value B",
        ),
    )

    md = doc.export_to_markdown()
    assert "Section Heading" in md
    assert "Document Title" in md
    table_lines = [line for line in md.splitlines() if line.startswith("|")]
    for line in table_lines:
        assert "#" not in line, f"Heading marker leaked into table cell: {line!r}\nFull output:\n{md}"


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


def test_md_line_breaks():
    """Newlines inside TextItem and its subclasses are serialized per GFM spec.

    Single `\\n` in body text and list items becomes a hard line break (`  \\n`).
    Double `\\n\\n` is preserved as a paragraph break.
    Heading newlines are collapsed to a space (headings cannot span multiple lines).
    Table cell newlines are replaced with a space (table syntax requires single lines).
    """
    # Single \n in a TextItem -> GFM hard line break (two trailing spaces).
    doc = DoclingDocument(name="lb")
    doc.add_text(label=DocItemLabel.TEXT, text="Hello\nWorld")
    assert MarkdownDocSerializer(doc=doc).serialize().text == "Hello  \nWorld"

    # \n\n in a TextItem -> paragraph break, no trailing spaces added.
    doc = DoclingDocument(name="lb")
    doc.add_text(label=DocItemLabel.TEXT, text="Para one\n\nPara two")
    assert MarkdownDocSerializer(doc=doc).serialize().text == "Para one\n\nPara two"

    # Mix of single and double newlines in one TextItem.
    doc = DoclingDocument(name="lb")
    doc.add_text(label=DocItemLabel.TEXT, text="A\nB\n\nC\nD")
    assert MarkdownDocSerializer(doc=doc).serialize().text == "A  \nB\n\nC  \nD"

    # \n in a SectionHeaderItem -> space (headings cannot span multiple lines;
    # "Hello\nWorld" must become "### Hello World", not "### HelloWorld").
    doc = DoclingDocument(name="lb")
    doc.add_heading(text="Hello\nWorld", level=2)
    assert MarkdownDocSerializer(doc=doc).serialize().text == "### Hello World"

    # \n in a ListItem -> GFM hard line break.
    doc = DoclingDocument(name="lb")
    lg = doc.add_group()
    doc.add_list_item(text="Line one\nLine two", parent=lg)
    assert MarkdownDocSerializer(doc=doc).serialize().text == "- Line one  \nLine two"

    # \n in a plain table cell -> space (keeps the table row on a single line).
    doc = DoclingDocument(name="lb")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    doc.add_table_cell(
        table_item=table,
        cell=TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Line one\nLine two",
        ),
    )
    result = MarkdownDocSerializer(doc=doc).serialize().text
    assert "\n" not in result.split("|")[1].strip()


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
    src = Path("./tests/data/doc/barchart.json")
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
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/activities.json")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    src = Path("./tests/data/doc/2408.09869_p1.json")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    src = Path("./tests/data/doc/2408.09869v3_enriched.json")
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
    root_dir = Path("./tests/data/doc")
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
    src = Path("./tests/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = HTMLDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.html"), actual=actual)


def test_html_rich_table(rich_table_doc):
    exp_file = Path("./tests/data/doc/rich_table.gt.html")

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
    src = Path("./tests/data/doc/inline_and_formatting.yaml")
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
    src = Path(f"tests/data/doc/webvtt_example_{example_num:02d}.json")
    doc = DoclingDocument.load_from_json(src)

    ser = WebVTTDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.vtt"), actual=actual)


def test_webvtt_params():
    """Test WebVTT serialization with WebVTTParams."""
    src = Path("./tests/data/doc/webvtt_example_01.json")
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


# A link destination to encode, paired with its expected encoding.
_EscapeCase: TypeAlias = tuple[AnyUrl | PurePath, str]

_ESCAPE_PATH_CASES: list[_EscapeCase] = [
    # A relative path with nothing to encode: the hash-based names generated by
    # `_with_pictures_refs` must pass through untouched.
    (PurePosixPath("doc_artifacts/image_000001_ab12.png"), "doc_artifacts/image_000001_ab12.png"),
    # A relative artifacts dir is derived from the output filename, so it can hold
    # spaces. An unencoded space ends the link destination.
    (PurePosixPath("My Report_artifacts/img.png"), "My%20Report_artifacts/img.png"),
    # Parentheses are only valid in a destination while balanced; encoding them keeps
    # an odd one in a filename from ending the link early.
    (PurePosixPath("artifacts/img (1).png"), "artifacts/img%20%281%29.png"),
    # In a path these are literal characters, not URI delimiters. "%" is kept as-is so
    # that an already-encoded URI is not encoded twice.
    (PurePosixPath("100%_scale/a#b?c.png"), "100%_scale/a%23b%3Fc.png"),
    # A root-relative POSIX path stays root-relative.
    (PurePosixPath("/home/a b/img.png"), "/home/a%20b/img.png"),
    # Backslash separators must become forward slashes: "\" is an escape character in
    # Markdown, so "dir\img.png" would render as "dirimg.png".
    (PureWindowsPath("My Report_artifacts/img.png"), "My%20Report_artifacts/img.png"),
    # An absolute Windows path becomes an RFC 8089 file:// URL, so that "C:" cannot be
    # read as a URL scheme.
    (PureWindowsPath("C:/Users/me/My Docs/img.png"), "file:///C:/Users/me/My%20Docs/img.png"),
    # A UNC share becomes the authority of the file:// URL, so that the leading "//"
    # cannot be read as a scheme-relative URL.
    (PureWindowsPath("//server/share/My Docs/img.png"), "file://server/share/My%20Docs/img.png"),
]

_ESCAPE_URL_CASES: list[_EscapeCase] = [
    # A URL keeps its scheme, authority, and delimiters; only the components get
    # encoded. Escapes pydantic already applied must not become "%2520".
    (AnyUrl("file:///home/a b/img.png"), "file:///home/a%20b/img.png"),
    (AnyUrl("s3://bucket/My Report_artifacts/img.png"), "s3://bucket/My%20Report_artifacts/img.png"),
    (AnyUrl("https://example.com:8080/a b.png?w=1&h=2#frag"), "https://example.com:8080/a%20b.png?w=1&h=2#frag"),
    # pydantic leaves parentheses as-is, so the encoding still has work to do here.
    (AnyUrl("https://example.com/img (1).png"), "https://example.com/img%20%281%29.png"),
]


def _as_destination(dest: str) -> AnyUrl | PurePath:
    """Re-read an encoded destination the way a caller would supply it."""
    return AnyUrl(dest) if "://" in dest else PurePosixPath(dest)


@pytest.mark.parametrize(("value", "expected"), _ESCAPE_PATH_CASES + _ESCAPE_URL_CASES)
def test_escape_uri_path(value: AnyUrl | PurePath, expected: str):
    """Test encoding of link destinations for both URLs and local paths."""
    assert MarkdownPictureSerializer._escape_uri_path(value) == expected


@pytest.mark.parametrize(("value", "expected"), _ESCAPE_PATH_CASES + _ESCAPE_URL_CASES)
def test_escape_uri_path_is_idempotent(value: AnyUrl | PurePath, expected: str):
    """Test that re-encoding an encoded destination is a no-op (no double-encoding)."""
    assert MarkdownPictureSerializer._escape_uri_path(_as_destination(expected)) == expected


@pytest.mark.parametrize(
    "posix_spelling",
    [
        "doc_artifacts/image_000001_ab12.png",
        "My Report_artifacts/img.png",
        "C:/Users/me/My Docs/img.png",
        "//server/share/My Docs/img.png",
    ],
)
def test_escape_uri_path_is_flavour_independent(posix_spelling: str):
    """Test that a path encodes the same way however its separators arrive.

    A document authored on Windows is re-read as a POSIX path on another host, and PRs
    #641 and #663 make `_with_pictures_refs` store the POSIX spelling on Windows. Neither
    may change the destination.
    """
    escape = MarkdownPictureSerializer._escape_uri_path
    windows = PureWindowsPath(posix_spelling)
    assert "\\" in str(windows)  # guard: the fixture really has separators to convert

    assert escape(PurePosixPath(posix_spelling)) == escape(windows)
    assert escape(PurePosixPath(str(windows))) == escape(windows)


def test_escape_uri_path_file_scheme_is_only_for_absolute_windows_paths():
    """Test that only an absolute Windows path is given a file:// scheme."""
    escape = MarkdownPictureSerializer._escape_uri_path

    # A drive letter and a UNC host are the two absolute Windows forms.
    assert escape(PureWindowsPath("C:/Users/me/img.png")) == "file:///C:/Users/me/img.png"
    assert escape(PureWindowsPath("//server/share/img.png")) == "file://server/share/img.png"

    # Nothing else acquires the scheme, a root-relative POSIX path included.
    for value in (
        PurePosixPath("artifacts/img.png"),
        PurePosixPath("/home/me/img.png"),
        PureWindowsPath("artifacts/img.png"),
        AnyUrl("https://example.com/img.png"),
        AnyUrl("s3://bucket/img.png"),
    ):
        assert not escape(value).startswith("file:")

    # A URL that already carries the scheme keeps it: dropping it would turn an absolute
    # filesystem reference into a root-relative URL.
    assert escape(AnyUrl("file:///home/me/img.png")) == "file:///home/me/img.png"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (Path("doc_artifacts/image_000001_ab12.png"), "doc_artifacts/image_000001_ab12.png"),
        (Path("My Report (final)_artifacts/img.png"), "My%20Report%20%28final%29_artifacts/img.png"),
        # A Windows-authored separator resolves the same way on either exporting host.
        (Path("My Report_artifacts\\img.png"), "My%20Report_artifacts/img.png"),
        # An AnyUrl was already percent-encoded by pydantic on parse: its escapes must
        # not be encoded a second time, nor its scheme mangled into "file%3A".
        (AnyUrl("file:///home/a b/img.png"), "file:///home/a%20b/img.png"),
        # `_save_image_and_resolve_uri` returns an AnyUrl for remote artifact dirs.
        (AnyUrl("s3://bucket/My Report_artifacts/img.png"), "s3://bucket/My%20Report_artifacts/img.png"),
        # Query delimiters must stay functional.
        (AnyUrl("https://example.com/img.png?w=1&h=2"), "https://example.com/img.png?w=1&h=2"),
    ],
)
def test_referenced_image_uri_is_encoded(uri, expected: str):
    """Test that `_serialize_image_part` encodes the URI it emits."""
    doc = DoclingDocument(name="x")
    doc.add_picture(image=ImageRef(mimetype="image/png", dpi=72, size=Size(width=10, height=10), uri=uri))

    assert doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED) == f"![Image]({expected})"


def test_referenced_image_uri_encoding_only_applies_to_referenced_mode():
    """Test that PLACEHOLDER mode still emits the placeholder, not an encoded URI."""
    doc = DoclingDocument(name="x")
    uri = Path("My Report_artifacts/img.png")
    doc.add_picture(image=ImageRef(mimetype="image/png", dpi=72, size=Size(width=10, height=10), uri=uri))

    assert doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER) == "<!-- image -->"


def test_referenced_image_data_uri_is_not_encoded():
    """Test that a data URI still falls back to the placeholder in REFERENCED mode."""
    doc = DoclingDocument(name="x")
    uri = AnyUrl("data:image/png;base64,iVBORw0KGgo=")
    doc.add_picture(image=ImageRef(mimetype="image/png", dpi=72, size=Size(width=10, height=10), uri=uri))

    assert doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED) == "<!-- image -->"


# ===============================
# Hyperlink destination tests
# ===============================


@pytest.mark.parametrize(
    ("hyperlink", "expected"),
    [
        # The plain relative link from the report (native and POSIX spelling).
        (Path("sub/next.html"), "sub/next.html"),
        (PurePosixPath("sub/next.html"), "sub/next.html"),
        # A Windows spelling normalizes on every host -- a native POSIX Path
        # treats '\' as a legal filename character, so the foreign spelling
        # must be expressed with PureWindowsPath to stay host-independent.
        (PureWindowsPath("sub\\next.html"), "sub/next.html"),
        # Fragment and query delimiters are part of the URL, not of the path, and must
        # not be percent-encoded the way an image path would be.
        (Path("sub/next.html#section"), "sub/next.html#section"),
        (Path("sub/next.html?x=1#section"), "sub/next.html?x=1#section"),
        # Spaces and existing escapes stay as they are.
        (Path("dir with space/x.html"), "dir with space/x.html"),
        (Path("a%20b/x.html"), "a%20b/x.html"),
        # A URL is untouched, including its own backslash-free spelling.
        (AnyUrl("https://example.com/a/b?q=1#f"), "https://example.com/a/b?q=1#f"),
    ],
)
def test_hyperlink_uri_is_portable(hyperlink: AnyUrl | PurePath, expected: str):
    """Test that `hyperlink_uri` emits a portable destination on every host."""
    from docling_core.transforms.serializer.common import hyperlink_uri

    assert hyperlink_uri(hyperlink) == expected


@pytest.mark.parametrize(
    ("hyperlink", "expected_md", "expected_html"),
    [
        (Path("sub/next.html"), "[next page](sub/next.html)", '<a href="sub/next.html">'),
        (
            Path("sub/next.html#section"),
            "[next page](sub/next.html#section)",
            '<a href="sub/next.html#section">',
        ),
        (
            Path("sub/next.html?x=1#section"),
            "[next page](sub/next.html?x=1#section)",
            '<a href="sub/next.html?x=1#section">',
        ),
    ],
)
def test_relative_hyperlink_export_uses_forward_slashes(hyperlink: AnyUrl | Path, expected_md: str, expected_html: str):
    """Test that Markdown and HTML export a relative hyperlink with `/` separators."""
    doc = DoclingDocument(name="x")
    doc.add_text(label=DocItemLabel.TEXT, text="next page", hyperlink=hyperlink)

    assert doc.export_to_markdown().strip() == expected_md
    html = doc.export_to_html(image_mode=ImageRefMode.PLACEHOLDER)
    assert expected_html in html


def test_relative_hyperlink_survives_json_round_trip():
    """Test that a JSON round trip keeps a relative hyperlink on any host."""
    doc = DoclingDocument(name="x")
    doc.add_text(label=DocItemLabel.TEXT, text="next page", hyperlink=Path("sub/next.html"))

    reloaded = DoclingDocument.model_validate(doc.model_dump(mode="json"))

    md = reloaded.export_to_markdown().strip()
    assert md == "[next page](sub/next.html)"
    assert "\\" not in md


def test_export_to_markdown_image_dir_saves_and_references_images(sample_doc, tmp_path):
    """export_to_markdown with image_dir saves images and references the portable URI."""
    image_dir = tmp_path / "images"

    md = sample_doc.export_to_markdown(
        image_mode=ImageRefMode.REFERENCED,
        image_dir=image_dir,
        image_uri_prefix="images/",
    )

    saved = sorted(image_dir.glob("*.png"))
    assert saved, "expected at least one image to be saved"
    for img in saved:
        assert f"images/{img.name}" in md
    assert str(image_dir.resolve()) not in md


def test_export_to_html_image_dir_saves_and_references_images(sample_doc, tmp_path):
    """export_to_html with image_dir saves images and references the portable URI."""
    image_dir = tmp_path / "images"

    html = sample_doc.export_to_html(
        image_mode=ImageRefMode.REFERENCED,
        image_dir=image_dir,
        image_uri_prefix="images/",
    )

    saved = sorted(image_dir.glob("*.png"))
    assert saved, "expected at least one image to be saved"
    for img in saved:
        assert f"images/{img.name}" in html


def test_export_to_markdown_image_dir_without_referenced_raises(sample_doc, tmp_path):
    """Passing image_dir without REFERENCED mode raises ValueError."""
    with pytest.raises(ValueError, match=r"ImageRefMode\.REFERENCED"):
        sample_doc.export_to_markdown(image_dir=tmp_path / "images")


def test_export_to_html_image_dir_without_referenced_raises(sample_doc, tmp_path):
    """Passing image_dir without REFERENCED mode raises ValueError."""
    with pytest.raises(ValueError, match=r"ImageRefMode\.REFERENCED"):
        sample_doc.export_to_html(image_dir=tmp_path / "images")


def _make_captioned_doc(
    *,
    kind: str,
    cap_bbox: BoundingBox | None,
    obj_bbox: BoundingBox | None,
    page_no: int = 1,
) -> DoclingDocument:
    doc = DoclingDocument(name="cap_order")
    doc.add_page(page_no=1, size=Size(width=100, height=200))

    def prov(bbox: BoundingBox | None) -> ProvenanceItem | None:
        if bbox is None:
            return None
        return ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, 0))

    caption = doc.add_text(label=DocItemLabel.CAPTION, text="THE CAPTION", prov=prov(cap_bbox))
    if kind == "table":
        table_data = TableData(num_rows=1, num_cols=1)
        table_data.add_row(["cell"])
        table_data.num_rows = 1
        doc.add_table(data=table_data, caption=caption, prov=prov(obj_bbox))
    else:
        doc.add_picture(caption=caption, prov=prov(obj_bbox))
    return doc


def _tl(t: float, b: float) -> BoundingBox:
    return BoundingBox(l=0, r=10, t=t, b=b, coord_origin=CoordOrigin.TOPLEFT)


def _bl(t: float, b: float) -> BoundingBox:
    return BoundingBox(l=0, r=10, t=t, b=b, coord_origin=CoordOrigin.BOTTOMLEFT)


def _serialize_md(doc: DoclingDocument, placement: CaptionPlacement) -> str:
    return MarkdownDocSerializer(doc=doc, params=MarkdownParams(caption_placement=placement)).serialize().text


@pytest.mark.parametrize("kind, marker", [("table", "cell"), ("picture", "<!-- image -->")])
@pytest.mark.parametrize(
    "cap_bbox, obj_bbox, layout_caption_first",
    [
        (None, None, True),  # no bboxes: falls back to standard order (caption first)
        (_tl(150, 170), None, True),  # object without bbox: falls back to standard order
        (_tl(10, 30), _tl(50, 100), True),  # caption above
        (_tl(150, 170), _tl(50, 100), False),  # caption below
        (_bl(190, 170), _bl(150, 100), True),  # caption above, BOTTOMLEFT origin
        (_bl(50, 30), _bl(150, 100), False),  # caption below, BOTTOMLEFT origin
        (_tl(150, 170), _bl(150, 100), False),  # mixed origins: caption (y=160) below object (y=75)
        (_tl(10, 30), _bl(150, 100), True),  # mixed origins: caption (y=20) above object (y=75)
    ],
)
def test_markdown_caption_placement(kind, marker, cap_bbox, obj_bbox, layout_caption_first):
    doc = _make_captioned_doc(kind=kind, cap_bbox=cap_bbox, obj_bbox=obj_bbox)

    # standard ignores bboxes: tables and pictures always have the caption first
    md = _serialize_md(doc, CaptionPlacement.STANDARD)
    assert md.index("THE CAPTION") < md.index(marker)
    assert md == doc.export_to_markdown()

    md = _serialize_md(doc, CaptionPlacement.LAYOUT)
    if layout_caption_first:
        assert md.index("THE CAPTION") < md.index(marker)
    else:
        assert md.index("THE CAPTION") > md.index(marker)


def test_markdown_caption_placement_layout_other_page_falls_back_to_standard():
    doc = DoclingDocument(name="cap_pages")
    doc.add_page(page_no=1, size=Size(width=100, height=200))
    doc.add_page(page_no=2, size=Size(width=100, height=200))
    caption = doc.add_text(
        label=DocItemLabel.CAPTION,
        text="THE CAPTION",
        prov=ProvenanceItem(page_no=2, bbox=_tl(150, 170), charspan=(0, 0)),
    )
    doc.add_picture(caption=caption, prov=ProvenanceItem(page_no=1, bbox=_tl(50, 100), charspan=(0, 0)))
    md = _serialize_md(doc, CaptionPlacement.LAYOUT)
    assert md.index("THE CAPTION") < md.index("<!-- image -->")


@pytest.mark.parametrize(
    "cap_bbox, placement, caption_first",
    [
        (None, CaptionPlacement.STANDARD, False),  # standard: code captions come after
        (_tl(10, 30), CaptionPlacement.STANDARD, False),  # standard ignores bboxes
        (None, CaptionPlacement.LAYOUT, False),  # no bboxes: falls back to standard
        (_tl(10, 30), CaptionPlacement.LAYOUT, True),  # caption above the code
        (_tl(150, 170), CaptionPlacement.LAYOUT, False),  # caption below the code
    ],
)
def test_markdown_caption_placement_code(cap_bbox, placement, caption_first):
    doc = DoclingDocument(name="cap_code")
    doc.add_page(page_no=1, size=Size(width=100, height=200))

    def prov(bbox):
        return None if bbox is None else ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, 0))

    caption = doc.add_text(label=DocItemLabel.CAPTION, text="THE CAPTION", prov=prov(cap_bbox))
    code = doc.add_code(text="print(1)", prov=prov(_tl(50, 100) if cap_bbox else None))
    code.captions.append(caption.get_ref())

    md = _serialize_md(doc, placement)
    assert (md.index("THE CAPTION") < md.index("print(1)")) == caption_first


@pytest.mark.parametrize("placement", [CaptionPlacement.LAYOUT, "layout"])
def test_export_and_save_markdown_caption_placement(tmp_path, placement):
    doc = _make_captioned_doc(kind="picture", cap_bbox=_tl(150, 170), obj_bbox=_tl(50, 100))

    md = doc.export_to_markdown(caption_placement=placement)
    assert md.index("THE CAPTION") > md.index("<!-- image -->")

    out = tmp_path / "out.md"
    doc.save_as_markdown(out, caption_placement=placement)
    assert out.read_text(encoding="utf-8") == md

    # default is unchanged
    assert doc.export_to_markdown().index("THE CAPTION") < doc.export_to_markdown().index("<!-- image -->")
    assert doc.export_to_markdown() == doc.export_to_markdown(caption_placement="standard")
