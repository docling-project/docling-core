"""Test serialization."""

from pathlib import Path

import pytest

from docling_core.transforms.serializer.common import _DEFAULT_LABELS
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    OrigListItemMarkerMode,
)
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    DescriptionAnnotation,
    DoclingDocument,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA


def verify(exp_file: Path, actual: str):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{actual}\n")
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
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
        )
    )
    ser = doc.export_to_markdown()
    assert ser == "| Fruits &#124; Veggies   |\n|-------------------------|"

# ===============================
# HTML tests
# ===============================


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


def test_html_inline_and_formatting():
    src = Path("./test/data/doc/inline_and_formatting.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = HTMLDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.html"), actual=actual)
