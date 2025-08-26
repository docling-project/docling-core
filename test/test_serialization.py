"""Test serialization."""

from pathlib import Path
from typing import Any

from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import _DEFAULT_LABELS, create_ser_result
from docling_core.transforms.serializer.doctags import DocTagsDocSerializer
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
    OrigListItemMarkerMode,
    _get_annotation_ser_result,
)
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument, MiscAnnotation, TableItem
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA
from .test_docling_doc import _construct_doc, _construct_rich_table_doc


class CustomAnnotationTableSerializer(MarkdownTableSerializer):
    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        params = MarkdownParams(**kwargs)

        res_parts: list[SerializationResult] = []

        if params.include_annotations:
            for ann in item.annotations:
                if isinstance(ann, MiscAnnotation):

                    # custom serialization logic:
                    ann_txt = "\n".join([f"{k}: {ann.content[k]}" for k in ann.content])

                    ann_ser_res = _get_annotation_ser_result(
                        ann_kind=ann.kind,
                        ann_text=ann_txt,
                        mark_annotation=params.mark_annotations,
                        doc_item=item,
                    )
                    res_parts.append(ann_ser_res)

        # reusing the existing result (excluding the annotations):
        parent_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **{**kwargs, **{"include_annotations": False}},
        )
        res_parts.append(parent_res)

        text_res = "\n\n".join([part.text for part in res_parts])
        return create_ser_result(text=text_res, span_source=res_parts)


def verify(exp_file: Path, actual: str):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{actual}\n")
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read().rstrip()

        assert expected == actual


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
    verify(exp_file=src.parent / f"{src.stem}.gt.html", actual=actual)


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
    verify(exp_file=src.parent / f"{src.stem}.gt.md", actual=actual)


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
    verify(exp_file=src.parent / f"{src.stem}.md", actual=actual)


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
    verify(exp_file=src.parent / f"{src.stem}.gt.html", actual=actual)


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
    verify(exp_file=src.parent / f"{src.stem}.gt.md", actual=actual)


def test_md_list_item_markers():
    doc = _construct_doc()
    root_dir = Path("./test/data/doc")
    for mode in OrigListItemMarkerMode:
        for valid in [False, True]:

            ser = MarkdownDocSerializer(
                doc=doc,
                params=MarkdownParams(
                    orig_list_item_marker_mode=mode,
                    ensure_valid_list_item_marker=valid,
                ),
            )
            actual = ser.serialize().text
            verify(
                root_dir
                / f"constructed_mode_{str(mode.value).lower()}_valid_{str(valid).lower()}.gt.md",
                actual=actual,
            )


def test_md_include_annotations_false():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        table_serializer=CustomAnnotationTableSerializer(),
        params=MarkdownParams(
            include_annotations=False,
            pages={1, 5},
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_include_annotations_false.gt.md",
        actual=actual,
    )


def test_md_mark_annotations_false():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        table_serializer=CustomAnnotationTableSerializer(),
        params=MarkdownParams(
            include_annotations=True,
            mark_annotations=False,
            pages={1, 5},
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_mark_annotations_false.gt.md",
        actual=actual,
    )


def test_md_mark_annotations_true():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(
        doc=doc,
        table_serializer=CustomAnnotationTableSerializer(),
        params=MarkdownParams(
            include_annotations=True,
            mark_annotations=True,
            pages={1, 5},
        ),
    )
    actual = ser.serialize().text
    verify(
        exp_file=src.parent / f"{src.stem}_p1_mark_annotations_true.gt.md",
        actual=actual,
    )


def test_md_nested_lists():
    src = Path("./test/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.gt.md", actual=actual)


def test_md_rich_table():
    exp_file = Path("./test/data/doc/rich_table.gt.md")
    doc = _construct_rich_table_doc()

    ser = MarkdownDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


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


def test_html_list_item_markers():
    doc = _construct_doc()
    root_dir = Path("./test/data/doc")
    for orig in [False, True]:

        ser = HTMLDocSerializer(
            doc=doc,
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
    verify(exp_file=src.parent / f"{src.stem}.gt.html", actual=actual)


def test_html_rich_table():
    exp_file = Path("./test/data/doc/rich_table.gt.html")
    doc = _construct_rich_table_doc()

    ser = HTMLDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def test_doctags_inline_loc_tags():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = DocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.out.dt", actual=actual)


def test_doctags_rich_table():

    exp_file = Path("./test/data/doc/rich_table.out.dt")
    doc = _construct_rich_table_doc()

    ser = DocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)
