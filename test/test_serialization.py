"""Test serialization."""

from pathlib import Path

from docling_core.experimental.serializer.common import _DEFAULT_LABELS
from docling_core.experimental.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.experimental.serializer.latex import (
    LaTeXDocSerializer,
    LaTeXOutputStyle,
    LaTeXParams,
)
from docling_core.experimental.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA


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
            pages={4, 6},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.gt.md", actual=actual)


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


def test_latex_basic_document():
    src = Path("./test/data/doc/2408.09869_p1.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            document_class="article",
            document_options="a4paper,12pt",
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.gt.tex", actual=actual)


def test_latex_presentation():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            document_class="beamer",
            presentation_theme="Warsaw",
            presentation_color_theme="dolphin",
            presentation_font_theme="structurebold",
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_presentation.gt.tex", actual=actual)


def test_latex_tables():
    src = Path("./test/data/doc/barchart.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            document_class="article",
            enable_chart_tables=True,
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.gt.tex", actual=actual)


def test_latex_formulas():
    src = Path("./test/data/doc/formulas.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            document_class="article",
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}.gt.tex", actual=actual)


def test_latex_cross_page_list():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            document_class="article",
            pages={1, 2},
        ),
    )
    actual = ser.serialize().text
    verify(exp_file=src.parent / f"{src.stem}_cross_page.gt.tex", actual=actual)
