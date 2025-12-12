"""Unit tests for IDocTags create_closing_token helper."""

from pathlib import Path

import pytest

from docling_core.experimental.idoctags import (
    IDocTagsDocSerializer,
    IDocTagsParams,
    IDocTagsVocabulary,
)
from docling_core.transforms.serializer.doctags import DocTagsParams
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableData

from .test_serialization import verify

# ===============================
# IDocTags unit-tests
# ===============================


def test_create_closing_token_from_opening_tag_simple():
    assert IDocTagsVocabulary.create_closing_token(token="<text>") == "</text>"
    assert (
        IDocTagsVocabulary.create_closing_token(token='\n  <heading level="2">  ')
        == "</heading>"
    )
    assert (
        IDocTagsVocabulary.create_closing_token(token=' <list ordered="true"> ')
        == "</list>"
    )
    # Inline with attribute
    assert (
        IDocTagsVocabulary.create_closing_token(token=' <inline class="code"> ')
        == "</inline>"
    )


def test_create_closing_token_returns_existing_closing():
    assert IDocTagsVocabulary.create_closing_token(token="</text>") == "</text>"


@pytest.mark.parametrize(
    "bad",
    [
        "<br/>",
        '<location value="3"/>',
        '<hour value="1"/>',
        '<thread id="abc"/>',
    ],
)
def test_create_closing_token_rejects_self_closing(bad):
    with pytest.raises(ValueError):
        IDocTagsVocabulary.create_closing_token(token=bad)


@pytest.mark.parametrize(
    "bad",
    [
        "text",  # not a tag
        "<text",  # incomplete
        "<text/>",  # self-closing form of non-self-closing token
        "</ unknown >",  # malformed closing
        "<unknown>",  # unknown token
    ],
)
def test_create_closing_token_invalid_inputs(bad):
    with pytest.raises(ValueError):
        IDocTagsVocabulary.create_closing_token(token=bad)


# ===============================
# IDocTags tests
# ===============================


def serialize_idoctags(doc: DoclingDocument, **param_overrides) -> str:
    params = IDocTagsParams(**param_overrides)
    ser = IDocTagsDocSerializer(doc=doc, params=params)
    return ser.serialize().text


def test_no_content_suppresses_caption_and_table_cell_text():
    doc = DoclingDocument(name="t")

    # Add a caption text item
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Table Caption Text")

    # Build a 2x2 table with header row and data row
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])  # header
    td.add_row(["C1", "C2"])  # data
    doc.add_table(data=td, caption=cap)

    txt = serialize_idoctags(doc, add_content=False)

    # Caption text suppressed
    assert "Table Caption Text" not in txt

    # No table cell text
    for cell_text in ["H1", "H2", "C1", "C2"]:
        assert cell_text not in txt

    # OTSL structural tokens should remain
    assert "<otsl>" in txt and "</otsl>" in txt


def test_no_content_suppresses_figure_caption_text():
    doc = DoclingDocument(name="t")
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Figure Caption Text")
    doc.add_picture(caption=cap)

    txt = serialize_idoctags(doc, add_content=False)
    assert "Figure Caption Text" not in txt


def test_list_items_not_double_wrapped_when_no_content():
    doc = DoclingDocument(name="t")
    lst = doc.add_list_group()
    doc.add_list_item("Item A", parent=lst)
    doc.add_list_item("Item B", parent=lst)

    txt = serialize_idoctags(doc, add_content=True)
    # print(f"txt with content:\n{txt}")

    txt = serialize_idoctags(doc, add_content=False)
    # print(f"txt without content:\n{txt}")

    # No nested <list_text><list_text>
    assert "<list_text><list_text>" not in txt

    # Should still have exactly two opening list_text wrappers (for the two items)
    # Note: other occurrences could appear in location tokens etc., so be conservative
    assert txt.count("<list_text>") >= 2


def test_idoctags():
    src = Path("./test/data/doc/ddoc_0.json")
    doc = DoclingDocument.load_from_json(src)

    if True:
        # Human readable, indented and with content
        params = IDocTagsParams()
        params.add_content = True

        ser = IDocTagsDocSerializer(doc=doc, params=params)
        actual = ser.serialize().text

        verify(exp_file=src.with_suffix(".v0.gt.idt"), actual=actual)

    if True:
        # Human readable, indented but without content
        params = IDocTagsParams()
        params.add_content = False

        ser = IDocTagsDocSerializer(doc=doc, params=params)
        actual = ser.serialize().text

        verify(exp_file=src.with_suffix(".v1.gt.idt"), actual=actual)

    if True:
        # Machine readable, not indented and without content
        params = IDocTagsParams()
        params.pretty_indentation = ""
        params.add_content = False
        params.mode = DocTagsParams.Mode.MINIFIED

        ser = IDocTagsDocSerializer(doc=doc, params=params)
        actual = ser.serialize().text

        verify(exp_file=src.with_suffix(".v2.gt.idt"), actual=actual)


def test_idoctags_meta():
    src = Path("./test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = IDocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.idt.xml"), actual=actual)
