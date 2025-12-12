"""Test serialization."""

from pathlib import Path

from docling_core.transforms.serializer.doctags import (
    DocTagsDocSerializer,
    DocTagsParams,
)
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DoclingDocument, TableData
from docling_core.types.doc.labels import DocItemLabel

from .test_serialization import verify

# ===============================
# DocTags tests
# ===============================


def serialize_doctags(doc: DoclingDocument, **param_overrides) -> str:
    params = DocTagsParams(**param_overrides)
    ser = DocTagsDocSerializer(doc=doc, params=params)
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

    txt = serialize_doctags(doc, add_content=False)

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

    txt = serialize_doctags(doc, add_content=False)
    assert "Figure Caption Text" not in txt


def test_list_items_not_double_wrapped_when_no_content():
    doc = DoclingDocument(name="t")
    lst = doc.add_list_group()
    doc.add_list_item("Item A", parent=lst)
    doc.add_list_item("Item B", parent=lst)

    txt = serialize_doctags(doc, add_content=True)
    # print(f"txt with content:\n{txt}")

    txt = serialize_doctags(doc, add_content=False)
    # print(f"txt without content:\n{txt}")

    # No nested <list_item><list_item>
    assert "<list_item><list_item>" not in txt

    # Should still have exactly two opening list_item wrappers (for the two items)
    # Note: other occurrences could appear in location tokens etc., so be conservative
    assert txt.count("<list_item>") >= 2


def test_doctags_inline_loc_tags():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = DocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".out.dt"), actual=actual)


def test_doctags_rich_table(rich_table_doc):
    exp_file = Path("./test/data/doc/rich_table.out.dt")

    ser = DocTagsDocSerializer(doc=rich_table_doc)
    actual = ser.serialize().text
    verify(exp_file=exp_file, actual=actual)


def test_doctags_inline_and_formatting():
    src = Path("./test/data/doc/inline_and_formatting.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = DocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.dt"), actual=actual)


def test_doctags_meta():
    src = Path("./test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = DocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.dt"), actual=actual)
