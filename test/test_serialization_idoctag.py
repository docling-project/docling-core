"""Unit tests for IDocTags create_closing_token helper."""

from pathlib import Path
from test.test_serialization import verify

import pytest

from docling_core.experimental.idoctags import (
    ContentType,
    EscapeMode,
    IDocTagsDocSerializer,
    IDocTagsParams,
    IDocTagsSerializationMode,
    IDocTagsVocabulary,
)
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    Script,
    TableData,
)
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DescriptionMetaField,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    SummaryMetaField,
    TabularChartMetaField,
)
from docling_core.types.doc.labels import CodeLanguageLabel, PictureClassificationLabel

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

    # Human readable, indented and with content
    params = IDocTagsParams()

    ser = IDocTagsDocSerializer(doc=doc, params=params)
    actual = ser.serialize().text

    verify(exp_file=src.with_suffix(".v0.gt.idt"), actual=actual)

    # Human readable, indented but without content
    ser = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            content_types={ContentType.TABLE},
        ),
    )
    actual = ser.serialize().text

    verify(exp_file=src.with_suffix(".v1.gt.idt"), actual=actual)

    # Machine readable, not indented and without content
    ser = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            pretty_indentation="",
            content_types={ContentType.TABLE},
            mode=IDocTagsSerializationMode.LLM_FRIENDLY,
        ),
    )
    actual = ser.serialize().text

    verify(exp_file=src.with_suffix(".v2.gt.idt"), actual=actual)


def test_idoctags_meta():
    src = Path("./test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = IDocTagsDocSerializer(doc=doc)
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.idt.xml"), actual=actual)


def _create_escape_test_doc(inp_doc: DoclingDocument):
    doc = inp_doc.model_copy(deep=True)
    doc.add_text(label=DocItemLabel.TEXT, text="Simple text")
    doc.add_text(label=DocItemLabel.TEXT, text="    4 leading spaces, 1 trailing ")
    doc.add_text(label=DocItemLabel.TEXT, text="Some 'single' quotes")
    doc.add_text(label=DocItemLabel.TEXT, text='Some "double" quotes')
    text_item = doc.add_text(label=DocItemLabel.TEXT, text="An ampersand: &")
    text_item.meta = PictureMeta(
        summary=SummaryMetaField(text="Summary with <tags> & ampersands"),
        description=DescriptionMetaField(text="Description content"),
    )
    doc.add_code(text="0 == 0")
    doc.add_code(text=" 1 leading space, 4 trailing    ")
    doc.add_code(text="0 < 1")
    doc.add_code(text="42 == 42", code_language=CodeLanguageLabel.PYTHON)
    doc.add_code(text="42 < 1337", code_language=CodeLanguageLabel.PYTHON)

    td = TableData(num_cols=2)
    td.add_row(["Foo", "Bar"])
    td.add_row(["Header & Title", "Value > 100"])
    td.add_row(["<script>", "A & B"])
    td.add_row(["Only", "<second>"])
    doc.add_table(data=td)

    # test combination of formatting and special characters
    doc.add_text(label=DocItemLabel.TEXT, text="0 < 1")
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="0 < 42",
        formatting=Formatting(bold=True, italic=True),
    )

    return doc


def test_cdata_always(sample_doc: DoclingDocument):
    """Test cdata_always mode."""
    doc = _create_escape_test_doc(sample_doc)
    serializer = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            escape_mode=EscapeMode.CDATA_ALWAYS,
        ),
    )
    ser_res = serializer.serialize()
    ser_txt = ser_res.text

    exp_file = Path("./test/data/doc/cdata_always.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_cdata_when_needed(sample_doc: DoclingDocument):
    """Test cdata_when_needed mode."""
    doc = _create_escape_test_doc(sample_doc)
    serializer = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            escape_mode=EscapeMode.CDATA_WHEN_NEEDED,
        ),
    )
    ser_res = serializer.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/cdata_when_needed.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_strikethrough_formatting():
    """Test strikethrough formatting serialization."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(strikethrough=True)
    doc.add_text(label=DocItemLabel.TEXT, text="Strike text", formatting=formatting)

    result = serialize_idoctags(
        doc, add_location=False, add_content=True, include_formatting=True
    )
    assert "<strikethrough>Strike text</strikethrough>" in result


def test_subscript_formatting():
    """Test subscript formatting serialization."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(script=Script.SUB)
    doc.add_text(label=DocItemLabel.TEXT, text="H2O", formatting=formatting)

    result = serialize_idoctags(
        doc, add_location=False, add_content=True, include_formatting=True
    )
    assert "<subscript>H2O</subscript>" in result


def test_superscript_formatting():
    """Test superscript formatting serialization."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(script=Script.SUPER)
    doc.add_text(label=DocItemLabel.TEXT, text="x^2", formatting=formatting)

    result = serialize_idoctags(
        doc, add_location=False, add_content=True, include_formatting=True
    )
    assert "<superscript>x^2</superscript>" in result


def test_combined_formatting():
    """Test combined formatting (bold + italic)."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(bold=True, italic=True)
    doc.add_text(label=DocItemLabel.TEXT, text="Bold and italic", formatting=formatting)

    result = serialize_idoctags(
        doc, add_location=False, add_content=True, include_formatting=True
    )
    # When both bold and italic are applied, they should be nested
    assert "<bold>" in result
    assert "<italic>" in result
    assert "Bold and italic" in result


def test_formatting_disabled():
    """Test that formatting is not applied when include_formatting=False."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(bold=True, italic=True)
    doc.add_text(label=DocItemLabel.TEXT, text="Plain text", formatting=formatting)

    result = serialize_idoctags(
        doc, add_location=False, add_content=True, include_formatting=False
    )
    # Formatting tags should not be present
    assert "<bold>" not in result
    assert "<italic>" not in result
    assert "Plain text" in result


def _create_content_filtering_doc(inp_doc: DoclingDocument):
    doc = inp_doc.model_copy(deep=True)
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    pic = doc.add_picture(
        caption=doc.add_text(label=DocItemLabel.CAPTION, text="Picture Caption")
    )
    pic.prov = [prov]
    pic.meta = PictureMeta(
        summary=SummaryMetaField(text="Picture Summary"),
        description=DescriptionMetaField(text="Picture Description"),
    )

    chart = doc.add_picture(
        caption=doc.add_text(label=DocItemLabel.CAPTION, text="Picture Caption")
    )
    chart.prov = [prov]
    chart.meta = PictureMeta(
        summary=SummaryMetaField(text="Picture Summary"),
        description=DescriptionMetaField(text="Picture Description"),
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name=PictureClassificationLabel.PIE_CHART.value,
                    confidence=1.0,
                )
            ]
        ),
    )
    chart_data = TableData(num_cols=2)
    chart_data.add_row(["Foo", "Bar"])
    chart_data.add_row(["One", "Two"])
    chart.meta.tabular_chart = TabularChartMetaField(
        title="Chart Title",
        chart_data=chart_data,
    )
    doc.add_code(text="0 == 0")
    doc.add_code(text="with location", prov=prov)

    return doc


def test_content_allow_all_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    serializer = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            content_types={ct for ct in ContentType},
        ),
    )
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/content_all.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_content_allow_no_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    serializer = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            content_types=set(),
        ),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/content_none.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_content_allow_specific_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    serializer = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            content_types={
                ContentType.PICTURE,
                ContentType.TABLE,
                ContentType.TABLE_CELL,
                ContentType.REF_CAPTION,
                ContentType.TEXT_CODE,
            },
        ),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/content_specific.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_content_block_specific_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    blocked_types = {
        ContentType.TABLE,
        ContentType.TEXT_CODE,
    }
    serializer = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            content_types={ct for ct in ContentType if ct not in blocked_types},
        ),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/content_block_specific.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_inline_group():
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )

    parent_txt = doc.add_text(label=DocItemLabel.TEXT, text="", prov=prov)
    simple_inline_gr = doc.add_inline_group(parent=parent_txt)
    doc.add_text(label=DocItemLabel.TEXT, text="One", parent=simple_inline_gr)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Two",
        parent=simple_inline_gr,
        formatting=Formatting(bold=True),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Three", parent=simple_inline_gr)

    li_inline_gr = doc.add_list_group()
    doc.add_list_item(text="Item 1", parent=li_inline_gr)
    li2 = doc.add_list_item(text="", parent=li_inline_gr)
    li2_inline_gr = doc.add_inline_group(parent=li2)
    doc.add_text(label=DocItemLabel.TEXT, text="Four", parent=li2_inline_gr)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Five",
        parent=li2_inline_gr,
        formatting=Formatting(bold=True),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Six", parent=li2_inline_gr)

    ser = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            add_content=True,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/inline_group.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_mini_inline():
    doc = DoclingDocument(name="test")
    ul = doc.add_list_group()
    li = doc.add_list_item(text="", parent=ul)
    inl = doc.add_inline_group(parent=li)
    doc.add_text(label=DocItemLabel.TEXT, text="foo", parent=inl)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="bar",
        parent=inl,
        formatting=Formatting(bold=True),
    )
    ser = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            add_content=True,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/mini_inline.gt.idt.xml")
    verify(exp_file=exp_file, actual=ser_txt)