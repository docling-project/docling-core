"""Unit tests for Doclang create_closing_token helper."""

from itertools import chain
from pathlib import Path
from typing import Optional

import pytest

from docling_core.experimental.doclang import (
    ContentType,
    EscapeMode,
    DoclangDocSerializer,
    DoclangParams,
    DoclangVocabulary,
    LayerMode,
    WrapMode,
)
from docling_core.types.doc import (
    BoundingBox,
    CodeLanguageLabel,
    CoordOrigin,
    DescriptionMetaField,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    Script,
    Size,
    SummaryMetaField,
    TableData,
    TabularChartMetaField,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import ContentLayer, GraphCell, GraphData, GraphLink, ImageRef, RichTableCell, TableCell
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
from test.test_serialization import verify
from test.test_data_gen_flag import GEN_TEST_DATA


def add_texts_section(doc: DoclingDocument):
    doc.add_text(label=DocItemLabel.TEXT, text="Simple text")
    inline1 = doc.add_inline_group()
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a code snippet: ",
        parent=inline1,
    )
    doc.add_code(
        text="help()",
        parent=inline1,
        code_language=CodeLanguageLabel.PYTHON,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=" (to be shown)",
        parent=inline1,
    )

def add_list_section(doc: DoclingDocument):
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    lg = doc.add_list_group()

    doc.add_list_item(text="foo", parent=lg)
    doc.add_list_item(text="bar", parent=lg)

    # just inline group with a formula
    li = doc.add_list_item(text="", parent=lg)
    inline = doc.add_inline_group(parent=li)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a formula: ",
        parent=inline,
    )
    doc.add_formula(text="E=mc^2 ", parent=inline)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="in line",
        parent=inline,
    )

    # just inline group with formatted span
    li = doc.add_list_item(text="", parent=lg)
    inline = doc.add_inline_group(parent=li)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a ",
        parent=inline,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="bold",
        parent=inline,
        formatting=Formatting(bold=True),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=" text",
        parent=inline,
    )

    li = doc.add_list_item(text="will contain sublist", parent=lg)
    lg_sub = doc.add_list_group(parent=li)
    doc.add_list_item(text="sublist item 1", parent=lg_sub)
    doc.add_list_item(text="sublist item 2", parent=lg_sub)

    li = doc.add_list_item(text="", parent=lg, prov=prov)
    inline = doc.add_inline_group(parent=li)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a ",
        parent=inline,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="both bold and italicized",
        parent=inline,
        formatting=Formatting(bold=True, italic=True),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=" text and a sublist:",
        parent=inline,
    )
    lg_sub = doc.add_list_group(parent=li)
    doc.add_list_item(text="sublist item a", parent=lg_sub)
    doc.add_list_item(text="sublist item b", parent=lg_sub)

    doc.add_list_item(text="final element", parent=lg)

# ===============================
# Doclang unit-tests
# ===============================


def test_create_closing_token_from_opening_tag_simple():
    assert DoclangVocabulary.create_closing_token(token="<text>") == "</text>"
    assert (
        DoclangVocabulary.create_closing_token(token='\n  <heading level="2">  ')
        == "</heading>"
    )
    assert (
        DoclangVocabulary.create_closing_token(token=' <list ordered="true"> ')
        == "</list>"
    )


def test_create_closing_token_returns_existing_closing():
    assert DoclangVocabulary.create_closing_token(token="</text>") == "</text>"


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
        DoclangVocabulary.create_closing_token(token=bad)


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
        DoclangVocabulary.create_closing_token(token=bad)


# ===============================
# Doclang tests
# ===============================


def serialize_doclang(doc: DoclingDocument, params: Optional[DoclangParams] = None) -> str:
    ser = DoclangDocSerializer(doc=doc, params=params or DoclangParams())
    return ser.serialize().text


def test_list_items_not_double_wrapped_when_no_content():
    doc = DoclingDocument(name="t")
    lst = doc.add_list_group()
    doc.add_list_item("Item A", parent=lst)
    doc.add_list_item("Item B", parent=lst)

    txt = serialize_doclang(doc, params=DoclangParams(content_types=set()))
    exp_txt = """
<doclang version="1.0.0">
  <list ordered="false">
    <list_text></list_text>
    <list_text></list_text>
  </list>
</doclang>
    """
    assert txt.strip() == exp_txt.strip()


def test_doclang():
    src = Path("./test/data/doc/ddoc_0.json")
    doc = DoclingDocument.load_from_json(src)

    # Human readable, indented and with content
    params = DoclangParams()

    ser = DoclangDocSerializer(doc=doc, params=params)
    actual = ser.serialize().text

    verify(exp_file=src.with_suffix(".v0.gt.dclg.xml"), actual=actual)

    # Human readable, indented but without content
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_types={ContentType.TABLE},
        ),
    )
    actual = ser.serialize().text

    verify(exp_file=src.with_suffix(".v1.gt.dclg.xml"), actual=actual)

    # Machine readable, not indented and without content
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            pretty_indentation=None,
            content_types={ContentType.TABLE},
        ),
    )
    actual = ser.serialize().text

    verify(exp_file=src.with_suffix(".v2.gt.dclg.xml"), actual=actual)


def test_doclang_meta():
    src = Path("./test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.EMBEDDED),
    )
    actual = ser.serialize().text
    verify(exp_file=src.with_suffix(".gt.dclg.xml"), actual=actual)


def test_doclang_crop_embedded():
    src = Path("./test/data/doc/activities_simplified.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.EMBEDDED),
    )
    actual = serializer.serialize().text

    # verifying everything except base64 data as the latter seems to be flaky across runs/platforms
    exp_prefix = """
<doclang version="1.0.0">
  <floating_group class="picture">
    <picture>
      <meta>
        <classification>Other</classification>
      </meta>
      <location value="43"/>
      <location value="117"/>
      <location value="172"/>
      <location value="208"/>
      <uri>data:image/png;base64,
    """.strip()
    assert actual.startswith(exp_prefix)

    exp_suffix = """
      </uri>
    </picture>
  </floating_group>
</doclang>
    """.strip()
    assert actual.endswith(exp_suffix)

def test_doclang_crop_placeholder():
    src = Path("./test/data/doc/activities_simplified.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.PLACEHOLDER),
    )
    actual = serializer.serialize().text
    exp_file = src.parent / f"{src.stem}_cropped_placeholder.dclg.xml"
    verify(exp_file=exp_file, actual=actual)

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
    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            escape_mode=EscapeMode.CDATA_ALWAYS,
            image_mode=ImageRefMode.EMBEDDED,
        ),
    )
    ser_res = serializer.serialize()
    ser_txt = ser_res.text

    exp_file = Path("./test/data/doc/cdata_always.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_cdata_when_needed(sample_doc: DoclingDocument):
    """Test cdata_when_needed mode."""
    doc = _create_escape_test_doc(sample_doc)
    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            escape_mode=EscapeMode.CDATA_WHEN_NEEDED,
            image_mode=ImageRefMode.EMBEDDED,
        ),
    )
    ser_res = serializer.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/cdata_when_needed.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_strikethrough_formatting():
    """Test strikethrough formatting serialization."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(strikethrough=True)
    doc.add_text(label=DocItemLabel.TEXT, text="Strike text", formatting=formatting)

    result = serialize_doclang(
        doc, params=DoclangParams(add_location=False)
    )
    assert "<strikethrough>Strike text</strikethrough>" in result


def test_subscript_formatting():
    """Test subscript formatting serialization."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(script=Script.SUB)
    doc.add_text(label=DocItemLabel.TEXT, text="H2O", formatting=formatting)

    result = serialize_doclang(
        doc, params=DoclangParams(add_location=False)
    )
    assert "<subscript>H2O</subscript>" in result


def test_superscript_formatting():
    """Test superscript formatting serialization."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(script=Script.SUPER)
    doc.add_text(label=DocItemLabel.TEXT, text="x^2", formatting=formatting)

    result = serialize_doclang(
        doc, params=DoclangParams(add_location=False)
    )
    assert "<superscript>x^2</superscript>" in result


def test_combined_formatting():
    """Test combined formatting (bold + italic)."""
    doc = DoclingDocument(name="test")
    formatting = Formatting(bold=True, italic=True)
    doc.add_text(label=DocItemLabel.TEXT, text="Bold and italic", formatting=formatting)

    result = serialize_doclang(
        doc, params=DoclangParams(add_location=False)
    )
    # When both bold and italic are applied, they should be nested
    assert "<bold>" in result
    assert "<italic>" in result
    assert "Bold and italic" in result




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


def test_handwritten_text_label(doc_with_handwritten: DoclingDocument):
    result = doc_with_handwritten.export_to_doclang()
    exp_file = Path("./test/data/doc/handwritten_text.gt.dclg.xml")
    verify(exp_file=exp_file, actual=result)


def test_content_allow_all_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_types=set(ContentType),
            image_mode=ImageRefMode.EMBEDDED,
        ),
    )
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/content_all.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_content_allow_no_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_types=set(),
            image_mode=ImageRefMode.EMBEDDED,
        ),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/content_none.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_content_allow_specific_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_types={
                ContentType.PICTURE,
                ContentType.TABLE,
                ContentType.TABLE_CELL,
                ContentType.REF_CAPTION,
                ContentType.TEXT_CODE,
            },
            image_mode=ImageRefMode.EMBEDDED,
        ),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/content_specific.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_content_block_specific_types(sample_doc: DoclingDocument):
    doc = _create_content_filtering_doc(sample_doc)
    blocked_types = {
        ContentType.TABLE,
        ContentType.TEXT_CODE,
    }
    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_types={ct for ct in ContentType if ct not in blocked_types},
            image_mode=ImageRefMode.EMBEDDED,
        ),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/content_block_specific.gt.dclg.xml")
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

    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/inline_group.gt.dclg.xml")
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
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/mini_inline.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def _create_wrapping_test_doc():
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="simple")
    doc.add_text(label=DocItemLabel.TEXT, text="  leading")
    doc.add_text(label=DocItemLabel.TEXT, text="trailing  ")
    doc.add_text(label=DocItemLabel.TEXT, text="< special")
    doc.add_text(label=DocItemLabel.TEXT, text="  leading and < special")

    doc.add_text(label=DocItemLabel.TEXT, text="w/prov simple", prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="  w/prov leading", prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="w/prov trailing  ", prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="w/prov < special", prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="  w/prov leading and < special", prov=prov)

    return doc

def test_content_wrapping_mode_when_needed():
    doc = _create_wrapping_test_doc()
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_wrapping_mode=WrapMode.WRAP_WHEN_NEEDED,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/wrapping_when_needed.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_content_wrapping_mode_always():
    doc = _create_wrapping_test_doc()
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_wrapping_mode=WrapMode.WRAP_ALWAYS,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/wrapping_always.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_default_mode():
    doc = DoclingDocument(name="test")
    add_texts_section(doc)
    add_list_section(doc)

    ser = DoclangDocSerializer(doc=doc)
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/default_mode.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_vlm_mode():
    doc = DoclingDocument(name="test")
    add_texts_section(doc)
    add_list_section(doc)

    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            pretty_indentation=None,
            escape_mode=EscapeMode.CDATA_ALWAYS,
            content_wrapping_mode=WrapMode.WRAP_ALWAYS,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/vlm_mode.gt.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_rich_cells(rich_table_doc):
    ser = DoclangDocSerializer(
        doc=rich_table_doc,
        params=DoclangParams(),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/rich_table.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def _create_simple_prov_doc():
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Hello", prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="World", prov=prov)
    return doc

def test_checkboxes():
    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text="TODO")
    doc.add_text(label=DocItemLabel.CHECKBOX_SELECTED, text="DONE")
    ser = DoclangDocSerializer(doc=doc)
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/checkboxes.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_def_prov_512():
    doc = _create_simple_prov_doc()
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            xsize=512,
            ysize=512,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/simple_prov_res_512.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_def_prov_256():
    doc = _create_simple_prov_doc()
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            xsize=256,
            ysize=256,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/simple_prov_res_256.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_chart():
    doc = DoclingDocument.load_from_json("./test/data/doc/barchart.json")
    ser = DoclangDocSerializer(
        doc=doc,
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/barchart.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def _verify_doc(doc: DoclingDocument, exp_json: Path):
    if GEN_TEST_DATA:
        doc.save_as_json(filename=exp_json)
    else:
        exp_doc = DoclingDocument.load_from_json(filename=exp_json)
        assert doc == exp_doc

def test_kv():
    doc = DoclingDocument(name="")
    kvm = doc.add_field_region()

    doc.add_field_heading(text="KV heading", parent=kvm)

    kve = doc.add_field_item(parent=kvm)
    doc.add_field_key(text="simple key", parent=kve)
    doc.add_field_value(text="simple value", parent=kve)

    doc.add_field_heading(level=2, text="KV sub-heading", parent=kvm)

    # inlined key-value pair (outer is <text>...</text>)
    # TODO: possibly support outer bounding box
    inl = doc.add_inline_group(parent=kvm)
    kve = doc.add_field_item(parent=inl)
    doc.add_field_key(text="my inline key1: ", parent=kve)
    doc.add_field_value(text="my inline value1", parent=kve, kind="fillable")

    # # inlined key-value pair (outer is <kv_entry>...</kv_entry>)
    # # TODO: possibly support outer bounding box
    # kve = doc.add_field_item(parent=kvm)
    # inl = doc.add_inline_group(parent=kve)
    # doc.add_field_key(text="my inline key2: ", parent=inl)
    # doc.add_field_value(text="my inline value2", parent=inl, kind="fillable")

    kve = doc.add_field_item(parent=kvm)
    doc.add_field_key(text="name", parent=kve)
    doc.add_field_value(text="John Doe", parent=kve, kind="fillable")
    doc.add_field_value(text="Max Mustermann", parent=kve, kind="fillable")

    kk = doc.add_field_value(text="", parent=kve, kind="fillable")
    opt_vis = doc.add_inline_group(parent=kk)
    doc.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text="", parent=opt_vis)
    doc.add_text(label=DocItemLabel.TEXT, text="Clark ", parent=opt_vis)
    doc.add_text(label=DocItemLabel.TEXT, text="Kent", parent=opt_vis, formatting=Formatting(bold=True))
    doc.add_field_hint(text="Select this if you are a Superman fan", parent=opt_vis)

    doc.add_field_value(text="", parent=kve)

    # inlined form inputs
    # TODO: add support for outer bounding box
    inl = doc.add_inline_group(parent=kve)

    doc.add_text(label=DocItemLabel.TEXT, text="My first input ", parent=inl)
    doc.add_field_value(text="", parent=inl, kind="fillable")
    doc.add_text(label=DocItemLabel.TEXT, text=" and my second input ", parent=inl)
    doc.add_field_value(text="m", parent=inl, kind="fillable")

    kv_entry_3 = doc.add_field_item(parent=kvm)
    doc.add_field_key(text="I am in the country as a: ", parent=kv_entry_3)

    vis = doc.add_field_value(text="", parent=kv_entry_3, kind="fillable")
    opt_vis = doc.add_inline_group(parent=vis)
    doc.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text="Visitor", parent=opt_vis)

    std = doc.add_field_value(text="", parent=kv_entry_3, kind="fillable")
    opt_std = doc.add_inline_group(parent=std)
    doc.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text=" Student", parent=opt_std)

    oth = doc.add_field_value(text="", parent=kv_entry_3, kind="fillable")
    opt_oth = doc.add_inline_group(parent=oth)
    doc.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text="Other (Specify)", parent=opt_oth)

    doc.add_field_value(text="", parent=kv_entry_3, kind="fillable")

    doc.add_text(label=DocItemLabel.TEXT, text="Some final stuff.")
    doc.add_text(label=DocItemLabel.TEXT, text="The end.")

    exp_json = Path("./test/data/doc/kv.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/kv.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)



def test_kv_invoice():
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    # prov = None
    image_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAASUVORK5CYII="

    # first key-value map
    kvm = doc.add_field_region(prov=prov)

    # inlined key-value pair
    kve = doc.add_field_item(parent=kvm)
    kvk = doc.add_field_key(text="", parent=kve)
    doc.add_picture(
        parent=kvk,
        image=ImageRef(
            mimetype="image/png",
            uri=image_uri,
            dpi=300,
            size=Size(width=100, height=100),
        ),
    )
    doc.add_field_value(text="+123-456-7890", parent=kve)


    # another inlined key-value pair
    kve = doc.add_field_item(parent=kvm)
    kvk = doc.add_field_key(text="", parent=kve)
    doc.add_picture(
        parent=kvk,
        image=ImageRef(
            mimetype="image/png",
            uri=image_uri,
            dpi=300,
            size=Size(width=100, height=100),
        ),
    )
    doc.add_field_value(text="hello@example.com", parent=kve)

    # second key-value map
    kvm = doc.add_field_region()

    # inlined key-value pair
    inl_outer = doc.add_inline_group(parent=kvm)
    kve = doc.add_field_item(parent=inl_outer)
    doc.add_field_key(text="Invoice No: ", parent=kve)
    doc.add_field_value(text="222", parent=kve)

    # another inlined key-value pair
    inl_outer = doc.add_inline_group(parent=kvm)
    kve = doc.add_field_item(parent=inl_outer)
    doc.add_field_key(text="Date: ", parent=kve)
    doc.add_field_value(text="02 May, 2021", parent=kve)

    # a last key-value map
    kvm = doc.add_field_region()
    kve = doc.add_field_item(parent=kvm)
    doc.add_field_key(text="Administrator", parent=kve, prov=prov)
    doc.add_field_value(text="John Doe", parent=kve, prov=prov)

    exp_json = Path("./test/data/doc/kv_invoice.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/kv_invoice.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)




def test_kv_advanced_inline():
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    prov = None
    image_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAASUVORK5CYII="

    # first key-value map
    kvm = doc.add_field_region()

    # inlined key-value pair
    inl_outer = doc.add_inline_group(parent=kvm)
    doc.add_text(label=DocItemLabel.TEXT, text="This certificate applies to ", parent=inl_outer)

    kve = doc.add_field_item(parent=inl_outer)
    doc.add_field_value(text="", parent=kve, kind="fillable")
    doc.add_text(label=DocItemLabel.TEXT, text=" percent of Buyer's purchases from ", parent=inl_outer)

    kve = doc.add_field_item(parent=inl_outer)
    doc.add_field_value(text="", parent=kve, kind="fillable")
    doc.add_text(label=DocItemLabel.TEXT, text=" (name, address, and employer idenficiation number of seller) as follows (complete as applicable): ", parent=inl_outer)

    kve = doc.add_field_item(parent=inl_outer)
    doc.add_field_value(text="", parent=kve, kind="fillable")
    doc.add_text(label=DocItemLabel.TEXT, text=".", parent=inl_outer)

    exp_json = Path("./test/data/doc/kv_advanced_inline.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/kv_advanced_inline.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_kv_nested():
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    # prov = None
    image_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAASUVORK5CYII="

    # first key-value map
    kvm = doc.add_field_region(prov=prov)

    kve = doc.add_field_item(parent=kvm)
    doc.add_field_key(text="A", parent=kve)
    kvv = doc.add_field_value(text="", parent=kve)
    kvm_inner = doc.add_field_region(parent=kvv)
    kve_inner = doc.add_field_item(parent=kvm_inner)
    doc.add_marker(text="1.", parent=kve_inner)
    doc.add_field_key(text="AA", parent=kve_inner)
    doc.add_field_hint(text="Some explanation for key AA", parent=kve_inner)
    doc.add_field_value(text="AAA", parent=kve_inner)
    doc.add_field_hint(text="Some explanation for value AAA", parent=kve_inner)
    doc.add_field_value(text="AAB", parent=kve_inner)
    doc.add_field_hint(text="Some explanation for value AAB", parent=kve_inner)
    kve_inner = doc.add_field_item(parent=kvm_inner)
    doc.add_marker(text="2.", parent=kve_inner)
    doc.add_field_key(text="AB", parent=kve_inner)
    doc.add_field_value(text="ABA", parent=kve_inner)
    doc.add_field_value(text="ABB", parent=kve_inner)

    exp_json = Path("./test/data/doc/kv_nested.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/kv_nested.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_kv_form_with_table():
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    prov = None
    image_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAASUVORK5CYII="

    # first key-value map
    kvm = doc.add_field_region()

    # table

    table_vals = [
        ["Description of property", "Cost or other basis, plus improvements and expense of sale", "Gain or loss"],
        [""  ,                      "gain",                                                       "150,997"],
        ["",                        "loss",                                                       "114,676"],
    ]
    num_rows = len(table_vals)
    num_cols = len(table_vals[0])
    table = doc.add_table(data=TableData(num_rows=num_rows, num_cols=num_cols), parent=kvm)

    for i in range(num_rows):
        for j in range(num_cols):
            if i == 0:  # headers
                cell = TableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    text=table_vals[i][j],
                    column_header=True,
                )
            else:
                kve = doc.add_field_item(parent=table)
                doc.add_field_value(text=table_vals[i][j], parent=kve, kind="fillable")
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    text="",
                    ref=kve.get_ref(),
                )
            doc.add_table_cell(table_item=table, cell=cell)

    exp_json = Path("./test/data/doc/kv_form_with_table.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    ser_txt = doc.export_to_doclang()

    exp_file = Path("./test/data/doc/kv_form_with_table.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_kv_migration_self_contained_scenario():
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Hello, world!")
    doc.add_key_values(
        graph=GraphData(
            cells=[
                # both TO_VALUE & TO_KEY links:
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=0,
                    text="Common name",
                    orig="Common name",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=1,
                    text="Duck",
                    orig="Duck",
                ),

                # TO_PARENT & TO_CHILD links:
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=2,
                    text="Anatoidea",
                    orig="Anatoidea",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=3,
                    text="Anatidae",
                    orig="Anatidae",
                ),

                # multiple TO_VALUE links:
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=4,
                    text="Distribution package",
                    orig="Distribution package",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=5,
                    text="docling",
                    orig="docling",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=6,
                    text="docling-core",
                    orig="docling-core",
                    prov=prov,
                ),
            ],
            links=[
                GraphLink(
                    label=GraphLinkLabel.TO_VALUE,
                    source_cell_id=0,
                    target_cell_id=1,
                ),
                GraphLink(label=GraphLinkLabel.TO_KEY, source_cell_id=1, target_cell_id=0),
                GraphLink(label=GraphLinkLabel.TO_CHILD, source_cell_id=2, target_cell_id=3),
                GraphLink(label=GraphLinkLabel.TO_PARENT, source_cell_id=3, target_cell_id=2),
                GraphLink(label=GraphLinkLabel.TO_VALUE, source_cell_id=4, target_cell_id=5),
                GraphLink(label=GraphLinkLabel.TO_VALUE, source_cell_id=4, target_cell_id=6),
            ],
        ),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Some more text...", parent=doc.body)
    doc.add_form(
        graph=GraphData(
            cells=[
                # both TO_VALUE & TO_KEY links:
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=0,
                    text="Color",
                    orig="Color",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=1,
                    text="Orange",
                    orig="Orange",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=2,
                    text="Black",
                    orig="Black",
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=3,
                    text="White",
                    orig="White",
                ),
            ],
            links=[
                GraphLink(label=GraphLinkLabel.TO_VALUE, source_cell_id=0, target_cell_id=1),
                GraphLink(label=GraphLinkLabel.TO_VALUE, source_cell_id=0, target_cell_id=2),
                GraphLink(label=GraphLinkLabel.TO_VALUE, source_cell_id=0, target_cell_id=3),
                GraphLink(label=GraphLinkLabel.TO_KEY, source_cell_id=3, target_cell_id=0),
            ],
        ),
        prov=prov,
    )
    doc.add_text(label=DocItemLabel.TEXT, text="The end.", parent=doc.body)

    exp_json = Path("./test/data/doc/kv_pre_migration.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    doc._migrate_to_field_regions()

    exp_json = Path("./test/data/doc/kv_post_migration.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/kv_migration.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)

def test_kv_migration_annot_scenario():
    roots = [
        "./test/data/doc/kv",
        "./test/data/doc/doclang_ref",
    ]
    for subdir in chain.from_iterable([Path(root).iterdir() for root in roots]):
        if not subdir.is_dir():
            continue
        input_json = subdir / "input.json"
        if not input_json.exists():
            continue
        doc = DoclingDocument.load_from_json(input_json)
        if GEN_TEST_DATA:
            modes = {
                # "ro": "reading_order",
                "kv": "key_value",
            }
            for mode_kw in modes:
                pages = doc.get_visualization(viz_mode=modes[mode_kw])
                for page_no, page in pages.items():
                    page.save(str(subdir / f"input_{mode_kw}_p{page_no}.png"))
        doc._migrate_to_field_regions()
        exp_json = subdir / "output.json"
        _verify_doc(doc=doc, exp_json=exp_json)
        ser_txt = doc.export_to_doclang()
        exp_file = subdir / "output.dclg.xml"
        verify(exp_file=exp_file, actual=ser_txt)

        ser = DoclangDocSerializer(
            doc=doc,
            params=DoclangParams(
                add_content=False,
            ),
        )
        ser_txt = ser.serialize().text
        exp_file = subdir / "output_no_content.dclg.xml"
        verify(exp_file=exp_file, actual=ser_txt)

        if GEN_TEST_DATA:
            modes = {
                # "ro": "reading_order",
                "kv": "key_value",
            }
            for mode_kw in modes:
                pages = doc.get_visualization(viz_mode=modes[mode_kw])
                for page_no, page in pages.items():
                    page.save(str(subdir / f"output_{mode_kw}_p{page_no}.png"))


# ===============================
# suppress_empty_elements tests
# ===============================

_SUPPRESS_PARAMS = DoclangParams(
    suppress_empty_elements=True,
    add_location=False,
    content_types=set(),  # no content → forces items empty
)


def test_suppress_empty_text_item():
    """An empty text item is omitted when suppress_empty_elements is True."""
    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.TEXT, text="")

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    # The empty <text></text> must not appear
    assert "<text" not in result
    # Document root must still be present
    assert "<doclang" in result


def test_empty_text_item_preserved_by_default():
    """Without suppress_empty_elements the empty tag pair is emitted."""
    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.TEXT, text="")

    default_params = DoclangParams(
        add_location=False,
        content_types=set(),
    )
    result = serialize_doclang(doc, params=default_params)
    assert "<text></text>" in result


def test_suppress_empty_heading():
    """An empty heading is suppressed."""
    doc = DoclingDocument(name="test")
    doc.add_heading(text="", level=2)

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<heading" not in result
    assert "</heading>" not in result


def test_suppress_empty_code():
    """An empty code block is suppressed."""
    doc = DoclingDocument(name="test")
    doc.add_code(text="")

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<code" not in result


def test_suppress_empty_picture():
    """A picture with no content, no caption, no footnotes is suppressed."""
    doc = DoclingDocument(name="test")
    doc.add_picture()

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<floating_group" not in result
    assert "<picture" not in result


def test_empty_picture_preserved_by_default():
    """Without suppress_empty_elements the empty floating group is emitted."""
    doc = DoclingDocument(name="test")
    doc.add_picture()

    default_params = DoclangParams(
        add_location=False,
        content_types=set(),
    )
    result = serialize_doclang(doc, params=default_params)
    assert "<floating_group" in result


def test_suppress_empty_table():
    """A table with no data, no caption, no footnotes is suppressed."""
    doc = DoclingDocument(name="test")
    doc.add_table(data=TableData())

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<floating_group" not in result
    assert "<otsl" not in result


def test_empty_table_preserved_by_default():
    """Without suppress_empty_elements the empty table group is emitted."""
    doc = DoclingDocument(name="test")
    doc.add_table(data=TableData())

    default_params = DoclangParams(
        add_location=False,
        content_types=set(),
    )
    result = serialize_doclang(doc, params=default_params)
    assert "<floating_group" in result


def test_suppress_empty_inline_group():
    """An inline group whose children are all empty is suppressed."""
    doc = DoclingDocument(name="test")
    inl = doc.add_inline_group()
    doc.add_text(label=DocItemLabel.TEXT, text="", parent=inl)

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    # The inline group emits <text> wrapper when unwrapped; both should vanish
    assert "<text" not in result


def test_suppress_list_with_all_empty_children():
    """A list group whose children all produce empty text is auto-suppressed.

    The list serializer already skips the <list> wrapper when no child text
    is produced, so suppressing individual empty <list_text> items causes
    the parent <list> to vanish too.
    """
    doc = DoclingDocument(name="test")
    lst = doc.add_list_group()
    doc.add_list_item(text="", parent=lst)
    doc.add_list_item(text="", parent=lst)

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<list" not in result
    assert "<list_text" not in result


def test_suppress_list_keeps_nonempty_items():
    """Only empty list items are suppressed; non-empty ones remain."""
    doc = DoclingDocument(name="test")
    lst = doc.add_list_group()
    doc.add_list_item(text="", parent=lst)
    doc.add_list_item(text="Keep me", parent=lst)
    doc.add_list_item(text="", parent=lst)

    params = DoclangParams(
        suppress_empty_elements=True,
        add_location=False,
    )
    result = serialize_doclang(doc, params=params)
    assert "<list " in result
    assert result.count("<list_text>") == 1
    assert "Keep me" in result


def test_suppress_mixed_content():
    """A document with a mix of empty and non-empty items.

    Empty items are suppressed, non-empty ones remain.
    """
    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.TEXT, text="")  # suppressed
    doc.add_text(label=DocItemLabel.TEXT, text="Visible paragraph")  # kept
    doc.add_picture()  # suppressed (empty picture)
    doc.add_heading(text="Visible Heading", level=1)  # kept
    doc.add_code(text="")  # suppressed

    params = DoclangParams(
        suppress_empty_elements=True,
        add_location=False,
    )
    result = serialize_doclang(doc, params=params)
    assert result.count("<text>") == 1
    assert "Visible paragraph" in result
    assert "<floating_group" not in result
    assert '<heading level="1">' in result
    assert "Visible Heading" in result
    assert "<code" not in result


def test_suppress_does_not_affect_nonempty():
    """Suppression flag has no effect on items that carry content."""
    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.TEXT, text="Hello")
    doc.add_heading(text="World", level=1)

    params = DoclangParams(
        suppress_empty_elements=True,
        add_location=False,
    )
    result = serialize_doclang(doc, params=params)
    assert "<text>Hello</text>" in result
    assert '<heading level="1">World</heading>' in result


def test_suppress_nested_section_with_empty_children():
    """A section containing only empty elements should still emit the section
    (sections are grouping tokens and not subject to content-level suppression),
    but all its empty children should be suppressed.
    """
    from docling_core.types.doc import GroupLabel

    doc = DoclingDocument(name="test")
    sec = doc.add_group(label=GroupLabel.SECTION, name="empty_sec")
    doc.add_text(label=DocItemLabel.TEXT, text="", parent=sec)
    doc.add_code(text="", parent=sec)

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    # Section grouping wrapper may or may not remain (depends on serializer),
    # but importantly no <text> or <code> tags appear
    assert "<text" not in result
    assert "<code" not in result


def test_suppress_empty_caption_and_footnote_on_picture():
    """A picture with an empty caption and empty footnote is suppressed when
    suppress_empty_elements is True and there is no other content.
    """
    doc = DoclingDocument(name="test")
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="")
    doc.add_picture(caption=cap)

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<floating_group" not in result
    assert "<picture" not in result


def test_suppress_empty_picture_with_nonempty_caption():
    """A picture with a non-empty caption should still be emitted even when
    suppress_empty_elements is True, because the composed_inner is non-empty.
    """
    doc = DoclingDocument(name="test")
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="My Figure")
    doc.add_picture(caption=cap)

    params = DoclangParams(
        suppress_empty_elements=True,
        add_location=False,
    )
    result = serialize_doclang(doc, params=params)
    assert "<floating_group" in result
    assert "My Figure" in result


def test_layer_minimal_mode(doc_with_layers):
    """Test MINIMAL mode omits default layer, includes non-default."""
    params = DoclangParams(layer_mode=LayerMode.MINIMAL)
    ser = DoclangDocSerializer(doc=doc_with_layers, params=params)
    ser_txt = ser.serialize().text

    exp_file = Path("./test/data/doc/layer_minimal_mode.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_layer_always_mode(doc_with_layers):
    """Test ALWAYS mode includes layer element for all items."""
    params = DoclangParams(layer_mode=LayerMode.ALWAYS)
    ser = DoclangDocSerializer(doc=doc_with_layers, params=params)
    ser_txt = ser.serialize().text

    exp_file = Path("./test/data/doc/layer_always_mode.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_layer_filter_body_only(doc_with_layers):
    """Test that layers parameter filters content to only show specified layers."""
    # Serialize with only body layer
    params = DoclangParams(
        layers={ContentLayer.BODY},
    )
    ser = DoclangDocSerializer(doc=doc_with_layers, params=params)
    ser_txt = ser.serialize().text

    exp_file = Path("./test/data/doc/layer_only_body.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)


def test_newline_to_br():

    code = """


import pytest

from docling_core.experimental.doclang import (
    ContentType,
    EscapeMode,
    DoclangDocSerializer,
    DoclangParams,
    DoclangVocabulary,
    LayerMode,
    WrapMode,
)
 """

    """Test that newlines survive serialization and deserialization roundtrip."""
    from docling_core.experimental.doclang import DoclangDeserializer
    from docling_core.types.doc import TextItem

    # Create a document with newlines
    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TEXT, text="foo\nbar")

    inl = doc.add_inline_group()
    doc.add_text(label=DocItemLabel.TEXT, text="eins\n", parent=inl)
    doc.add_text(label=DocItemLabel.TEXT, text=" zwei\n ", parent=inl)
    doc.add_text(label=DocItemLabel.TEXT, text="drei", parent=inl, formatting=Formatting(bold=True))

    doc.add_code(text=code)

    # NOTE: this particular case seems bit brittle as to how it's preserved by XML tooling
    doc.add_text(label=DocItemLabel.TEXT, text="\n")

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/newline_to_br.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)
