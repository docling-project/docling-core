"""Unit tests for Doclang create_closing_token helper."""

from pathlib import Path
from typing import Optional

import pytest

from docling_core.experimental.doclang import (
    ContentType,
    EscapeMode,
    DoclangDocSerializer,
    DoclangParams,
    DoclangVocabulary,
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
from test.test_serialization import verify


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
    # Inline with attribute
    assert (
        DoclangVocabulary.create_closing_token(token=' <inline class="code"> ')
        == "</inline>"
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

def test_kv():
    doc = DoclingDocument(name="")
    kv_map = doc.add_key_value_map()

    doc.add_kv_heading(text="First form section", parent=kv_map)

    kv_entry_1 = doc.add_kv_entry(parent=kv_map)
    doc.add_text(label=DocItemLabel.KV_KEY, text="number", parent=kv_entry_1)
    doc.add_text(label=DocItemLabel.KV_VALUE, text="1", parent=kv_entry_1)

    doc.add_kv_heading(level=2, text="Second form section", parent=kv_map)

    kv_entry_2 = doc.add_kv_entry(parent=kv_map)
    doc.add_text(label=DocItemLabel.KV_KEY, text="name", parent=kv_entry_2)

    doc.add_text(label=DocItemLabel.KV_VALUE, text="John Doe", parent=kv_entry_2)
    doc.add_text(label=DocItemLabel.KV_VALUE, text="Max Mustermann", parent=kv_entry_2)

    kk = doc.add_text(label=DocItemLabel.KV_VALUE, text="", parent=kv_entry_2)
    opt_1 = doc.add_inline_group(parent=kk)
    doc.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text="", parent=opt_1)
    doc.add_text(label=DocItemLabel.TEXT, text="Klark ", parent=opt_1)
    doc.add_text(label=DocItemLabel.TEXT, text="Kent", parent=opt_1, formatting=Formatting(bold=True))
    doc.add_text(label=DocItemLabel.KV_HINT, text="Select this if you are a Superman fan", parent=opt_1)

    doc.add_text(label=DocItemLabel.KV_VALUE, text="", parent=kv_entry_2)

    ser = DoclangDocSerializer(
        doc=doc,
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/kv.out.dclg.xml")
    verify(exp_file=exp_file, actual=ser_txt)
