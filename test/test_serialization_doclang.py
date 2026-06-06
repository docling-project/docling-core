"""Unit tests for Doclang create_closing_token helper."""

from itertools import chain
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl

from docling_core.experimental.doclang import (
    ContentType,
    EscapeMode,
    DoclangDocSerializer,
    DoclangParams,
    LabelMode,
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
from test.doclang_validation import (
    doclang_validator,
    validate_dclg_xml,
)
from test.test_data_gen_flag import GEN_TEST_DATA
from test.test_serialization import verify

INCOMPAT_DATA = Path("./test/data/doc/incompat")


def verify_doclang(exp_file: Path, actual: str) -> None:
    """Golden compare plus reference DocLang validation."""
    verify(exp_file=exp_file, actual=actual)
    if not GEN_TEST_DATA:
        validate_dclg_xml(actual)


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
# Doclang tests
# ===============================


def serialize_doclang(doc: DoclingDocument, params: Optional[DoclangParams] = None) -> str:
    ser = DoclangDocSerializer(doc=doc, params=params or DoclangParams())
    text = ser.serialize().text
    if not GEN_TEST_DATA:
        validate_dclg_xml(text)
    return text


def test_list_items_not_double_wrapped_when_no_content():
    doc = DoclingDocument(name="t")
    lst = doc.add_list_group()
    doc.add_list_item("Item A", parent=lst)
    doc.add_list_item("Item B", parent=lst)

    txt = serialize_doclang(doc, params=DoclangParams(content_types=set()))
    exp_txt = f"""
<doclang>
  <list>
    <ldiv/>
    <ldiv/>
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

    verify_doclang(exp_file=src.with_suffix(".v0.gt.dclg.xml"), actual=actual)

    # Human readable, indented but without content
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            content_types={ContentType.TABLE},
        ),
    )
    actual = ser.serialize().text

    verify_doclang(exp_file=src.with_suffix(".v1.gt.dclg.xml"), actual=actual)

    # Machine readable, not indented and without content
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            pretty_indentation=None,
            content_types={ContentType.TABLE},
        ),
    )
    actual = ser.serialize().text

    verify_doclang(exp_file=src.with_suffix(".v2.gt.dclg.xml"), actual=actual)


def test_doclang_meta():
    src = Path("./test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.EMBEDDED),
    )
    actual = ser.serialize().text
    verify_doclang(exp_file=src.with_suffix(".gt.dclg.xml"), actual=actual)


def test_doclang_crop_embedded():
    src = Path("./test/data/doc/activities_simplified.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.EMBEDDED),
    )
    actual = serializer.serialize().text

    # verifying everything except base64 data as the latter seems to be flaky across runs/platforms
    exp_prefix = f"""
<doclang>
  <picture>
    <label value="other"/>
    <location value="43"/>
    <location value="117"/>
    <location value="172"/>
    <location value="208"/>
    <src uri="data:image/png;base64,
    """.strip()
    assert actual.startswith(exp_prefix)

    exp_suffix = """
  </picture>
</doclang>
    """.strip()
    assert actual.endswith(exp_suffix)
    if not GEN_TEST_DATA:
        validate_dclg_xml(actual)


def test_doclang_crop_placeholder():
    src = Path("./test/data/doc/activities_simplified.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.PLACEHOLDER),
    )
    actual = serializer.serialize().text
    exp_file = src.parent / f"{src.stem}_cropped_placeholder.dclg.xml"
    verify_doclang(exp_file=exp_file, actual=actual)

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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=result)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

def test_default_mode():
    doc = DoclingDocument(name="test")
    add_texts_section(doc)
    add_list_section(doc)

    ser = DoclangDocSerializer(doc=doc)
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/default_mode.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)

def test_vlm_mode():
    doc = DoclingDocument(name="test")
    add_texts_section(doc)
    add_list_section(doc)
    doc.add_picture(
        image=ImageRef(
            mimetype="image/png",
            uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAASUVORK5CYII=",
            dpi=300,
            size=Size(width=100, height=100),
        ),
        caption=doc.add_text(label=DocItemLabel.CAPTION, text="""Picture
Caption"""),
    )
    doc.add_code(text="0 == 0")
    ser = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(
            pretty_indentation=None,
            escape_mode=EscapeMode.CDATA_ALWAYS,
            content_wrapping_mode=WrapMode.WRAP_ALWAYS,
            traverse_pictures=True,
            include_namespace=False,
            include_version=False,
            use_virtual_text=True,
            label_mode=LabelMode.ALWAYS,
            interpret_code_unknown_as_other=True,
        ),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/vlm_mode.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)

def test_rich_cells(rich_table_doc):
    ser = DoclangDocSerializer(
        doc=rich_table_doc,
        params=DoclangParams(),
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/rich_table.out.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

def test_chart():
    doc = DoclingDocument.load_from_json("./test/data/doc/barchart.json")
    ser = DoclangDocSerializer(
        doc=doc,
    )
    ser_res = ser.serialize()
    ser_txt = ser_res.text
    exp_file = Path("./test/data/doc/barchart.out.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)



def _create_kv_invoice_doc() -> DoclingDocument:
    """Helper to create a key-value invoice document with various field types."""
    doc = DoclingDocument(name="")
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 2),
    )
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

    return doc


def test_kv_invoice():
    doc = _create_kv_invoice_doc()

    exp_json = Path("./test/data/doc/kv_invoice.out.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.PLACEHOLDER),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/kv_invoice.out.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(image_mode=ImageRefMode.EMBEDDED),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/kv_invoice_embedded.out.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)

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
        verify_doclang(exp_file=exp_file, actual=ser_txt)

        ser = DoclangDocSerializer(
            doc=doc,
            params=DoclangParams(
                add_content=False,
            ),
        )
        ser_txt = ser.serialize().text
        exp_file = subdir / "output_no_content.dclg.xml"
        verify_doclang(exp_file=exp_file, actual=ser_txt)

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
    """An empty code block is suppressed when ``UNKNOWN`` maps to ``undefined``."""
    doc = DoclingDocument(name="test")
    doc.add_code(text="")

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<code" not in result


def test_suppress_empty_picture():
    """A picture with no content, no caption, no footnotes is suppressed."""
    doc = DoclingDocument(name="test")
    doc.add_picture()

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<group" not in result
    assert "<picture" not in result


def test_empty_picture_preserved_by_default():
    """Without suppress_empty_elements the empty picture is preserved."""
    doc = DoclingDocument(name="test")
    doc.add_picture()

    default_params = DoclangParams(
        add_location=False,
        content_types=set(),
    )
    result = serialize_doclang(doc, params=default_params)
    # After fix: empty pictures now emit <picture></picture> instead of <group></group>
    assert "<picture></picture>" in result
    assert "<group" not in result


def test_suppress_empty_table():
    """A table with no data, no caption, no footnotes is suppressed."""
    doc = DoclingDocument(name="test")
    doc.add_table(data=TableData())

    result = serialize_doclang(doc, params=_SUPPRESS_PARAMS)
    assert "<group" not in result
    assert "<otsl" not in result


def test_empty_table_preserved_by_default():
    """Without suppress_empty_elements the empty table element is preserved."""
    doc = DoclingDocument(name="test")
    doc.add_table(data=TableData())

    default_params = DoclangParams(
        add_location=False,
        content_types=set(),
    )
    result = serialize_doclang(doc, params=default_params)
    assert "<table" in result
    assert "<group" not in result

def test_document_index_serialization():
    """Test that DOCUMENT_INDEX tables are serialized as <index>."""
    doc = DoclingDocument(name="test")

    # Add a regular table
    table_data = TableData(num_cols=2)
    table_data.add_row(['Header 1', 'Header 2'])
    table_data.grid[0][0].column_header = True
    table_data.grid[0][1].column_header = True
    table_data.add_row(['Data 1', 'Data 2'])
    doc.add_table(data=table_data, label=DocItemLabel.TABLE)

    # Add a DOCUMENT_INDEX table
    index_data = TableData(num_cols=2)
    index_data.add_row(['Index 1', 'Page 1'])
    index_data.add_row(['Index 2', 'Page 2'])
    doc.add_table(data=index_data, label=DocItemLabel.DOCUMENT_INDEX)

    result = serialize_doclang(doc)

    # Verify against expected output
    exp_file = Path("./test/data/doc/document_index.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=result)


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
    assert "<ldiv" not in result


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
    assert "<list>" in result
    assert result.count("<ldiv/>") == 1
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
    assert "<group" not in result
    assert '<heading level="2">' in result
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
    assert '<heading level="2">World</heading>' in result


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
    assert "<group" not in result
    assert "<picture" not in result


def test_suppress_empty_picture_with_nonempty_caption():
    """A picture with a non-empty caption should still be emitted even when
    suppress_empty_elements is True, because the composed content is non-empty.
    """
    doc = DoclingDocument(name="test")
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="My Figure")
    doc.add_picture(caption=cap)

    params = DoclangParams(
        suppress_empty_elements=True,
        add_location=False,
    )
    result = serialize_doclang(doc, params=params)
    assert "<picture" in result
    assert "<caption" in result
    assert "My Figure" in result
    assert "<group" not in result


def test_layer_minimal_mode(doc_with_layers):
    """Test MINIMAL mode omits default layer, includes non-default."""
    params = DoclangParams(layer_mode=LayerMode.MINIMAL)
    ser = DoclangDocSerializer(doc=doc_with_layers, params=params)
    ser_txt = ser.serialize().text

    exp_file = Path("./test/data/doc/layer_minimal_mode.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_layer_always_mode(doc_with_layers):
    """Test ALWAYS mode includes layer element for all items."""
    params = DoclangParams(layer_mode=LayerMode.ALWAYS)
    ser = DoclangDocSerializer(doc=doc_with_layers, params=params)
    ser_txt = ser.serialize().text

    exp_file = Path("./test/data/doc/layer_always_mode.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_layer_filter_body_only(doc_with_layers):
    """Test that layers parameter filters content to only show specified layers."""
    # Serialize with only body layer
    params = DoclangParams(
        layers={ContentLayer.BODY},
    )
    ser = DoclangDocSerializer(doc=doc_with_layers, params=params)
    ser_txt = ser.serialize().text

    exp_file = Path("./test/data/doc/layer_only_body.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def _doc_with_labeled_code_and_pictures() -> DoclingDocument:
    doc = DoclingDocument(name="t")
    doc.add_code(text="x = 1", code_language=CodeLanguageLabel.PYTHON)
    doc.add_code(text="y = 2")
    classified = doc.add_picture()
    classified.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name=PictureClassificationLabel.OTHER.value,
                    confidence=1.0,
                )
            ]
        )
    )
    doc.add_picture()
    return doc


def test_label_mode_when_defined():
    result = serialize_doclang(
        _doc_with_labeled_code_and_pictures(),
        params=DoclangParams(label_mode=LabelMode.WHEN_DEFINED, add_location=False),
    )
    assert result.count('<label value="Python"/>') == 1
    assert result.count('<label value="other"/>') == 1
    assert '<label value="unknown"/>' not in result
    assert '<label value="undefined"/>' not in result


def test_label_mode_always():
    result = serialize_doclang(
        _doc_with_labeled_code_and_pictures(),
        params=DoclangParams(label_mode=LabelMode.ALWAYS, add_location=False),
    )
    assert result.count('<label value="Python"/>') == 1
    assert result.count('<label value="other"/>') == 1
    assert result.count('<label value="undefined"/>') == 1
    assert result.count('<label value="unknown"/>') == 1


def test_label_mode_always_empty_code_emits_undefined_by_default():
    doc = DoclingDocument(name="t")
    doc.add_code(text="")
    result = serialize_doclang(
        doc,
        params=DoclangParams(label_mode=LabelMode.ALWAYS, add_location=False),
    )
    assert '<label value="undefined"/>' in result
    assert '<label value="other"/>' not in result


def test_interpret_code_unknown_as_other_true():
    doc = DoclingDocument(name="t")
    doc.add_code(text="x = 1")
    result = serialize_doclang(
        doc,
        params=DoclangParams(
            interpret_code_unknown_as_other=True,
            label_mode=LabelMode.WHEN_DEFINED,
            add_location=False,
        ),
    )
    assert '<label value="other"/>' in result
    assert '<label value="undefined"/>' not in result


def test_interpret_code_unknown_as_other_false():
    doc = DoclingDocument(name="t")
    doc.add_code(text="x = 1")
    result = serialize_doclang(
        doc,
        params=DoclangParams(
            interpret_code_unknown_as_other=False,
            label_mode=LabelMode.ALWAYS,
            add_location=False,
        ),
    )
    assert '<label value="undefined"/>' in result
    assert '<label value="other"/>' not in result


def test_label_mode_never():
    result = serialize_doclang(
        _doc_with_labeled_code_and_pictures(),
        params=DoclangParams(label_mode=LabelMode.NEVER, add_location=False),
    )
    assert "<label" not in result


def test_label_mode_when_defined_suppresses_empty_unknown_code():
    doc = DoclingDocument(name="t")
    doc.add_code(text="")
    result = serialize_doclang(
        doc,
        params=DoclangParams(
            label_mode=LabelMode.WHEN_DEFINED,
            suppress_empty_elements=True,
            add_location=False,
        ),
    )
    assert "<code" not in result


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
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_list_item_with_code_child():
    """Test list item with empty text but CodeItem as child."""
    doc = DoclingDocument(name="test")
    lst = doc.add_list_group()

    # List item with empty text but has a code child
    li = doc.add_list_item(text="", parent=lst, marker="•")
    doc.add_code(text="print('hello')", parent=li)

    ser_txt = doc.export_to_doclang()
    exp_file = Path("./test/data/doc/list_item_with_code.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_list_item_with_code_child_and_bbox():
    """Test list item with empty text, CodeItem child, and bounding box."""
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=0, size=Size(width=100, height=100), image=None)
    lst = doc.add_list_group()

    # List item with empty text but has a code child and provenance
    li = doc.add_list_item(text="", parent=lst, marker="•")
    doc.add_code(
        text="print('hello')",
        parent=li,
        prov=ProvenanceItem(
            page_no=0,
            bbox=BoundingBox.from_tuple((100, 200, 300, 250), origin=CoordOrigin.TOPLEFT),
            charspan=(0, 0),
        ),
    )

    serializer = DoclangDocSerializer(
        doc=doc,
        params=DoclangParams(add_location=True, xsize=256, ysize=256),
    )
    ser_txt = serializer.serialize().text
    exp_file = Path("./test/data/doc/list_item_with_code_and_bbox.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def _create_virtual_text_test_doc(add_location: bool = False) -> DoclingDocument:
    """Helper to create a test document for virtual text testing.

    Args:
        add_location: If True, add provenance/location info to items.

    Returns:
        DoclingDocument with list items and table cells for testing.
    """
    doc = DoclingDocument(name="test_virtual_texts")

    # Add page if we need location
    if add_location:
        doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)

    # Add a list with various item types
    lg = doc.add_list_group()

    # Regular list item with text
    prov = None
    if add_location:
        prov = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox.from_tuple((10, 20, 30, 25), origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, 12),
        )
    doc.add_list_item(text="Regular item", parent=lg, prov=prov)

    # List item with empty text and CodeItem child
    li_with_code = doc.add_list_item(text="", parent=lg)
    doc.add_code(
        text="print('hello')",
        parent=li_with_code,
        code_language=CodeLanguageLabel.PYTHON,
    )

    # List item with text
    prov2 = None
    if add_location:
        prov2 = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox.from_tuple((10, 30, 30, 35), origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, 12),
        )
    doc.add_list_item(text="Another item", parent=lg, prov=prov2)

    # Add a table with cells (mix of regular and rich cells)
    # Add provenance to the table so cell locations can be serialized
    table_prov = None
    if add_location:
        table_prov = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox.from_tuple((2, 40, 90, 80), origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, 50),
        )
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2), prov=table_prov)

    cell: TableCell
    # Add cells to the table
    for i in range(2):
        for j in range(2):
            # Make cell (1,1) a RichTableCell with a formula
            if i == 1 and j == 1:
                formula_item = doc.add_formula(text="E=mc^2", parent=table)
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    text="",
                    ref=formula_item.get_ref(),
                )
            else:
                cell = TableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    text=f"Cell {i * 2 + j + 1}",
                    bbox=prov2.bbox if prov2 and i + j == 0 else None,
                )
            doc.add_table_cell(table_item=table, cell=cell)

    return doc


def test_virtual_text_true_no_location():
    """Test use_virtual_text=True (default) without location info."""
    doc = _create_virtual_text_test_doc(add_location=False)

    params = DoclangParams(
        add_location=False,
    )
    serializer = DoclangDocSerializer(doc=doc, params=params)
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/virtual_texts_true_no_loc.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_virtual_text_true_with_location():
    """Test use_virtual_text=True (default) with location info."""
    doc = _create_virtual_text_test_doc(add_location=True)

    params = DoclangParams(
        add_location=True,
        add_table_cell_location=True,
    )
    serializer = DoclangDocSerializer(doc=doc, params=params)
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/virtual_texts_true_with_loc.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_virtual_text_false_no_location():
    """Test use_virtual_text=False without location info."""
    doc = _create_virtual_text_test_doc(add_location=False)

    params = DoclangParams(
        use_virtual_text=False,
        add_location=False,
    )
    serializer = DoclangDocSerializer(doc=doc, params=params)
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/virtual_texts_false_no_loc.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


def test_virtual_text_false_with_location():
    """Test use_virtual_text=False with location info."""
    doc = _create_virtual_text_test_doc(add_location=True)

    params = DoclangParams(
        use_virtual_text=False,
        add_location=True,
        add_table_cell_location=True,
    )
    serializer = DoclangDocSerializer(doc=doc, params=params)
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/virtual_texts_false_with_loc.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)

def test_include_namespace_and_version():
    """Test that include_namespace and include_version parameters work."""
    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.TEXT, text="Hello world")

    params = DoclangParams(include_namespace=True, include_version=True)
    serializer = DoclangDocSerializer(doc=doc, params=params)
    ser_txt = serializer.serialize().text

    exp_file = Path("./test/data/doc/include_namespace_and_version.gt.dclg.xml")
    verify_doclang(exp_file=exp_file, actual=ser_txt)


# ===============================
# Field region / key-value auto-wrapping
# ===============================


def _default_page(doc: DoclingDocument) -> None:
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=None)


def _doc_field_region_flat_key_value() -> DoclingDocument:
    doc = DoclingDocument(name="field_region_flat_kv")
    _default_page(doc)
    region = doc.add_field_region()
    doc.add_field_key(text="Key", parent=region)
    doc.add_field_value(text="Value", parent=region, kind="read_only")
    return doc


@doclang_validator
def test_field_region_flat_key_value_wraps_field_item():
    """Key/value directly under field_region are auto-wrapped in field_item."""
    doc = _doc_field_region_flat_key_value()
    ser_txt = DoclangDocSerializer(doc=doc).serialize().text
    verify_doclang(
        exp_file=Path("./test/data/doc/field_region_flat_key_value.gt.dclg.xml"),
        actual=ser_txt,
    )


def _doc_field_region_value_only() -> DoclingDocument:
    doc = DoclingDocument(name="field_region_value_only")
    _default_page(doc)
    region = doc.add_field_region()
    doc.add_field_value(text="42", parent=region, kind="read_only")
    return doc


@doclang_validator
def test_field_region_value_only_wraps_field_item():
    """Value directly under field_region is auto-wrapped in field_item."""
    doc = _doc_field_region_value_only()
    ser_txt = DoclangDocSerializer(doc=doc).serialize().text
    verify_doclang(
        exp_file=Path("./test/data/doc/field_region_value_only.gt.dclg.xml"),
        actual=ser_txt,
    )


def _doc_field_item_outside_field_region() -> DoclingDocument:
    doc = DoclingDocument(name="field_item_outside_region")
    _default_page(doc)
    item = doc.add_field_item(parent=doc.body)
    doc.add_field_key(text="K", parent=item)
    doc.add_field_value(text="V", parent=item, kind="read_only")
    return doc


@doclang_validator
def test_field_item_without_region_wraps_field_region():
    """Orphan field_item is auto-wrapped in <field_region>."""
    doc = _doc_field_item_outside_field_region()
    ser_txt = DoclangDocSerializer(doc=doc).serialize().text
    verify_doclang(
        exp_file=Path("./test/data/doc/field_item_outside_field_region.gt.dclg.xml"),
        actual=ser_txt,
    )


# ===============================
# Hyperlinks
# ===============================


def _doc_text_with_hyperlink() -> DoclingDocument:
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="see repository",
        hyperlink=AnyUrl("https://github.com/docling-project/docling"),
    )
    return doc


@doclang_validator
def test_text_with_hyperlink():
    """Text with hyperlink emits <href uri=\"...\"/> in element head."""
    doc = _doc_text_with_hyperlink()
    ser_txt = DoclangDocSerializer(doc=doc).serialize().text
    verify_doclang(
        exp_file=Path("./test/data/doc/text_with_hyperlink.gt.dclg.xml"),
        actual=ser_txt,
    )


# ===============================
# Known spec gaps (validator-OK)
# ===============================


def _doc_multi_prov_text() -> DoclingDocument:
    """Single text item with two provenance spans (split across pages/columns)."""
    doc = DoclingDocument(name="multi_prov")
    _default_page(doc)
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox.from_tuple((0, 0, 10, 10), origin=CoordOrigin.TOPLEFT),
        charspan=(0, 5),
    )
    item = doc.add_text(label=DocItemLabel.TEXT, text="split", prov=prov)
    item.orig = "split"
    item.prov.append(
        ProvenanceItem(
            page_no=1,
            bbox=BoundingBox.from_tuple((20, 20, 30, 30), origin=CoordOrigin.TOPLEFT),
            charspan=(5, 10),
        )
    )
    return doc


@doclang_validator
def test_multi_prov_splits_text_not_thread_snapshot():
    """Spec expects one <text> with <thread thread_id=\"N\"/>; we emit two <text> fragments."""
    doc = _doc_multi_prov_text()
    ser_txt = DoclangDocSerializer(doc=doc).serialize().text
    exp_file = INCOMPAT_DATA / "multi_prov_splits_text.dclg.xml"
    verify_doclang(exp_file=exp_file, actual=ser_txt)
    if not GEN_TEST_DATA:
        assert ser_txt.count("<text>") == 2
        assert "<thread" not in ser_txt
