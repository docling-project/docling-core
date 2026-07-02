import re
from collections.abc import Callable
from pathlib import Path

import pytest

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import (
    DocLangDocSerializer,
    DocLangParams,
    LabelMode,
)
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DescriptionMetaField,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    ImageRefMode,
    MoleculeMetaField,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
    ProvenanceItem,
    RichTableCell,
    Size,
    SummaryMetaField,
    TableCell,
    TableData,
    TableItem,
    TabularChartMetaField,
)
from docling_core.types.doc.document import GroupLabel
from docling_core.types.doc.labels import CodeLanguageLabel, PictureClassificationLabel
from test.doclang_validation import assert_valid_dclg_xml, doclang_validator
from test.test_data_gen_flag import GEN_TEST_DATA
from test.test_serialization_doclang import (
    _doc_cross_column_list,
    _doc_cross_page_list,
    _doc_cross_page_paragraph,
    _doc_cross_page_table,
    _doc_multi_prov_text,
    _verify_doc,
    add_list_section,
    add_texts_section,
    verify_doclang,
)
from test.test_serialization_doctag import verify

DO_PRINT: bool = False


def _serialize(doc: DoclingDocument) -> str:
    ser = DocLangDocSerializer(
        doc=doc,
        params=DocLangParams(include_version=False),
    )
    text = ser.serialize().text
    if not GEN_TEST_DATA:
        assert_valid_dclg_xml(text)
    return text


def _deserialize(text: str, *, validate: bool = True) -> DoclingDocument:
    if validate and not GEN_TEST_DATA:
        assert_valid_dclg_xml(text)
    return DocLangDocDeserializer().deserialize_str(text)


def _add_default_page(doc: DoclingDocument):
    doc.add_page(page_no=0, size=Size(width=1000, height=1000))


def _default_prov() -> ProvenanceItem:
    return ProvenanceItem(
        page_no=0,
        bbox=BoundingBox(l=100, t=100, r=300, b=200),
        charspan=(0, 0),
    )


class _VirtualTextMixedBboxFactory:
    """Assign distinct page bboxes for the virtual-text mixed fixture."""

    def __init__(self, *, page_no: int = 0, page_w: float = 1000, page_h: float = 1000) -> None:
        self.page_no = page_no
        self.page_w = page_w
        self.page_h = page_h
        self._slot = 0

    def next_prov(self) -> ProvenanceItem:
        row, col = divmod(self._slot, 4)
        self._slot += 1
        cell_w = self.page_w / 5
        cell_h = self.page_h / 20
        left = 20 + col * cell_w
        top = self.page_h - 40 - row * cell_h
        return ProvenanceItem(
            page_no=self.page_no,
            bbox=BoundingBox.from_tuple(
                (left, top - cell_h, left + cell_w - 10, top),
                origin=CoordOrigin.BOTTOMLEFT,
            ),
            charspan=(0, 0),
        )

    def next_bbox(self) -> BoundingBox:
        return self.next_prov().bbox


def _serialize_virtual_text_mixed(doc: DoclingDocument, *, add_location: bool = True) -> str:
    ser = DocLangDocSerializer(
        doc=doc,
        params=DocLangParams(include_version=False, add_table_cell_location=True, add_location=add_location),
    )
    text = ser.serialize().text
    if not GEN_TEST_DATA:
        assert_valid_dclg_xml(text)
    return text


def test_roundtrip_text():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.TEXT, text="Hello world")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_dt = """
<doclang>
  <text>Hello world</text>
</doclang>
    """
    assert dt2.strip() == exp_dt.strip()


def test_deserialize_include_namespace_and_version():
    """Deserialize DocLang XML with namespace and version, then roundtrip."""
    exp_file = Path("./test/data/doc/deserialize_include_namespace_and_version.gt.dclg.xml")
    xml = exp_file.read_text(encoding="utf-8")

    doc = _deserialize(xml)
    assert len(doc.texts) == 1
    assert doc.texts[0].text == "Hello world"

    reserialized = (
        DocLangDocSerializer(
            doc=doc,
            params=DocLangParams(include_namespace=True, include_version=True),
        )
        .serialize()
        .text
    )
    verify_doclang(exp_file=exp_file, actual=reserialized)


def test_roundtrip_title():
    doc = DoclingDocument(name="t")
    doc.add_title(text="My Title")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_dt = """
<doclang>
  <heading>My Title</heading>
</doclang>
    """
    assert dt2.strip() == exp_dt.strip()


def test_roundtrip_heading():
    doc = DoclingDocument(name="t")
    doc.add_heading(text="Section A", level=2)
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_dt = """
<doclang>
  <heading level="3">Section A</heading>
</doclang>
    """
    assert dt2.strip() == exp_dt.strip()


def test_roundtrip_caption():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.CAPTION, text="Cap text")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_dt = """
<doclang>
  <text>Cap text</text>
</doclang>
    """
    assert dt2.strip() == exp_dt.strip()


def test_roundtrip_footnote():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.FOOTNOTE, text="Foot note")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.FOOTNOTE
    assert doc2.texts[0].text == "Foot note"


def test_roundtrip_page_header():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Header")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.PAGE_HEADER
    assert doc2.texts[0].text == "Header"


def test_roundtrip_page_footer():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text="Footer")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.PAGE_FOOTER
    assert doc2.texts[0].text == "Footer"


def test_roundtrip_code():
    doc = DoclingDocument(name="t")
    doc.add_code(text="print('hi')", code_language=CodeLanguageLabel.PYTHON)
    doc.add_code(text='disp("Hello world!")', code_language=CodeLanguageLabel.OCTAVE)
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_dt = """
<doclang>
  <code>
    <label value="Python"/>
<![CDATA[print('hi')]]>  </code>
  <code>
    <label value="MATLAB"/>
<![CDATA[disp("Hello world!")]]>  </code>
</doclang>
    """
    assert dt2.strip() == exp_dt.strip()


def test_roundtrip_code_python_and_unknown():
    doc = DoclingDocument(name="t")
    doc.add_code(text="x = 1", code_language=CodeLanguageLabel.PYTHON)
    doc.add_code(text="y = 2")  # CodeLanguageLabel.UNKNOWN by default

    xml = _serialize(doc)
    assert '<label value="Python"/>' in xml
    assert '<label value="other"/>' not in xml
    assert '<label value="undefined"/>' not in xml
    assert '<label value="unknown"/>' not in xml

    doc2 = _deserialize(xml)
    codes = [t for t in doc2.texts if t.label == DocItemLabel.CODE]
    assert len(codes) == 2
    assert codes[0].code_language == CodeLanguageLabel.PYTHON
    assert codes[1].code_language == CodeLanguageLabel.UNKNOWN
    assert codes[0].text.strip() == "x = 1"
    assert codes[1].text.strip() == "y = 2"


@pytest.mark.parametrize(
    ("docling_lang", "linguist_label", "roundtrip_lang"),
    [
        (CodeLanguageLabel.BASH, "Shell", CodeLanguageLabel.BASH),
        (CodeLanguageLabel.LATEX, "TeX", CodeLanguageLabel.LATEX),
        (CodeLanguageLabel.LISP, "Common Lisp", CodeLanguageLabel.LISP),
        (CodeLanguageLabel.OBJECTIVEC, "Objective-C", CodeLanguageLabel.OBJECTIVEC),
        (CodeLanguageLabel.SML, "Standard ML", CodeLanguageLabel.SML),
        (CodeLanguageLabel.VISUALBASIC, "Visual Basic .NET", CodeLanguageLabel.VISUALBASIC),
        (CodeLanguageLabel.OCTAVE, "MATLAB", CodeLanguageLabel.MATLAB),
        (CodeLanguageLabel.BC, "other", CodeLanguageLabel.UNKNOWN),
        (CodeLanguageLabel.DOCLANG, "XML", CodeLanguageLabel.XML),
    ],
)
def test_code_language_linguist_mapping(docling_lang, linguist_label, roundtrip_lang):
    doc = DoclingDocument(name="t")
    doc.add_code(text="snippet", code_language=docling_lang)

    xml = DocLangDocSerializer(doc=doc, params=DocLangParams(include_version=False)).serialize().text
    assert f'<label value="{linguist_label}"/>' in xml

    doc2 = _deserialize(xml)
    assert doc2.texts[0].code_language == roundtrip_lang


def test_roundtrip_code_unknown_as_other_when_enabled():
    doc = DoclingDocument(name="t")
    doc.add_code(text="y = 2")
    xml = (
        DocLangDocSerializer(
            doc=doc,
            params=DocLangParams(include_version=False, interpret_code_unknown_as_other=True),
        )
        .serialize()
        .text
    )
    assert '<label value="other"/>' in xml
    doc2 = _deserialize(xml)
    assert doc2.texts[0].code_language == CodeLanguageLabel.UNKNOWN


def test_roundtrip_picture_other_and_unknown_labels():
    doc = DoclingDocument(name="t")
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

    xml = _serialize(doc)
    assert '<label value="other"/>' in xml
    assert '<label value="unknown"/>' not in xml

    doc2 = _deserialize(xml)
    pics = [item for item, _ in doc2.iterate_items() if isinstance(item, PictureItem)]
    assert pics[0].meta.classification.get_main_prediction().class_name == "other"
    assert pics[1].meta is None or pics[1].meta.classification is None

    xml_always = (
        DocLangDocSerializer(
            doc=doc,
            params=DocLangParams(include_version=False, label_mode=LabelMode.ALWAYS, add_location=False),
        )
        .serialize()
        .text
    )
    assert xml_always.count('<label value="undefined"/>') == 1


def test_code_language_unmapped_linguist_deserializes_to_unknown():
    xml = '<doclang><code><label value="CoffeeScript"/>foo</code></doclang>'
    doc = _deserialize(xml, validate=False)
    assert doc.texts[0].code_language == CodeLanguageLabel.UNKNOWN


def test_roundtrip_formula():
    doc = DoclingDocument(name="t")
    doc.add_formula(text="E=mc^2")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.FORMULA
    assert doc2.texts[0].text == "E=mc^2"


def test_roundtrip_list_unordered():
    doc = DoclingDocument(name="t")
    lg = doc.add_list_group()
    doc.add_list_item("A", parent=lg, enumerated=False)
    doc.add_list_item("B", parent=lg, enumerated=False)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    # Ensure group created and items present
    assert len(doc2.groups) == 1
    assert len(doc2.texts) == 2
    assert [it.text for it in doc2.texts] == ["A", "B"]


def test_roundtrip_list_ordered():
    doc = DoclingDocument(name="t")
    lg = doc.add_list_group()
    doc.add_list_item("1", parent=lg, enumerated=True)
    doc.add_list_item("2", parent=lg, enumerated=True)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.groups) == 1
    assert len(doc2.texts) == 2
    assert [it.text for it in doc2.texts] == ["1", "2"]


def test_roundtrip_picture_with_caption():
    doc = DoclingDocument(name="t")
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Fig 1")
    doc.add_picture(caption=cap)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.pictures) == 1
    pic = doc2.pictures[0]
    assert len(pic.captions) == 1
    cap_item = pic.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION and cap_item.text == "Fig 1"


def test_roundtrip_table_simple():
    doc = DoclingDocument(name="t")
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])  # header row semantics not required here
    td.add_row(["C1", "C2"])  # data row
    doc.add_table(data=td)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.tables) == 1
    t2 = doc2.tables[0].data
    assert t2.num_rows == 2 and t2.num_cols == 2
    grid_texts = [[cell.text for cell in row] for row in t2.grid]
    assert grid_texts == [["H1", "H2"], ["C1", "C2"]]


def test_roundtrip_table_with_caption():
    doc = DoclingDocument(name="t")
    # Create a caption and a simple 2x2 table
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Tbl 1")
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])  # header row
    td.add_row(["C1", "C2"])  # data row
    doc.add_table(data=td, caption=cap)

    dt = _serialize(doc)
    if DO_PRINT:
        print(dt)
    doc2 = _deserialize(dt)

    # One table reconstructed with same grid
    assert len(doc2.tables) == 1
    t2 = doc2.tables[0]
    assert t2.data.num_rows == 2 and t2.data.num_cols == 2
    grid_texts = [[cell.text for cell in row] for row in t2.data.grid]
    assert grid_texts == [["H1", "H2"], ["C1", "C2"]]

    # Caption preserved and linked to the table
    assert len(t2.captions) == 1
    cap_item = t2.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION and cap_item.text == "Tbl 1"


def test_roundtrip_text_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_text(label=DocItemLabel.TEXT, text="Hello world", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\n", dt2)
        print(f"`{doc2.texts[0].text}`")
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TEXT
    assert doc2.texts[0].text == "Hello world"


def test_roundtrip_title_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_title(
        text="My Title",
        prov=_default_prov(),
    )
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_dt = """
<doclang>
  <heading>
    <location value="51"/>
    <location value="51"/>
    <location value="154"/>
    <location value="102"/>
    My Title
  </heading>
</doclang>
    """
    assert dt2.strip() == exp_dt.strip()


def test_roundtrip_heading_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_heading(text="Section A", level=2, prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    h = doc2.texts[0]
    assert h.label == DocItemLabel.SECTION_HEADER and getattr(h, "level", 0) == 2
    assert h.text == "Section A"


def test_roundtrip_caption_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_text(label=DocItemLabel.TEXT, text="Cap text", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TEXT
    assert doc2.texts[0].text == "Cap text"


def test_roundtrip_footnote_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_text(label=DocItemLabel.FOOTNOTE, text="Foot note", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.FOOTNOTE
    assert doc2.texts[0].text == "Foot note"


def test_roundtrip_page_header_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Header", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.PAGE_HEADER
    assert doc2.texts[0].text == "Header"


def test_roundtrip_page_footer_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text="Footer", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.PAGE_FOOTER
    assert doc2.texts[0].text == "Footer"


def test_roundtrip_code_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_code(text="print('hi')", code_language=CodeLanguageLabel.PYTHON, prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.CODE
    assert doc2.texts[0].text.strip() == "print('hi')"
    assert getattr(doc2.texts[0], "code_language", None) == CodeLanguageLabel.PYTHON


def test_roundtrip_formula_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    doc.add_formula(text="E=mc^2", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.FORMULA
    assert doc2.texts[0].text == "E=mc^2"


def test_roundtrip_list_unordered_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    lg = doc.add_list_group()
    prov = _default_prov()
    doc.add_list_item("A", parent=lg, prov=prov)
    doc.add_list_item("B", parent=lg, prov=prov)

    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    assert dt2.strip() == dt.strip()


def test_roundtrip_list_ordered_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    lg = doc.add_list_group()
    prov = _default_prov()
    doc.add_list_item("1", parent=lg, enumerated=True, prov=prov)
    doc.add_list_item("2", parent=lg, enumerated=True, prov=prov)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.groups) == 1
    assert len(doc2.texts) == 2
    assert [it.text for it in doc2.texts] == ["1", "2"]


def test_roundtrip_picture_with_caption_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Fig 1", prov=_default_prov())
    doc.add_picture(caption=cap, prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.pictures) == 1
    pic = doc2.pictures[0]
    assert len(pic.captions) == 1
    cap_item = pic.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION and cap_item.text == "Fig 1"


def test_roundtrip_table_simple_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])  # header row semantics not required here
    td.add_row(["C1", "C2"])  # data row
    doc.add_table(data=td, prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.tables) == 1
    t2 = doc2.tables[0].data
    assert t2.num_rows == 2 and t2.num_cols == 2
    grid_texts = [[cell.text for cell in row] for row in t2.grid]
    assert grid_texts == [["H1", "H2"], ["C1", "C2"]]


def test_roundtrip_table_with_caption_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Tbl 1", prov=_default_prov())
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])  # header row
    td.add_row(["C1", "C2"])  # data row
    doc.add_table(data=td, caption=cap, prov=_default_prov())

    dt = _serialize(doc)
    if DO_PRINT:
        print(dt)
    doc2 = _deserialize(dt)

    assert len(doc2.tables) == 1
    t2 = doc2.tables[0]
    assert t2.data.num_rows == 2 and t2.data.num_cols == 2
    grid_texts = [[cell.text for cell in row] for row in t2.data.grid]
    assert grid_texts == [["H1", "H2"], ["C1", "C2"]]

    assert len(t2.captions) == 1
    cap_item = t2.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION and cap_item.text == "Tbl 1"


def test_roundtrip_complex_table_with_caption_prov():
    doc = DoclingDocument(name="t")
    _add_default_page(doc)
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Tbl 1", prov=_default_prov())
    td = TableData(num_rows=3, num_cols=4)

    cell_0_0 = TableCell(
        row_span=1,
        col_span=1,
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        text="H 0-0",
    )
    td.table_cells.append(cell_0_0)

    cell_0_1 = TableCell(
        row_span=1,
        col_span=3,
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=1,
        end_col_offset_idx=4,
        text="H 1-4",
    )
    td.table_cells.append(cell_0_1)

    cell_1_0 = TableCell(
        row_span=2,
        col_span=1,
        start_row_offset_idx=1,
        end_row_offset_idx=3,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        text="R 1-3",
    )
    td.table_cells.append(cell_1_0)

    cell_1_1 = TableCell(
        row_span=2,
        col_span=3,
        start_row_offset_idx=1,
        end_row_offset_idx=3,
        start_col_offset_idx=1,
        end_col_offset_idx=4,
        text="R 2-2",
    )
    td.table_cells.append(cell_1_1)

    doc.add_table(data=td, caption=cap, prov=_default_prov())

    dt = _serialize(doc)
    if DO_PRINT:
        print(dt)
    doc2 = _deserialize(dt)

    assert len(doc2.tables) == 1
    t2 = doc2.tables[0]
    assert t2.data.num_rows == 3 and t2.data.num_cols == 4
    grid_texts = [[cell.text for cell in row] for row in t2.data.grid]
    assert grid_texts == [
        ["H 0-0", "H 1-4", "H 1-4", "H 1-4"],
        ["R 1-3", "R 2-2", "R 2-2", "R 2-2"],
        ["R 1-3", "R 2-2", "R 2-2", "R 2-2"],
    ]

    assert len(t2.captions) == 1
    cap_item = t2.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION and cap_item.text == "Tbl 1"


def test_roundtrip_nested_list_unordered_in_unordered():
    """Test nested unordered list within unordered list."""
    doc = DoclingDocument(name="t")
    lg_outer = doc.add_list_group()
    doc.add_list_item("Outer Item 1", parent=lg_outer, enumerated=False)

    # Create nested list
    lg_inner = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Inner Item 1.1", parent=lg_inner, enumerated=False)
    doc.add_list_item("Inner Item 1.2", parent=lg_inner, enumerated=False)

    doc.add_list_item("Outer Item 2", parent=lg_outer, enumerated=False)

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    # Should have 2 groups (outer and inner)
    assert len(doc2.groups) == 2
    # Should have 4 text items total (1 ListItem for "Outer Item 1" + 2 inner ListItems + 1 outer ListItem)
    assert len(doc2.texts) == 4
    # Verify text content
    text_contents = [it.text for it in doc2.texts]
    assert "Outer Item 1" in text_contents
    assert "Inner Item 1.1" in text_contents
    assert "Inner Item 1.2" in text_contents
    assert "Outer Item 2" in text_contents

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def test_roundtrip_nested_list_ordered_in_ordered():
    """Test nested ordered list within ordered list."""
    doc = DoclingDocument(name="t")
    lg_outer = doc.add_list_group()
    doc.add_list_item("Step 1", parent=lg_outer, enumerated=False)

    # Create nested ordered list
    lg_inner = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Step 1.1", parent=lg_inner, enumerated=True)
    doc.add_list_item("Step 1.2", parent=lg_inner, enumerated=True)

    doc.add_list_item("Step 2", parent=lg_outer, enumerated=True)

    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    assert dt2.strip() == dt.strip()


def test_roundtrip_nested_list_ordered_in_unordered():
    """Test nested ordered list within unordered list."""
    doc = DoclingDocument(name="t")
    lg_outer = doc.add_list_group()
    doc.add_list_item("Bullet A", parent=lg_outer, enumerated=False)

    # Create nested ordered list
    lg_inner = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Numbered 1", parent=lg_inner, enumerated=True)
    doc.add_list_item("Numbered 2", parent=lg_inner, enumerated=True)

    doc.add_list_item("Bullet B", parent=lg_outer, enumerated=False)

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    assert len(doc2.groups) == 2
    assert len(doc2.texts) == 4
    text_contents = [it.text for it in doc2.texts]
    assert "Bullet A" in text_contents
    assert "Numbered 1" in text_contents
    assert "Numbered 2" in text_contents
    assert "Bullet B" in text_contents

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def test_roundtrip_nested_list_unordered_in_ordered():
    """Test nested unordered list within ordered list."""
    doc = DoclingDocument(name="t")
    lg_outer = doc.add_list_group()
    doc.add_list_item("Step 1", parent=lg_outer, enumerated=True)

    # Create nested unordered list
    lg_inner = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Bullet point", parent=lg_inner, enumerated=False)
    doc.add_list_item("Another bullet", parent=lg_inner, enumerated=False)

    doc.add_list_item("Step 2", parent=lg_outer, enumerated=True)

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    assert len(doc2.groups) == 2
    assert len(doc2.texts) == 4
    text_contents = [it.text for it in doc2.texts]
    assert "Step 1" in text_contents
    assert "Bullet point" in text_contents
    assert "Another bullet" in text_contents
    assert "Step 2" in text_contents

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def test_roundtrip_deeply_nested_list():
    """Test deeply nested lists (3 levels)."""
    doc = DoclingDocument(name="t")

    # Level 1
    lg_level1 = doc.add_list_group()
    doc.add_list_item("Level 1 Item 1", parent=lg_level1, enumerated=False)

    # Level 2 (nested in level 1)
    lg_level2 = doc.add_list_group(parent=lg_level1)
    doc.add_list_item("Level 2 Item 1", parent=lg_level2, enumerated=False)

    # Level 3 (nested in level 2)
    lg_level3 = doc.add_list_group(parent=lg_level2)
    doc.add_list_item("Level 3 Item 1", parent=lg_level3, enumerated=False)
    doc.add_list_item("Level 3 Item 2", parent=lg_level3, enumerated=False)

    doc.add_list_item("Level 2 Item 2", parent=lg_level2, enumerated=False)

    doc.add_list_item("Level 1 Item 2", parent=lg_level1, enumerated=False)

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    # Should have 3 groups (3 levels)
    assert len(doc2.groups) == 3
    # Should have 6 text items total
    assert len(doc2.texts) == 6
    text_contents = [it.text for it in doc2.texts]
    assert "Level 1 Item 1" in text_contents
    assert "Level 2 Item 1" in text_contents
    assert "Level 3 Item 1" in text_contents
    assert "Level 3 Item 2" in text_contents
    assert "Level 2 Item 2" in text_contents
    assert "Level 1 Item 2" in text_contents

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def test_roundtrip_multiple_nested_lists_same_level():
    """Test multiple nested lists at the same level."""
    doc = DoclingDocument(name="t")

    lg_outer = doc.add_list_group()
    doc.add_list_item("Item 1", parent=lg_outer, enumerated=False)

    # First nested list
    lg_inner1 = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Nested 1.1", parent=lg_inner1, enumerated=False)
    doc.add_list_item("Nested 1.2", parent=lg_inner1, enumerated=False)

    doc.add_list_item("Item 2", parent=lg_outer, enumerated=False)

    # Second nested list
    lg_inner2 = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Nested 2.1", parent=lg_inner2, enumerated=False)
    doc.add_list_item("Nested 2.2", parent=lg_inner2, enumerated=False)

    doc.add_list_item("Item 3", parent=lg_outer, enumerated=False)

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    # Should have 3 groups (1 outer, 2 inner)
    assert len(doc2.groups) == 3
    # Should have 7 text items total
    assert len(doc2.texts) == 7
    text_contents = [it.text for it in doc2.texts]
    assert "Item 1" in text_contents
    assert "Nested 1.1" in text_contents
    assert "Nested 1.2" in text_contents
    assert "Item 2" in text_contents
    assert "Nested 2.1" in text_contents
    assert "Nested 2.2" in text_contents
    assert "Item 3" in text_contents

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def test_roundtrip_list_item_with_inline_group():
    doc = DoclingDocument(name="t")

    add_texts_section(doc)
    add_list_section(doc)

    dt = _serialize(doc)
    exp_ser_file = Path(__file__).parent / "data" / "doc" / "roundtrip_list_item_with_inline_serialized.dclg.xml"
    verify(exp_ser_file, dt)
    doc2 = _deserialize(dt)
    deser_yaml_file = Path(__file__).parent / "data" / "doc" / "roundtrip_list_item_with_inline_deserialized.yaml"
    doc2.save_as_yaml(deser_yaml_file)
    dt2 = _serialize(doc2)

    exp_dt2_file = Path(__file__).parent / "data" / "doc" / "roundtrip_list_item_with_inline_reserialized.dclg.xml"
    verify(exp_dt2_file, dt2)


def test_deserialize_bare_picture():
    dt = "<doclang><picture></picture></doclang>"
    doc = _deserialize(dt)

    assert len(doc.pictures) == 1


def test_deserialize_bare_table():
    dt = "<doclang><table></table></doclang>"
    doc = _deserialize(dt)

    assert len(doc.tables) == 1
    assert doc.tables[0].data.num_rows == 0
    assert doc.tables[0].data.num_cols == 0


def test_roundtrip_table_with_caption_and_footnotes():
    """Test table with caption and multiple footnotes."""
    doc = DoclingDocument(name="t")

    # Create caption and footnotes
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1: Sample Data")

    # Create a simple 2x2 table
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["Header 1", "Header 2"])
    td.add_row(["Data 1", "Data 2"])

    # Add table with caption
    table = doc.add_table(data=td, caption=cap)

    footnote1 = doc.add_text(label=DocItemLabel.FOOTNOTE, text="First footnote")
    footnote2 = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Second footnote")
    footnote3 = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Third footnote")

    # Add footnotes to the table
    table.footnotes.append(footnote1.get_ref())
    table.footnotes.append(footnote2.get_ref())
    table.footnotes.append(footnote3.get_ref())

    # Serialize and deserialize
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt2:", dt2)

    # Verify table structure
    assert len(doc2.tables) == 1
    t2 = doc2.tables[0]
    assert t2.data.num_rows == 2 and t2.data.num_cols == 2
    grid_texts = [[cell.text for cell in row] for row in t2.data.grid]
    assert grid_texts == [["Header 1", "Header 2"], ["Data 1", "Data 2"]]

    # Verify caption
    assert len(t2.captions) == 1
    cap_item = t2.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION
    assert cap_item.text == "Table 1: Sample Data"

    # Verify footnotes
    assert len(t2.footnotes) == 3
    footnote_texts = [fn.resolve(doc2).text for fn in t2.footnotes]
    assert footnote_texts == ["First footnote", "Second footnote", "Third footnote"]

    # Verify all footnotes have the correct label
    for fn_ref in t2.footnotes:
        fn_item = fn_ref.resolve(doc2)
        assert fn_item.label == DocItemLabel.FOOTNOTE

    assert dt2 == dt


def test_roundtrip_picture_with_caption_and_footnotes():
    """Test picture with caption and multiple footnotes."""
    doc = DoclingDocument(name="t")

    # Create caption and footnotes
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Figure 1: Sample Image")
    footnote1 = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Image source: Dataset A")
    footnote2 = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Resolution: 1024x768")

    # Add picture with caption
    picture = doc.add_picture(caption=cap)

    # Add footnotes to the picture
    picture.footnotes.append(footnote1.get_ref())
    picture.footnotes.append(footnote2.get_ref())

    # Serialize and deserialize
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    # Verify picture exists
    assert len(doc2.pictures) == 1
    pic = doc2.pictures[0]

    # Verify caption
    assert len(pic.captions) == 1
    cap_item = pic.captions[0].resolve(doc2)
    assert cap_item.label == DocItemLabel.CAPTION
    assert cap_item.text == "Figure 1: Sample Image"

    # Verify footnotes
    assert len(pic.footnotes) == 2
    footnote_texts = [fn.resolve(doc2).text for fn in pic.footnotes]
    assert footnote_texts == ["Image source: Dataset A", "Resolution: 1024x768"]

    # Verify all footnotes have the correct label
    for fn_ref in pic.footnotes:
        fn_item = fn_ref.resolve(doc2)
        assert fn_item.label == DocItemLabel.FOOTNOTE

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def test_roundtrip_table_with_rich_cells():
    """Test table with RichTableCells containing paragraphs, lists, and nested tables."""
    doc = DoclingDocument(name="t")

    # Create the main table (3x2 grid)
    table = doc.add_table(data=TableData(num_rows=3, num_cols=2))

    # Create content for rich cells
    # Cell (0, 0): Multiple paragraphs
    para1 = doc.add_text(parent=table, label=DocItemLabel.TEXT, text="First paragraph in cell.")
    doc.add_text(parent=table, label=DocItemLabel.TEXT, text="Second paragraph in cell.")

    # Cell (1, 0): A list
    list_group = doc.add_list_group(parent=table)
    doc.add_list_item(parent=list_group, text="List item 1", enumerated=False)
    doc.add_list_item(parent=list_group, text="List item 2", enumerated=False)
    doc.add_list_item(parent=list_group, text="List item 3", enumerated=False)

    # Cell (2, 1): A nested table
    nested_table_data = TableData(num_rows=0, num_cols=2)
    nested_table_data.add_row(["A1", "A2"])
    nested_table_data.add_row(["B1", "B2"])
    nested_table = doc.add_table(data=nested_table_data, parent=table)

    # Create the table cells
    # Row 0: Rich cell with paragraphs in (0,0), simple cell in (0,1)
    rich_cell_0_0 = RichTableCell(
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        ref=para1.get_ref(),  # Note: In a real scenario, we'd want a group containing both paragraphs
        text="cell 0,0",
    )
    doc.add_table_cell(table_item=table, cell=rich_cell_0_0)

    simple_cell_0_1 = TableCell(
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=1,
        end_col_offset_idx=1,
        text="Simple cell 0,1",
    )
    doc.add_table_cell(table_item=table, cell=simple_cell_0_1)

    # Row 1: Rich cell with list in (1,0), simple cell in (1,1)
    rich_cell_1_0 = RichTableCell(
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        ref=list_group.get_ref(),
        text="cell 1,0",
    )
    doc.add_table_cell(table_item=table, cell=rich_cell_1_0)

    simple_cell_1_1 = TableCell(
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=1,
        end_col_offset_idx=1,
        text="Simple cell 1,1",
    )
    doc.add_table_cell(table_item=table, cell=simple_cell_1_1)

    # Row 2: Simple cell in (2,0), rich cell with nested table in (2,1)
    simple_cell_2_0 = TableCell(
        start_row_offset_idx=2,
        end_row_offset_idx=3,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        text="Simple cell 2,0",
    )
    doc.add_table_cell(table_item=table, cell=simple_cell_2_0)

    rich_cell_2_1 = RichTableCell(
        start_row_offset_idx=2,
        end_row_offset_idx=3,
        start_col_offset_idx=1,
        end_col_offset_idx=1,
        ref=nested_table.get_ref(),
        text="cell 2,1",
    )
    doc.add_table_cell(table_item=table, cell=rich_cell_2_1)

    # Serialize and deserialize
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    # Verify main table structure
    assert len(doc2.tables) >= 1
    main_table = doc2.tables[0]
    assert main_table.data.num_rows == 3
    assert main_table.data.num_cols == 2

    # Verify that we have rich table cells
    rich_cells = [cell for cell in main_table.data.table_cells if isinstance(cell, RichTableCell)]
    assert len(rich_cells) >= 1  # At least one rich cell should be preserved

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


def _create_virtual_text_list_doc() -> DoclingDocument:
    """Document exercising virtual vs explicit ``<text>`` in list items."""
    doc = DoclingDocument(name="virtual_text_list")
    _add_default_page(doc)
    bbox = _VirtualTextMixedBboxFactory()

    lg = doc.add_list_group()
    doc.add_list_item(text="plain list item", parent=lg, prov=bbox.next_prov())

    li_inline = doc.add_list_item(text="", parent=lg, prov=bbox.next_prov())
    inline = doc.add_inline_group(parent=li_inline)
    doc.add_text(label=DocItemLabel.TEXT, text="this is a ", parent=inline)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="bold",
        parent=inline,
        formatting=Formatting(bold=True),
    )
    doc.add_text(label=DocItemLabel.TEXT, text=" text", parent=inline)

    li_nested = doc.add_list_item(
        text="list item with nested list",
        parent=lg,
        prov=bbox.next_prov(),
    )
    lg_sub = doc.add_list_group(parent=li_nested)
    doc.add_list_item(text="nested list item", parent=lg_sub, prov=bbox.next_prov())

    li_picture = doc.add_list_item(
        text="list item with picture",
        parent=lg,
        prov=bbox.next_prov(),
    )
    doc.add_picture(parent=li_picture, prov=bbox.next_prov())

    return doc


def _assert_virtual_text_list_dclg(dclg: str) -> None:
    """Sanity-check virtual vs explicit ``<text>`` in list output."""
    assert "<ldiv/>" in dclg
    assert "plain list item" in dclg
    assert "<text>plain list item</text>" not in dclg
    assert "<text>" in dclg
    assert "this is a " in dclg
    assert "<bold>bold</bold>" in dclg
    assert " text" in dclg
    assert re.search(
        r"<ldiv/>\s*(?:<location[^>]*/>\s*)*<content>this is a </content>\s*<bold>bold</bold>\s*<content> text</content>",
        dclg,
    )
    assert not re.search(
        r"<text>\s*(?:<location[^>]*/>\s*)*<content>this is a </content>\s*<bold>bold</bold>",
        dclg,
    )
    assert "list item with nested list" in dclg
    assert "list item with picture" in dclg
    assert re.search(
        r"<text>\s*(?:<location[^>]*/>\s*)*list item with picture\s*</text>",
        dclg,
    )
    assert "<picture" in dclg
    assert "<location value=" in dclg


@doclang_validator
def test_virtual_text_list_roundtrip():
    """Round-trip list virtual-text edge cases through DocLang."""
    data_dir = Path(__file__).parent / "data" / "doc" / "virtual_text_list"
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = _create_virtual_text_list_doc()
    _verify_doc(doc=doc, exp_json=input_json)

    dt = _serialize_virtual_text_mixed(doc)
    verify(serialized_dclg, dt)
    _assert_virtual_text_list_dclg(dt)
    assert_valid_dclg_xml(dt)

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize_virtual_text_mixed(doc2, add_location=True)
    verify(reserialized_dclg, dt2)
    _assert_virtual_text_list_dclg(dt2)
    assert_valid_dclg_xml(dt2)


@doclang_validator
def test_virtual_text_table_roundtrip():
    """Round-trip table virtual-text edge cases through DocLang."""
    data_dir = Path(__file__).parent / "data" / "doc" / "virtual_text_table"
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = _create_virtual_text_table_doc()
    _verify_doc(doc=doc, exp_json=input_json)

    dt = _serialize_virtual_text_mixed(doc)
    verify(serialized_dclg, dt)
    _assert_virtual_text_table_dclg(dt)
    assert_valid_dclg_xml(dt)

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize_virtual_text_mixed(doc2, add_location=True)
    verify(reserialized_dclg, dt2)
    _assert_virtual_text_table_dclg(dt2)
    assert_valid_dclg_xml(dt2)


@doclang_validator
def test_virtual_text_index_roundtrip():
    """Round-trip index virtual-text edge cases through DocLang."""
    data_dir = Path(__file__).parent / "data" / "doc" / "virtual_text_index"
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = _create_virtual_text_index_doc()
    _verify_doc(doc=doc, exp_json=input_json)

    dt = _serialize_virtual_text_mixed(doc)
    verify(serialized_dclg, dt)
    _assert_virtual_text_index_dclg(dt)
    assert_valid_dclg_xml(dt)

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize_virtual_text_mixed(doc2, add_location=True)
    verify(reserialized_dclg, dt2)
    _assert_virtual_text_index_dclg(dt2)
    assert_valid_dclg_xml(dt2)


@doclang_validator
def test_multi_page_roundtrip():
    """Round-trip a programmatic multi-page document through DocLang.

    Page 1: title; page 2: document index; page 3: three text paragraphs.
    Materializes serialized/reserialized XML and input/deserialized JSON goldens.
    """
    data_dir = Path(__file__).parent / "data" / "doc" / "multi_page_roundtrip"
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = _create_multi_page_roundtrip_doc()
    _verify_doc(doc=doc, exp_json=input_json)

    dt = _serialize(doc)
    verify(serialized_dclg, dt)
    assert_valid_dclg_xml(dt)
    assert dt.count("<page_break") == 2
    assert "<index>" in dt

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize(doc2)
    verify(reserialized_dclg, dt2)
    assert_valid_dclg_xml(dt2)


def _create_multi_page_roundtrip_doc() -> DoclingDocument:
    """Build a three-page document with title, index, and body paragraphs."""
    page_size = Size(width=512, height=512)
    doc = DoclingDocument(name="multi_page_roundtrip")
    for page_no in (1, 2, 3):
        doc.add_page(page_no=page_no, size=page_size, image=None)

    doc.add_title(
        text="Document Title",
        prov=_page_prov(page_no=1, bbox=(10, 10, 200, 40)),
    )

    index_data = TableData(num_cols=2)
    index_data.add_row(["Chapter", "Page"])
    index_data.add_row(["Intro", "1"])
    index_data.add_row(["Body", "3"])
    doc.add_table(
        data=index_data,
        label=DocItemLabel.DOCUMENT_INDEX,
        prov=_page_prov(page_no=2, bbox=(10, 10, 400, 120)),
    )

    for i, text in enumerate(["First paragraph.", "Second paragraph.", "Third paragraph."], start=1):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=_page_prov(page_no=3, bbox=(10, 40 + i * 30, 400, 60 + i * 30)),
        )
    return doc


def _page_prov(*, page_no: int, bbox: tuple[float, float, float, float]) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox.from_tuple(bbox, origin=CoordOrigin.TOPLEFT),
        charspan=(0, 0),
    )


def _create_virtual_text_table_doc() -> DoclingDocument:
    """Document exercising virtual vs explicit ``<text>`` in table cells."""
    doc = DoclingDocument(name="virtual_text_table")
    _add_default_page(doc)
    bbox = _VirtualTextMixedBboxFactory()

    def _add_row_table(*, label: DocItemLabel) -> None:
        table = doc.add_table(data=TableData(num_rows=1, num_cols=3), label=label, prov=bbox.next_prov())

        cell_group = doc.add_group(parent=table, label=GroupLabel.UNSPECIFIED)
        doc.add_text(
            label=DocItemLabel.TEXT,
            text="cell with list:",
            parent=cell_group,
            prov=bbox.next_prov(),
        )
        cell_list = doc.add_list_group(parent=cell_group)
        doc.add_list_item(text="nested in cell", parent=cell_list, prov=bbox.next_prov())

        rich_group = doc.add_group(parent=table, label=GroupLabel.UNSPECIFIED)
        doc.add_text(
            label=DocItemLabel.TEXT,
            text="group text",
            parent=rich_group,
            prov=bbox.next_prov(),
        )
        doc.add_picture(parent=rich_group, prov=bbox.next_prov())

        doc.add_table_cell(
            table_item=table,
            cell=TableCell(
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                text="plain cell",
                bbox=bbox.next_bbox(),
            ),
        )
        doc.add_table_cell(
            table_item=table,
            cell=RichTableCell(
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=1,
                end_col_offset_idx=2,
                ref=cell_group.get_ref(),
                text="cell with list:",
                bbox=bbox.next_bbox(),
            ),
        )
        doc.add_table_cell(
            table_item=table,
            cell=RichTableCell(
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=2,
                end_col_offset_idx=3,
                ref=rich_group.get_ref(),
                text="group cell",
                bbox=bbox.next_bbox(),
            ),
        )

    _add_row_table(label=DocItemLabel.TABLE)
    return doc


def _create_virtual_text_index_doc() -> DoclingDocument:
    """Document exercising virtual vs explicit ``<text>`` in index cells."""
    doc = DoclingDocument(name="virtual_text_index")
    _add_default_page(doc)
    bbox = _VirtualTextMixedBboxFactory()

    table = doc.add_table(
        data=TableData(num_rows=1, num_cols=3),
        label=DocItemLabel.DOCUMENT_INDEX,
        prov=bbox.next_prov(),
    )

    cell_group = doc.add_group(parent=table, label=GroupLabel.UNSPECIFIED)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="cell with list:",
        parent=cell_group,
        prov=bbox.next_prov(),
    )
    cell_list = doc.add_list_group(parent=cell_group)
    doc.add_list_item(text="nested in cell", parent=cell_list, prov=bbox.next_prov())

    rich_group = doc.add_group(parent=table, label=GroupLabel.UNSPECIFIED)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="group text",
        parent=rich_group,
        prov=bbox.next_prov(),
    )
    doc.add_picture(parent=rich_group, prov=bbox.next_prov())

    doc.add_table_cell(
        table_item=table,
        cell=TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="plain cell",
            bbox=bbox.next_bbox(),
        ),
    )
    doc.add_table_cell(
        table_item=table,
        cell=RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            ref=cell_group.get_ref(),
            text="cell with list:",
            bbox=bbox.next_bbox(),
        ),
    )
    doc.add_table_cell(
        table_item=table,
        cell=RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            ref=rich_group.get_ref(),
            text="group cell",
            bbox=bbox.next_bbox(),
        ),
    )

    return doc


def _assert_virtual_text_table_dclg(dclg: str) -> None:
    host_block = dclg.split("<table>", 1)[1].split("</table>", 1)[0]
    assert "plain cell" in host_block
    assert "<text>plain cell</text>" not in host_block
    assert "cell with list:" in host_block
    assert "<text>" in host_block
    assert "group text" in host_block
    assert "<picture" in host_block
    assert "<location value=" in host_block


def _assert_virtual_text_index_dclg(dclg: str) -> None:
    host_block = dclg.split("<index>", 1)[1].split("</index>", 1)[0]
    assert "plain cell" in host_block
    assert "<text>plain cell</text>" not in host_block
    assert "cell with list:" in host_block
    assert "<text>" in host_block
    assert "group text" in host_block
    assert "<picture" in host_block
    assert "<location value=" in host_block


def _create_referenced_caption_doc() -> DoclingDocument:
    """Document with picture and table, each with an associated caption."""
    doc = DoclingDocument(name="referenced_caption")
    _add_default_page(doc)
    bbox = _VirtualTextMixedBboxFactory()

    cap_pic = doc.add_text(
        label=DocItemLabel.CAPTION,
        text="Figure 1",
        prov=bbox.next_prov(),
    )
    doc.add_picture(caption=cap_pic, prov=bbox.next_prov())

    cap_tbl = doc.add_text(
        label=DocItemLabel.CAPTION,
        text="Table 1",
        prov=bbox.next_prov(),
    )
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])
    td.add_row(["C1", "C2"])
    doc.add_table(data=td, caption=cap_tbl, prov=bbox.next_prov())

    return doc


def _assert_referenced_caption_doc(doc: DoclingDocument) -> None:
    """Verify picture/table caption refs are preserved in the document model."""
    assert len(doc.pictures) == 1
    pic = doc.pictures[0]
    assert len(pic.captions) == 1
    pic_cap = pic.captions[0].resolve(doc)
    assert pic_cap.label == DocItemLabel.CAPTION
    assert pic_cap.text == "Figure 1"

    assert len(doc.tables) == 1
    tbl = doc.tables[0]
    assert len(tbl.captions) == 1
    tbl_cap = tbl.captions[0].resolve(doc)
    assert tbl_cap.label == DocItemLabel.CAPTION
    assert tbl_cap.text == "Table 1"
    grid_texts = [[cell.text for cell in row] for row in tbl.data.grid]
    assert grid_texts == [["H1", "H2"], ["C1", "C2"]]


def _assert_referenced_caption_dclg(dclg: str, *, with_location: bool) -> None:
    """Sanity-check referenced captions appear in host element heads."""
    pic_block = dclg.split("<picture>", 1)[1].split("</picture>", 1)[0]
    assert "Figure 1" in pic_block
    assert re.search(r"<caption>\s*(?:<location[^>]*/>\s*)*Figure 1\s*</caption>", pic_block)

    tbl_block = dclg.split("<table>", 1)[1].split("</table>", 1)[0]
    assert "Table 1" in tbl_block
    assert re.search(r"<caption>\s*(?:<location[^>]*/>\s*)*Table 1\s*</caption>", tbl_block)
    assert "H1" in tbl_block and "C2" in tbl_block

    if with_location:
        assert "<location value=" in dclg


@doclang_validator
def test_referenced_caption_roundtrip():
    """Round-trip picture/table with associated captions through DocLang."""
    data_dir = Path(__file__).parent / "data" / "doc" / "referenced_caption"
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = _create_referenced_caption_doc()
    _verify_doc(doc=doc, exp_json=input_json)
    _assert_referenced_caption_doc(doc)

    dt = _serialize_virtual_text_mixed(doc)
    verify(serialized_dclg, dt)
    _assert_referenced_caption_dclg(dt, with_location=True)
    assert_valid_dclg_xml(dt)

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)
    _assert_referenced_caption_doc(doc2)

    dt2 = _serialize_virtual_text_mixed(doc2, add_location=True)
    verify(reserialized_dclg, dt2)
    _assert_referenced_caption_dclg(dt2, with_location=True)
    assert_valid_dclg_xml(dt2)


############################################
### Feature complete document test-cases ###
############################################


def test_constructed_doc(sample_doc: DoclingDocument):
    doc = sample_doc

    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    dt2 = _serialize(doc2)

    exp_reserialized_dt_file = Path(__file__).parent / "data" / "doc" / "constr_doc_reserialized.dclg.xml"
    verify(exp_reserialized_dt_file, dt2)


def test_constructed_rich_table_doc(rich_table_doc: DoclingDocument):
    doc = rich_table_doc

    dt = _serialize(doc)

    doc2 = _deserialize(dt)

    dt2 = _serialize(doc2)

    assert dt2 == dt


def test_wrapping():
    dt = """
<doclang>
  <text>simple</text>
  <text>
    <content>  leading</content>
  </text>
  <text>
    <content>trailing  </content>
  </text>
  <text><![CDATA[< special]]></text>
  <text>
    <content><![CDATA[  leading and < special]]></content>
  </text>
  <text>
    <location value="5"/>
    <location value="492"/>
    <location value="15"/>
    <location value="502"/>
    w/prov simple
  </text>
  <text>
    <location value="5"/>
    <location value="492"/>
    <location value="15"/>
    <location value="502"/>
    <content>  w/prov leading</content>
  </text>
  <text>
    <location value="5"/>
    <location value="492"/>
    <location value="15"/>
    <location value="502"/>
    <content>w/prov trailing  </content>
  </text>
  <text>
    <location value="5"/>
    <location value="492"/>
    <location value="15"/>
    <location value="502"/>
<![CDATA[w/prov < special]]>  </text>
  <text>
    <location value="5"/>
    <location value="492"/>
    <location value="15"/>
    <location value="502"/>
    <content><![CDATA[  w/prov leading and < special]]></content>
  </text>
</doclang>
    """
    doc = _deserialize(dt)
    dt2 = _serialize(doc)
    assert dt2.strip() == dt.strip()


def test_rich_table_cells():
    dt = """
<doclang>
  <table>
    <fcel/>
    <text>foo</text>
    <fcel/>
    <text>
      <italic>text in italic</italic>
    </text>
    <nl/>
    <fcel/>
    <table>
      <fcel/>
      <text>inner cell 0,0</text>
      <fcel/>
      <text>inner cell 0,1</text>
      <nl/>
      <fcel/>
      <text>inner cell 1,0</text>
      <fcel/>
      <text>
        <content>inner cell 1,1 </content>
        <bold>in bold</bold>
      </text>
      <nl/>
    </table>
    <fcel/>
    <text>bar</text>
    <nl/>
  </table>
</doclang>
"""
    doc = _deserialize(dt)
    dt2 = _serialize(doc)
    assert dt2.strip() == dt.strip()


def test_picture_tabular_chart_content_cdata_cells():
    """Deserializer must extract text from <content><![CDATA[...]]></content> in OTSL cells."""
    doclang = """<doclang><group><picture class="chart"><location value="0"/><location value="0"/><location value="511"/><location value="511"/><tabular><fcel/><content><![CDATA[Characteristic]]></content><fcel/><content><![CDATA[Player expenses in million U.S. dollars]]></content><nl/><fcel/><content><![CDATA[19/20]]></content><fcel/><content><![CDATA[111]]></content><nl/></tabular></picture></group></doclang>"""
    doc = _deserialize(doclang)
    first_cell_text = doc.pictures[0].meta.tabular_chart.chart_data.grid[0][0].text
    assert first_cell_text == "Characteristic"
    assert doc.pictures[0].meta.tabular_chart.chart_data.grid[0][1].text == "Player expenses in million U.S. dollars"
    assert doc.pictures[0].meta.tabular_chart.chart_data.grid[1][0].text == "19/20"
    assert doc.pictures[0].meta.tabular_chart.chart_data.grid[1][1].text == "111"


def test_picture_body_table_is_semantic_content_not_chart_tabular():
    """``<table>`` after the preamble is nested picture content, not ``meta.tabular_chart``."""
    doclang = (
        '<doclang><picture class="chart">'
        "<tabular><fcel/>Chart<fcel/>1<nl/></tabular>"
        "<table><fcel/>Nested<fcel/>Cell<nl/></table>"
        "</picture></doclang>"
    )
    doc = _deserialize(doclang)
    pic = doc.pictures[0]
    assert pic.meta is not None
    assert pic.meta.tabular_chart.chart_data.grid[0][0].text == "Chart"
    assert len(pic.children) == 1
    nested = pic.children[0].resolve(doc)
    assert isinstance(nested, TableItem)
    assert nested.data.grid[0][0].text == "Nested"


# SMILES from test/data/doc/dummy_doc_with_meta.yaml (molecule_data annotation)
_EXAMPLE_MOLECULE_SMILES = "CC1=NNC(C2=CN3C=CN=C3C(CC3=CC(F)=CC(F)=C3)=N2)=N1"
_PICTURE_META_SUMMARY = "Picture meta summary"
_PICTURE_META_DESCRIPTION = "Picture meta description"
_PICTURE_META_CUSTOM_VALUE = "custom field on picture meta"
_CHART_TITLE = "Chart Title"


def _create_picture_molecule_doc() -> DoclingDocument:
    doc = DoclingDocument(name="picture_molecule_meta")
    pic = doc.add_picture()
    pic.meta = PictureMeta(
        summary=SummaryMetaField(text=_PICTURE_META_SUMMARY),
        description=DescriptionMetaField(text=_PICTURE_META_DESCRIPTION),
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name=PictureClassificationLabel.PIE_CHART.value,
                    confidence=1.0,
                )
            ]
        ),
        molecule=MoleculeMetaField(smi=_EXAMPLE_MOLECULE_SMILES),
    )
    pic.meta.set_custom_field(
        namespace="my_corp",
        name="note",
        value=_PICTURE_META_CUSTOM_VALUE,
    )
    # Tabular chart data (pattern from test_serialization_doclang._create_content_filtering_doc)
    chart_data = TableData(num_cols=2)
    chart_data.add_row(["Foo", "Bar"])
    chart_data.add_row(["One", "Two"])
    pic.meta.tabular_chart = TabularChartMetaField(
        title=_CHART_TITLE,
        chart_data=chart_data,
    )
    return doc


_KV_ANNOT_ROOT = Path(__file__).parent / "data" / "doc" / "kv"

# KV annot fixtures with lossless DocLang XML round-trip today.
_KV_ANNOT_XML_LOSSLESS = frozenset(
    {
        "01d07afe1cb54ecd23eedfe4d91b81dd88e61bf4e0dbe2467784db4177a6c691",
        "08212053e2db1a70dd60a4f85650ceb33d7519af34f502e3ac894389d76663d6",
        "1eac20e5ac5fac655a611343f86927d6a76277e170430c1eba741585437a2e90",
        "ba4120cada21304563625490e9ad13911e96114d3f07df056a6bf62397a859e1",
    }
)


def _kv_annot_fixture_dirs() -> list[Path]:
    """Return migrated KV annot fixture dirs with serialized DocLang goldens."""
    return sorted(
        p
        for p in _KV_ANNOT_ROOT.iterdir()
        if p.is_dir() and (p / "output.json").is_file() and (p / "output.dclg.xml").is_file()
    )


def _serialize_kv_annot_fixture(doc: DoclingDocument) -> str:
    text = DocLangDocSerializer(doc=doc, params=DocLangParams(include_version=False)).serialize().text
    if not GEN_TEST_DATA:
        assert_valid_dclg_xml(text)
    return text


@pytest.mark.parametrize(
    "fixture_dir",
    _kv_annot_fixture_dirs(),
    ids=[p.name for p in _kv_annot_fixture_dirs()],
)
def test_kv_annot_doclang_roundtrip(fixture_dir: Path):
    """Round-trip migrated KV annot fixtures through DocLang (see ``test/data/doc/kv/``)."""
    output_json = fixture_dir / "output.json"
    serialized_dclg = fixture_dir / "output.dclg.xml"
    deserialized_json = fixture_dir / "deserialized.json"
    reserialized_dclg = fixture_dir / "reserialized.dclg.xml"

    doc = DoclingDocument.load_from_json(output_json)

    dt = _serialize_kv_annot_fixture(doc)
    verify_doclang(exp_file=serialized_dclg, actual=dt)

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize_kv_annot_fixture(doc2)
    verify_doclang(exp_file=reserialized_dclg, actual=dt2)

    if fixture_dir.name in _KV_ANNOT_XML_LOSSLESS:
        assert dt.strip() == dt2.strip()


def _serialize_field_region_fixture(doc: DoclingDocument, *, fixture_dir: str) -> str:
    """Serialize a field-region fixture doc (invoice uses placeholder pictures)."""
    params = (
        DocLangParams(include_version=False, image_mode=ImageRefMode.PLACEHOLDER)
        if fixture_dir == "field_region_kv_invoice"
        else DocLangParams(include_version=False)
    )
    text = DocLangDocSerializer(doc=doc, params=params).serialize().text
    if not GEN_TEST_DATA:
        assert_valid_dclg_xml(text)
    return text


@pytest.mark.parametrize(
    "fixture_dir",
    [
        "field_region_kv_migration",
        "field_region_kv",
        "field_region_kv_invoice",
    ],
)
def test_field_region_doclang_roundtrip(fixture_dir: str):
    """Round-trip field regions/items through DocLang deserialization."""
    data_dir = Path(__file__).parent / "data" / "doc" / fixture_dir
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = DoclingDocument.load_from_json(input_json)
    _verify_doc(doc=doc, exp_json=input_json)
    assert doc.field_regions

    dt = _serialize_field_region_fixture(doc, fixture_dir=fixture_dir)
    verify_doclang(exp_file=serialized_dclg, actual=dt)

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)
    assert doc2.field_regions

    dt2 = _serialize_field_region_fixture(doc2, fixture_dir=fixture_dir)
    verify_doclang(exp_file=reserialized_dclg, actual=dt2)
    assert dt.strip() == dt2.strip()


def test_picture_molecule_meta_roundtrip():
    """Round-trip picture meta (molecule, chart/tabular, summary, description, custom) through DocLang."""
    data_dir = Path(__file__).parent / "data" / "doc" / "picture_molecule_meta"
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = _create_picture_molecule_doc()
    _verify_doc(doc=doc, exp_json=input_json)

    dt = _serialize(doc)
    verify(serialized_dclg, dt)
    assert f"<summary>{_PICTURE_META_SUMMARY}</summary>" in dt
    assert f"<description>{_PICTURE_META_DESCRIPTION}</description>" in dt
    assert f"<docling__smiles>{_EXAMPLE_MOLECULE_SMILES}</docling__smiles>" in dt
    assert f"<my_corp__note>{_PICTURE_META_CUSTOM_VALUE}</my_corp__note>" in dt
    assert 'class="chart"' in dt
    assert 'value="pie_chart"' in dt
    assert "<tabular>" in dt
    assert "Foo" in dt and "Bar" in dt and "One" in dt and "Two" in dt

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize(doc2)
    verify(reserialized_dclg, dt2)


def test_roundtrip_with_layers():
    """Test roundtrip with content layers."""
    from docling_core.types.doc import ContentLayer

    doc = DoclingDocument(name="t")
    # Add items with different layers
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Header", content_layer=ContentLayer.FURNITURE)
    doc.add_text(label=DocItemLabel.TEXT, text="Body text", content_layer=ContentLayer.BODY)
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text="Footer", content_layer=ContentLayer.FURNITURE)

    # Serialize with ALWAYS mode to ensure layers are included
    from docling_core.transforms.serializer.doclang import LayerMode

    ser = DocLangDocSerializer(
        doc=doc,
        params=DocLangParams(include_version=False, layer_mode=LayerMode.ALWAYS),
    )
    dt = ser.serialize().text

    # Deserialize
    doc2 = _deserialize(dt)

    # Verify layers are preserved
    assert len(doc2.body.children) == 3
    items = [doc2.body.children[i].resolve(doc2) for i in range(3)]
    assert items[0].content_layer == ContentLayer.FURNITURE
    assert items[1].content_layer == ContentLayer.BODY
    assert items[2].content_layer == ContentLayer.FURNITURE


def test_roundtrip_with_newlines():
    """Test that newlines in <content> survive deserialization and reserialization."""
    doclang_str = """
<doclang>
  <text>
    <content>foo
bar</content>
  </text>
  <text>
    <content>zoo
 </content>
    <bold>zen</bold>
  </text>
</doclang>"""

    doc = _deserialize(doclang_str)
    dt2 = _serialize(doc)
    assert dt2.strip() == doclang_str.strip()


def test_roundtrip_document_index_table():
    """Test that DOCUMENT_INDEX label is preserved through serialization/deserialization."""
    doc = DoclingDocument(name="test")
    _add_default_page(doc)

    # Add a regular table
    table_data = TableData(num_cols=2)
    table_data.add_row(["Header 1", "Header 2"])
    table_data.grid[0][0].column_header = True
    table_data.grid[0][1].column_header = True
    table_data.add_row(["Data 1", "Data 2"])
    doc.add_table(data=table_data, label=DocItemLabel.TABLE, prov=_default_prov())

    # Add a DOCUMENT_INDEX table
    index_data = TableData(num_cols=2)
    index_data.add_row(["Index 1", "Page 1"])
    index_data.add_row(["Index 2", "Page 2"])
    doc.add_table(data=index_data, label=DocItemLabel.DOCUMENT_INDEX, prov=_default_prov())

    # Serialize
    xml_str = _serialize(doc)

    assert "<index>" in xml_str

    # Deserialize
    doc2 = _deserialize(xml_str)

    # Verify we have 2 tables
    assert len(doc2.tables) == 2

    # Verify labels are preserved
    assert doc2.tables[0].label == DocItemLabel.TABLE
    assert doc2.tables[1].label == DocItemLabel.DOCUMENT_INDEX

    # Verify table data is preserved
    assert doc2.tables[0].data.num_rows == 2
    assert doc2.tables[0].data.num_cols == 2
    assert doc2.tables[1].data.num_rows == 2
    assert doc2.tables[1].data.num_cols == 2


def _thread_roundtrip(
    *,
    name: str,
    doc_factory: Callable[[], DoclingDocument],
    page_breaks: int = 0,
) -> None:
    """Round-trip a threaded document fixture through serialize → deserialize → reserialize."""
    data_dir = Path(__file__).parent / "data" / "doc" / name
    input_json = data_dir / "input.json"
    serialized_dclg = data_dir / "serialized.dclg.xml"
    deserialized_json = data_dir / "deserialized.json"
    reserialized_dclg = data_dir / "reserialized.dclg.xml"

    doc = doc_factory()
    _verify_doc(doc=doc, exp_json=input_json)

    dt = _serialize(doc)
    verify(serialized_dclg, dt)
    if page_breaks:
        assert dt.count("<page_break") == page_breaks

    doc2 = _deserialize(dt)
    _verify_doc(doc=doc2, exp_json=deserialized_json)

    dt2 = _serialize(doc2)
    verify(reserialized_dclg, dt2)


@doclang_validator
def test_cross_page_paragraph_roundtrip():
    """Round-trip a cross-page threaded paragraph."""
    _thread_roundtrip(name="cross_page_paragraph", doc_factory=_doc_cross_page_paragraph, page_breaks=1)


@doclang_validator
def test_cross_column_paragraph_roundtrip():
    """Round-trip a cross-column threaded paragraph (same page, two boxes)."""
    _thread_roundtrip(name="multi_prov_thread", doc_factory=_doc_multi_prov_text, page_breaks=0)


@doclang_validator
def test_cross_page_list_roundtrip():
    """Round-trip a cross-page threaded list (whole items per page)."""
    _thread_roundtrip(name="cross_page_list", doc_factory=_doc_cross_page_list, page_breaks=1)


@doclang_validator
def test_cross_page_table_roundtrip():
    """Round-trip a cross-page threaded table."""
    _thread_roundtrip(name="cross_page_table", doc_factory=_doc_cross_page_table, page_breaks=1)


@doclang_validator
def test_cross_column_list_roundtrip():
    """Round-trip a cross-column list (same page, one item per column)."""
    _thread_roundtrip(name="cross_column_list", doc_factory=_doc_cross_column_list, page_breaks=0)


def test_table_with_class_raises_error():
    r"""Test that ``<table class=\"…\">`` is rejected (v0.5: use ``<index>`` for document indexes)."""
    xml_str = """<doclang>
  <table class="index">
    <fcel/>
    <text>Data 1</text>
    <nl/>
  </table>
</doclang>"""

    with pytest.raises(ValueError, match="table element must not have a class attribute"):
        _deserialize(xml_str, validate=False)
