import pytest

from docling_core.experimental.idoctags import (
    IDocTagsDocDeserializer,
    IDocTagsDocSerializer,
    IDocTagsParams,
)
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableData


def _serialize(doc: DoclingDocument) -> str:
    ser = IDocTagsDocSerializer(doc=doc, params=IDocTagsParams())
    return ser.serialize().text


def _deserialize(text: str) -> DoclingDocument:
    return IDocTagsDocDeserializer().deserialize(doctags=text)


def test_roundtrip_text():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.TEXT, text="Hello world")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TEXT
    assert doc2.texts[0].text == "Hello world"


def test_roundtrip_title():
    doc = DoclingDocument(name="t")
    doc.add_title(text="My Title")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TITLE
    assert doc2.texts[0].text == "My Title"


def test_roundtrip_heading():
    doc = DoclingDocument(name="t")
    doc.add_heading(text="Section A", level=2)
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    h = doc2.texts[0]
    assert h.label == DocItemLabel.SECTION_HEADER and getattr(h, "level", 0) == 2
    assert h.text == "Section A"


def test_roundtrip_caption():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.CAPTION, text="Cap text")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.CAPTION
    assert doc2.texts[0].text == "Cap text"


def test_roundtrip_footnote():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.FOOTNOTE, text="Foot note")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.FOOTNOTE
    assert doc2.texts[0].text == "Foot note"


def test_roundtrip_page_header():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Header")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.PAGE_HEADER
    assert doc2.texts[0].text == "Header"


def test_roundtrip_page_footer():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.PAGE_FOOTER, text="Footer")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.PAGE_FOOTER
    assert doc2.texts[0].text == "Footer"


def test_roundtrip_code():
    doc = DoclingDocument(name="t")
    doc.add_code(text="print('hi')")
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.CODE
    assert doc2.texts[0].text.strip() == "print('hi')"


def test_roundtrip_formula():
    doc = DoclingDocument(name="t")
    doc.add_formula(text="E=mc^2")
    dt = _serialize(doc)
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
    doc2 = _deserialize(dt)
    assert len(doc2.groups) == 1
    assert len(doc2.texts) == 2
    assert [it.text for it in doc2.texts] == ["1", "2"]


def test_roundtrip_picture_with_caption():
    doc = DoclingDocument(name="t")
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Fig 1")
    doc.add_picture(caption=cap)
    dt = _serialize(doc)
    print(dt)
    
    doc2 = _deserialize(dt)
    assert len(doc2.pictures) == 1
    # Caption added as a separate text item referenced by the picture
    assert len(doc2.texts) >= 1
    cap_texts = [t for t in doc2.texts if t.label == DocItemLabel.CAPTION]
    assert len(cap_texts) == 1 and cap_texts[0].text == "Fig 1"


def test_roundtrip_table_simple():
    doc = DoclingDocument(name="t")
    td = TableData(num_rows=0, num_cols=2)
    td.add_row(["H1", "H2"])  # header row semantics not required here
    td.add_row(["C1", "C2"])  # data row
    doc.add_table(data=td)
    dt = _serialize(doc)
    doc2 = _deserialize(dt)
    assert len(doc2.tables) == 1
    t2 = doc2.tables[0].data
    assert t2.num_rows == 2 and t2.num_cols == 2
    grid_texts = [[cell.text for cell in row] for row in t2.grid]
    assert grid_texts == [["H1", "H2"], ["C1", "C2"]]

