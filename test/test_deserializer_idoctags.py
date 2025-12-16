import pytest

from docling_core.experimental.idoctags import (
    IDocTagsDocDeserializer,
    IDocTagsDocSerializer,
    IDocTagsParams,
)
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import CodeLanguageLabel

DO_PRINT: bool = False


def _serialize(
    doc: DoclingDocument,
    add_location: bool = True,
    add_content: bool = True,
    add_table_cell_location: bool = False,
    add_table_cell_text: bool = True,
    xml_compliant: bool = True,
) -> str:
    ser = IDocTagsDocSerializer(
        doc=doc,
        params=IDocTagsParams(
            add_location=add_location,
            add_content=add_content,
            add_table_cell_location=add_table_cell_location,
            add_table_cell_text=add_table_cell_text,
            xml_compliant=xml_compliant,
        ),
    )
    return ser.serialize().text


def _deserialize(text: str) -> DoclingDocument:
    return IDocTagsDocDeserializer().deserialize(doctags=text)


def _add_default_page(doc: DoclingDocument):
    doc.add_page(page_no=0, size=Size(width=1000, height=1000))


def _default_prov() -> ProvenanceItem:
    return ProvenanceItem(
        page_no=0,
        bbox=BoundingBox(l=100, t=100, r=300, b=200),
        charspan=(0, 0),
    )


def test_roundtrip_text():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.TEXT, text="Hello world")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TEXT
    assert doc2.texts[0].text == "Hello world"


def test_roundtrip_title():
    doc = DoclingDocument(name="t")
    doc.add_title(text="My Title")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TITLE
    assert doc2.texts[0].text == "My Title"


def test_roundtrip_heading():
    doc = DoclingDocument(name="t")
    doc.add_heading(text="Section A", level=2)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    h = doc2.texts[0]
    assert h.label == DocItemLabel.SECTION_HEADER and getattr(h, "level", 0) == 2
    assert h.text == "Section A"


def test_roundtrip_caption():
    doc = DoclingDocument(name="t")
    doc.add_text(label=DocItemLabel.CAPTION, text="Cap text")
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.CAPTION
    assert doc2.texts[0].text == "Cap text"


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
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.CODE
    assert doc2.texts[0].text.strip() == "print('hi')"
    assert getattr(doc2.texts[0], "code_language", None) == CodeLanguageLabel.PYTHON


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
    doc.add_title(text="My Title", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.TITLE
    assert doc2.texts[0].text == "My Title"


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
    doc.add_text(label=DocItemLabel.CAPTION, text="Cap text", prov=_default_prov())
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.texts) == 1
    assert doc2.texts[0].label == DocItemLabel.CAPTION
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
    doc.add_code(
        text="print('hi')", code_language=CodeLanguageLabel.PYTHON, prov=_default_prov()
    )
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
    doc.add_list_item("A", parent=lg, enumerated=False, prov=prov)
    doc.add_list_item("B", parent=lg, enumerated=False, prov=prov)
    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)
    assert len(doc2.groups) == 1
    assert len(doc2.texts) == 2
    assert [it.text for it in doc2.texts] == ["A", "B"]


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
    assert len(doc2.texts) >= 1
    cap_texts = [t for t in doc2.texts if t.label == DocItemLabel.CAPTION]
    assert len(cap_texts) == 1 and cap_texts[0].text == "Fig 1"


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

    dt = _serialize(doc, add_table_cell_text=True, add_content=True)
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
    # Should have 4 text items total
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
    doc.add_list_item("Step 1", parent=lg_outer, enumerated=True)

    # Create nested ordered list
    lg_inner = doc.add_list_group(parent=lg_outer)
    doc.add_list_item("Step 1.1", parent=lg_inner, enumerated=True)
    doc.add_list_item("Step 1.2", parent=lg_inner, enumerated=True)

    doc.add_list_item("Step 2", parent=lg_outer, enumerated=True)

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)
    doc2 = _deserialize(dt)

    assert len(doc2.groups) == 2
    assert len(doc2.texts) == 4
    text_contents = [it.text for it in doc2.texts]
    assert "Step 1" in text_contents
    assert "Step 1.1" in text_contents
    assert "Step 1.2" in text_contents
    assert "Step 2" in text_contents

    # Verify round-trip serialization
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


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
    """Test list item containing inline group with text, code, and formula."""
    doc = DoclingDocument(name="t")
    lg = doc.add_list_group()

    # First list item with inline group containing code
    li1 = doc.add_list_item(text="", parent=lg, enumerated=False)
    inline1 = doc.add_inline_group(parent=li1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a code snippet:",
        parent=inline1,
    )
    doc.add_code(
        text='print("Hello world")',
        parent=inline1,
        code_language=CodeLanguageLabel.PYTHON,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="(to be displayed inline)",
        parent=inline1,
    )

    # Second list item with inline group containing formula
    li2 = doc.add_list_item(text="", parent=lg, enumerated=False)
    inline2 = doc.add_inline_group(parent=li2)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a formula:",
        parent=inline2,
    )
    doc.add_formula(text="E=mc^2", parent=inline2)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="(to be displayed inline)",
        parent=inline2,
    )

    dt = _serialize(doc)
    if DO_PRINT:
        print("\n", dt)

    doc2 = _deserialize(dt)

    # Verify structure
    assert len(doc2.groups) >= 1  # At least one list group
    # Should have at least 2 list items, plus inline content (texts, code, formula)
    assert len(doc2.texts) >= 2

    # Verify round-trip
    dt2 = _serialize(doc2)
    if DO_PRINT:
        print("\ndt:", dt)
        print("\ndt2:", dt2)
    assert dt2 == dt


############################################
### Feature complete document test-cases ###
############################################


def test_constructed_doc(sample_doc: DoclingDocument):
    doc = sample_doc

    dt = _serialize(doc, add_table_cell_text=True, add_content=True)

    doc2 = _deserialize(dt)

    dt2 = _serialize(doc2, add_table_cell_text=True, add_content=True)

    # if DO_PRINT:
    # print(f"--------------------------dt:\n\n{dt}\n\n")
    # print(f"--------------------------dt2:\n\n{dt2}\n\n")

    assert dt2 == dt


@pytest.mark.xfail(
    reason="Known feature incompletenes in serialization/deseralization for rich table cells!"
)
def test_constructed_rich_table_doc(rich_table_doc: DoclingDocument):
    doc = rich_table_doc

    dt = _serialize(doc, add_table_cell_text=True, add_content=True, xml_compliant=True)

    doc2 = _deserialize(dt)

    dt2 = _serialize(
        doc2, add_table_cell_text=True, add_content=True, xml_compliant=True
    )

    assert dt2 == dt
