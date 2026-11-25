from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument, TableCell, TableData
from docling_core.types.doc.utils import parse_otsl_table_content


def test_table_export_to_otsl():
    data_table_cells = []
    num_cols = 6
    num_rows = 5
    # ======================================
    data_table_cells.append(
        TableCell(
            text="AB",
            row_span=1,
            col_span=2,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=3,
            col_header=False,
            row_header=True,
        )
    )

    data_table_cells.append(
        TableCell(
            text="C",
            row_span=1,
            col_span=1,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=True,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="1",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            col_header=False,
            row_header=True,
        )
    )

    data_table_cells.append(
        TableCell(
            text="2",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="3",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=False,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="2D",
            row_span=2,
            col_span=3,
            start_row_offset_idx=0,
            end_row_offset_idx=2,
            start_col_offset_idx=3,
            end_col_offset_idx=6,
            col_header=True,
            row_header=False,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="4",
            row_span=2,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=4,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            col_header=False,
            row_header=True,
        )
    )

    data_table_cells.append(
        TableCell(
            text="5",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="6",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="next 2 cells empty",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=3,
            end_col_offset_idx=4,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=4,
            end_col_offset_idx=5,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=5,
            end_col_offset_idx=6,
            col_header=False,
            row_header=False,
        )
    )

    # ======================================

    data_table_cells.append(
        TableCell(
            text="Q",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="W",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="E",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=3,
            end_col_offset_idx=4,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="R",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=4,
            end_col_offset_idx=5,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="T",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=5,
            end_col_offset_idx=6,
            col_header=False,
            row_header=False,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="Section header",
            row_span=1,
            col_span=6,
            start_row_offset_idx=4,
            end_row_offset_idx=5,
            start_col_offset_idx=0,
            end_col_offset_idx=6,
            col_header=False,
            row_header=False,
            row_section=True,
        )
    )

    # ======================================
    doc = DoclingDocument(name="test_otsl")
    data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=data_table_cells)
    doc.add_table(data=data)

    otsl_string = doc.tables[0].export_to_otsl(add_cell_location=False, add_cell_text=False, doc=doc)
    otsl_string.split("<nl>")
    # print("OTSL out:")

    """
    for s in print_friendly:
        print(s)
    """

    assert (
        otsl_string
        == "<rhed><lcel><rhed><fcel><lcel><lcel><nl><rhed><fcel><fcel><ucel><xcel><xcel><nl><rhed><fcel><fcel><fcel><ecel><ecel><nl><ucel><fcel><fcel><fcel><fcel><fcel><nl><srow><lcel><lcel><lcel><lcel><lcel><nl>"
    )


def test_table_export_to_otsl_page_less_document_with_cell_bbox():
    doc = DoclingDocument(name="page_less_doc")

    data = TableData(
        num_rows=1,
        num_cols=1,
        table_cells=[
            TableCell(
                text="cell",
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                bbox=BoundingBox(l=0, t=0, r=10, b=10),
            )
        ],
    )

    table = doc.add_table(data=data)

    assert table.export_to_otsl(doc=doc)


def test_table_export_to_otsl_2d_span_next_to_other_cells():
    # "Y" spans rows 0-1 and columns 1-2, "X" sits to its left in row 1
    def cell(text, r0, r1, c0, c1):
        return TableCell(
            text=text,
            row_span=r1 - r0,
            col_span=c1 - c0,
            start_row_offset_idx=r0,
            end_row_offset_idx=r1,
            start_col_offset_idx=c0,
            end_col_offset_idx=c1,
        )

    doc = DoclingDocument(name="test_otsl_2d_span")
    data = TableData(
        num_rows=2,
        num_cols=3,
        table_cells=[cell("A", 0, 1, 0, 1), cell("Y", 0, 2, 1, 3), cell("X", 1, 2, 0, 1)],
    )
    table = doc.add_table(data=data)

    otsl_string = table.export_to_otsl(doc=doc, add_cell_location=False)

    # a slot continuing the cell only from the left is lcel, only from above is ucel
    assert otsl_string == "<fcel>A<fcel>Y<lcel><nl><fcel>X<ucel><xcel><nl>"

    spans = {
        c.text: (c.start_row_offset_idx, c.end_row_offset_idx, c.start_col_offset_idx, c.end_col_offset_idx)
        for c in parse_otsl_table_content(otsl_string).table_cells
    }
    assert spans == {"A": (0, 1, 0, 1), "Y": (0, 2, 1, 3), "X": (1, 2, 0, 1)}
