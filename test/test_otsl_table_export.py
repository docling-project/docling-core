from docling_core.types.doc.document import DoclingDocument, TableCell, TableData


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

    otsl_string = doc.tables[0].export_to_otsl(
        add_cell_location=False, add_cell_text=False, doc=doc
    )
    print_friendly = otsl_string.split("<nl>")
    print("OTSL out:")
    for s in print_friendly:
        print(s)
    assert (
        otsl_string
        == "<rhed><lcel><rhed><fcel><xcel><xcel><nl><rhed><fcel><fcel><xcel><xcel><xcel><nl><rhed><fcel><fcel><fcel><ecel><ecel><nl><ucel><fcel><fcel><fcel><fcel><fcel><nl><srow><lcel><lcel><lcel><lcel><lcel><nl>"
    )
