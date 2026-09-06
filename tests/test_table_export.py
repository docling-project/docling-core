# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

from unittest.mock import PropertyMock, patch

from docling_core.types.doc import DoclingDocument, TableCell, TableData


def test_export_to_dataframe_builds_grid_once():
    num_rows = 100
    data = TableData(
        num_rows=num_rows,
        num_cols=1,
        table_cells=[
            TableCell(
                text="header",
                row_span=num_rows,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=num_rows,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                column_header=True,
            )
        ],
    )
    table = DoclingDocument(name="test").add_table(data=data)
    grid = data.grid

    with patch.object(TableData, "grid", new_callable=PropertyMock, return_value=grid) as grid_property:
        dataframe = table.export_to_dataframe()

    assert grid_property.call_count == 1
    assert dataframe.shape == (0, 1)
