from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import TableData

# Table bbox - defines region of a table, everything outside will be ignored
table_bbox: BoundingBox = BoundingBox(l=0, t=0, r=100, b=175)

# List of regions that defines rows for table structure
rows: list[BoundingBox] = [
    BoundingBox(l=1, t=1, r=99, b=25),
    BoundingBox(l=1, t=25, r=99, b=50),
    BoundingBox(l=1, t=50, r=99, b=75),
    BoundingBox(l=1, t=75, r=99, b=99),
    BoundingBox(l=1, t=100, r=99, b=149),
    BoundingBox(l=1, t=150, r=99, b=175),
]

# List of regions that defines columns for table structure
cols: list[BoundingBox] = [
    BoundingBox(l=1, t=1, r=25, b=149),
    BoundingBox(l=25, t=1, r=50, b=149),
    BoundingBox(l=50, t=1, r=75, b=149),
    BoundingBox(l=75, t=1, r=99, b=149),
]

# List of regions that defines merged cells on top of row/clumn grid (spans)
merges: list[BoundingBox] = [
    BoundingBox(l=0, t=0, r=50, b=25),
    BoundingBox(l=50, t=0, r=99, b=25),
]

# (OPTIONAL) Semantic of a table - region that cover column headers
col_headers: list[BoundingBox] = [
    BoundingBox(l=0, t=0, r=99, b=25),
]

# (OPTIONAL) Semantic of a table - region that cover row headers
row_headers: list[BoundingBox] = [
    BoundingBox(l=0, t=0, r=50, b=150),
]

# (OPTIONAL) Semantic of a table - region that cover section rows
row_section: list[BoundingBox] = [
    BoundingBox(l=1, t=75, r=99, b=99),
]


def test_regions_to_table_convert():
    # Converts regions: rows, columns, merged cells
    # into table_data structure,
    # Adds semantics for regions of row_headers, col_headers, row_section
    table_data = TableData.from_regions(
        table_bbox=table_bbox,
        rows=rows,
        cols=cols,
        merges=merges,
        row_headers=row_headers,
        col_headers=col_headers,
        row_sections=row_section,
    )

    assert table_data.num_cols == 4
    assert table_data.num_rows == 6

    assert table_data.table_cells[0].bbox.l == 1.0
    assert table_data.table_cells[0].bbox.t == 1.0
    assert table_data.table_cells[0].bbox.r == 50.0
    assert table_data.table_cells[0].bbox.b == 25.0

    assert table_data.table_cells[0].col_span == 2
    assert table_data.table_cells[0].column_header
    assert table_data.table_cells[1].column_header

    assert table_data.table_cells[10].row_header
    assert table_data.table_cells[12].row_section

    assert table_data.table_cells[17].bbox.l == 75.0
    assert table_data.table_cells[17].bbox.t == 100.0
    assert table_data.table_cells[17].bbox.r == 99.0
    assert table_data.table_cells[17].bbox.b == 149.0
