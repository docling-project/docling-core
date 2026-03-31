"""Standalone script to exercise every branch of TripletTableSerializer."""

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    TripletTableSerializer,
)
from docling_core.types.doc import DoclingDocument, TableData
from docling_core.types.doc.document import RichTableCell, TableCell


def _cell(text, row, col, *, column_header=False, row_header=False, row_span=1, col_span=1):
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
        column_header=column_header,
        row_header=row_header,
        row_span=row_span,
        col_span=col_span,
    )


def _serialize(doc, table_item):
    ser = ChunkingDocSerializer(doc=doc)
    return TripletTableSerializer().serialize(
        item=table_item, doc_serializer=ser, doc=doc
    ).text


def _header(label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


# ── 1. Both column + row headers ─────────────────────────────
_header("1. Both column and row headers")
#        | Q1  | Q2
# -------+-----+-----
#  Rev   | 100 | 200
#  Cost  |  50 |  80

doc = DoclingDocument(name="both_headers")
t = doc.add_table(data=TableData(num_rows=3, num_cols=3))
for j, h in enumerate(["", "Q1", "Q2"]):
    doc.add_table_cell(table_item=t, cell=_cell(h, 0, j, column_header=True))
for i, (rh, vals) in enumerate([("Rev", ["100", "200"]), ("Cost", ["50", "80"])], 1):
    doc.add_table_cell(table_item=t, cell=_cell(rh, i, 0, row_header=True))
    for j, v in enumerate(vals, 1):
        doc.add_table_cell(table_item=t, cell=_cell(v, i, j))

print(_serialize(doc, t))


# ── 2. Column headers only (multi-column) ────────────────────
_header("2. Column headers only (multi-column)")
# | Year | Revenue | Employees |
# | 2014 |    92.7 |   379,592 |
# | 2015 |    81.7 |   377,757 |

doc = DoclingDocument(name="col_headers_only")
t = doc.add_table(data=TableData(num_rows=3, num_cols=3))
for j, h in enumerate(["Year", "Revenue", "Employees"]):
    doc.add_table_cell(table_item=t, cell=_cell(h, 0, j, column_header=True))
for i, row in enumerate([["2014", "92.7", "379,592"], ["2015", "81.7", "377,757"]], 1):
    for j, v in enumerate(row):
        doc.add_table_cell(table_item=t, cell=_cell(v, i, j))

print(_serialize(doc, t))


# ── 3. Column headers only (single column) ───────────────────
_header("3. Column headers only (single column)")
# | Country     |
# | Italy       |
# | Canada      |
# | Switzerland |

doc = DoclingDocument(name="single_col_header")
t = doc.add_table(data=TableData(num_rows=4, num_cols=1))
doc.add_table_cell(table_item=t, cell=_cell("Country", 0, 0, column_header=True))
for i, name in enumerate(["Italy", "Canada", "Switzerland"], 1):
    doc.add_table_cell(table_item=t, cell=_cell(name, i, 0))

print(_serialize(doc, t))


# ── 4. Row headers only ──────────────────────────────────────
_header("4. Row headers only")
#  Rev  | 100 | 200
#  Cost |  50 |  80

doc = DoclingDocument(name="row_headers_only")
t = doc.add_table(data=TableData(num_rows=2, num_cols=3))
for i, (rh, vals) in enumerate([("Rev", ["100", "200"]), ("Cost", ["50", "80"])]):
    doc.add_table_cell(table_item=t, cell=_cell(rh, i, 0, row_header=True))
    for j, v in enumerate(vals, 1):
        doc.add_table_cell(table_item=t, cell=_cell(v, i, j))

print(_serialize(doc, t))


# ── 5. No headers at all ─────────────────────────────────────
_header("5. No headers")
# | a | b |
# | c | d |

doc = DoclingDocument(name="no_headers")
t = doc.add_table(data=TableData(num_rows=2, num_cols=2))
for i, row in enumerate([["a", "b"], ["c", "d"]]):
    for j, v in enumerate(row):
        doc.add_table_cell(table_item=t, cell=_cell(v, i, j))

print(_serialize(doc, t))


# ── 6. Single column, no headers ─────────────────────────────
_header("6. Single column, no headers")

doc = DoclingDocument(name="single_col_no_header")
t = doc.add_table(data=TableData(num_rows=3, num_cols=1))
for i, v in enumerate(["x", "y", "z"]):
    doc.add_table_cell(table_item=t, cell=_cell(v, i, 0))

print(_serialize(doc, t))


# ── 7. Merged cell (row-span) ────────────────────────────────
_header("7. Merged cell (row-span)")
# | Year | Revenue |
# | 2024 |     100 |   ← "2024" spans rows 1–2
# |      |     200 |

doc = DoclingDocument(name="merged")
t = doc.add_table(data=TableData(num_rows=3, num_cols=2))
doc.add_table_cell(table_item=t, cell=_cell("Year", 0, 0, column_header=True))
doc.add_table_cell(table_item=t, cell=_cell("Revenue", 0, 1, column_header=True))
doc.add_table_cell(table_item=t, cell=_cell("2024", 1, 0, row_span=2))
doc.add_table_cell(table_item=t, cell=_cell("100", 1, 1))
doc.add_table_cell(table_item=t, cell=_cell("200", 2, 1))

print(_serialize(doc, t))


# ── 8. Nested table ──────────────────────────────────────────
_header("8. Nested table (inner table embedded in a cell)")

doc = DoclingDocument(name="nested")
outer = doc.add_table(data=TableData(num_rows=1, num_cols=2))
inner = doc.add_table(data=TableData(num_rows=1, num_cols=2), parent=outer)
doc.add_table_cell(table_item=inner, cell=_cell("x", 0, 0))
doc.add_table_cell(table_item=inner, cell=_cell("y", 0, 1))
doc.add_table_cell(table_item=outer, cell=_cell("plain", 0, 0))
doc.add_table_cell(
    table_item=outer,
    cell=RichTableCell(
        text="",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=1,
        end_col_offset_idx=2,
        ref=inner.get_ref(),
    ),
)

print(_serialize(doc, outer))

print()
