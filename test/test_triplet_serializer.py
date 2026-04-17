"""Pytest coverage for TripletTableSerializer scenarios."""

import pandas as pd

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    TripletTableSerializer,
)
from docling_core.types.doc import DoclingDocument, TableData
from docling_core.types.doc.document import RichTableCell, TableCell, TableItem


def _cell(
    text: str,
    row: int,
    col: int,
    *,
    column_header: bool = False,
    row_header: bool = False,
    row_span: int = 1,
    col_span: int = 1,
) -> TableCell:
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


def _serialize(doc: DoclingDocument, table_item: TableItem) -> str:
    serializer = ChunkingDocSerializer(doc=doc)
    table_serializer = TripletTableSerializer()
    return table_serializer.serialize(
        item=table_item,
        doc_serializer=serializer,
        doc=doc,
    ).text


def test_triplet_both_row_and_column_headers():
    #        | Q1  | Q2
    # -------+-----+-----
    #  Rev   | 100 | 200
    #  Cost  |  50 |  80
    doc = DoclingDocument(name="both_headers")
    table = doc.add_table(data=TableData(num_rows=3, num_cols=3))

    for j, header in enumerate(["", "Q1", "Q2"]):
        doc.add_table_cell(
            table_item=table, cell=_cell(header, 0, j, column_header=True)
        )

    for i, (row_header, values) in enumerate(
        [("Rev", ["100", "200"]), ("Cost", ["50", "80"])],
        start=1,
    ):
        doc.add_table_cell(
            table_item=table, cell=_cell(row_header, i, 0, row_header=True)
        )
        for j, value in enumerate(values, start=1):
            doc.add_table_cell(table_item=table, cell=_cell(value, i, j))

    text = _serialize(doc, table)
    assert (
        text
        == "Rev, Q1 = 100. Rev, Q2 = 200. Cost, Q1 = 50. Cost, Q2 = 80"
    )


def test_triplet_column_headers_only_multi_column():
    # | Year | Revenue | Employees |
    # | 2014 |    92.7 |   379,592 |
    # | 2015 |    81.7 |   377,757 |
    doc = DoclingDocument(name="col_headers_only")
    table = doc.add_table(data=TableData(num_rows=3, num_cols=3))

    for j, header in enumerate(["Year", "Revenue", "Employees"]):
        doc.add_table_cell(
            table_item=table, cell=_cell(header, 0, j, column_header=True)
        )

    for i, row in enumerate(
        [["2014", "92.7", "379,592"], ["2015", "81.7", "377,757"]],
        start=1,
    ):
        for j, value in enumerate(row):
            doc.add_table_cell(table_item=table, cell=_cell(value, i, j))

    text = _serialize(doc, table)
    assert (
        text
        == "row_0, Year = 2014. row_0, Revenue = 92.7. row_0, Employees = 379,592. "
        "row_1, Year = 2015. row_1, Revenue = 81.7. row_1, Employees = 377,757"
    )


def test_triplet_column_headers_only_single_column():
    # | Country     |
    # | Italy       |
    # | Canada      |
    # | Switzerland |
    doc = DoclingDocument(name="single_col_header")
    table = doc.add_table(data=TableData(num_rows=4, num_cols=1))
    doc.add_table_cell(table_item=table, cell=_cell("Country", 0, 0, column_header=True))

    for i, name in enumerate(["Italy", "Canada", "Switzerland"], start=1):
        doc.add_table_cell(table_item=table, cell=_cell(name, i, 0))

    text = _serialize(doc, table)
    assert text == "Country = Italy. Country = Canada. Country = Switzerland"


def test_triplet_row_headers_only():
    # Rev  | 100 | 200
    # Cost |  50 |  80
    doc = DoclingDocument(name="row_headers_only")
    table = doc.add_table(data=TableData(num_rows=2, num_cols=3))

    for i, (row_header, values) in enumerate(
        [("Rev", ["100", "200"]), ("Cost", ["50", "80"])]
    ):
        doc.add_table_cell(
            table_item=table, cell=_cell(row_header, i, 0, row_header=True)
        )
        for j, value in enumerate(values, start=1):
            doc.add_table_cell(table_item=table, cell=_cell(value, i, j))

    text = _serialize(doc, table)
    assert text == "Rev, col_1 = 100. Rev, col_2 = 200. Cost, col_1 = 50. Cost, col_2 = 80"


def test_triplet_no_headers():
    # | a | b |
    # | c | d |
    doc = DoclingDocument(name="no_headers")
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2))

    for i, row in enumerate([["a", "b"], ["c", "d"]]):
        for j, value in enumerate(row):
            doc.add_table_cell(table_item=table, cell=_cell(value, i, j))

    text = _serialize(doc, table)
    assert text == "row_0, col_0 = a. row_0, col_1 = b. row_1, col_0 = c. row_1, col_1 = d"


def test_triplet_single_column_no_headers():
    # | x |
    # | y |
    # | z |
    doc = DoclingDocument(name="single_col_no_header")
    table = doc.add_table(data=TableData(num_rows=3, num_cols=1))

    for i, value in enumerate(["x", "y", "z"]):
        doc.add_table_cell(table_item=table, cell=_cell(value, i, 0))

    text = _serialize(doc, table)
    assert text == "row_0, col_0 = x. row_1, col_0 = y. row_2, col_0 = z"


def test_triplet_merged_row_span_keeps_first_data_row():
    # | Year | Revenue |
    # | 2024 |     100 |  <- "2024" spans rows 1-2
    # |      |     200 |
    doc = DoclingDocument(name="merged")
    table = doc.add_table(data=TableData(num_rows=3, num_cols=2))

    doc.add_table_cell(table_item=table, cell=_cell("Year", 0, 0, column_header=True))
    doc.add_table_cell(table_item=table, cell=_cell("Revenue", 0, 1, column_header=True))
    doc.add_table_cell(table_item=table, cell=_cell("2024", 1, 0, row_span=2))
    doc.add_table_cell(table_item=table, cell=_cell("100", 1, 1))
    doc.add_table_cell(table_item=table, cell=_cell("200", 2, 1))

    text = _serialize(doc, table)
    assert text == "row_0, Year = 2024. row_0, Revenue = 100. row_1, Year = 2024. row_1, Revenue = 200"


def test_triplet_nested_table_in_cell():
    # Outer table:
    # | plain | [inner table] |
    #
    # Inner table:
    # | x | y |
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

    text = _serialize(doc, outer)
    assert text == "row_0, col_0 = plain. row_0, col_1 -> row_0, col_0 = x. row_0, col_1 = y"


def test_triplet_excluded_table_returns_early(monkeypatch):
    doc = DoclingDocument(name="excluded")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    doc.add_table_cell(table_item=table, cell=_cell("x", 0, 0))

    serializer = ChunkingDocSerializer(doc=doc)
    table_serializer = TripletTableSerializer()
    monkeypatch.setattr(
        ChunkingDocSerializer,
        "get_excluded_refs",
        lambda self, **kwargs: {table.self_ref},
    )

    text = table_serializer.serialize(
        item=table,
        doc_serializer=serializer,
        doc=doc,
    ).text
    assert text == ""


def test_triplet_empty_dataframe_returns_early(monkeypatch):
    doc = DoclingDocument(name="empty_df")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    doc.add_table_cell(table_item=table, cell=_cell("x", 0, 0))

    serializer = ChunkingDocSerializer(doc=doc)
    table_serializer = TripletTableSerializer()
    monkeypatch.setattr(table, "_export_to_dataframe_with_options", lambda *args, **kwargs: pd.DataFrame())

    text = table_serializer.serialize(
        item=table,
        doc_serializer=serializer,
        doc=doc,
    ).text
    assert text == ""


def test_triplet_fallback_detects_column_headers_from_dataframe(monkeypatch):
    doc = DoclingDocument(name="fallback_col_headers")
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2))
    doc.add_table_cell(table_item=table, cell=_cell("a", 0, 0))
    doc.add_table_cell(table_item=table, cell=_cell("b", 0, 1))
    doc.add_table_cell(table_item=table, cell=_cell("c", 1, 0))
    doc.add_table_cell(table_item=table, cell=_cell("d", 1, 1))

    serializer = ChunkingDocSerializer(doc=doc)
    table_serializer = TripletTableSerializer()
    df = pd.DataFrame([["a", "b"], ["c", "d"]], columns=["Q1", "Q2"])
    monkeypatch.setattr(table, "_export_to_dataframe_with_options", lambda *args, **kwargs: df)

    text = table_serializer.serialize(
        item=table,
        doc_serializer=serializer,
        doc=doc,
    ).text
    assert text == "row_0, Q1 = a. row_0, Q2 = b. row_1, Q1 = c. row_1, Q2 = d"


def test_triplet_build_triplets_rangeindex_uses_first_row_as_headers():
    df = pd.DataFrame(
        [
            ["Year", "Revenue"],
            ["2024", "100"],
            ["2025", "120"],
        ]
    )
    triplets = TripletTableSerializer._build_triplets(
        df,
        has_row_headers=False,
        has_col_headers=True,
    )
    assert triplets == [
        "row_1, Year = 2024",
        "row_1, Revenue = 100",
        "row_2, Year = 2025",
        "row_2, Revenue = 120",
    ]


def test_triplet_single_column_empty_header_falls_back_to_col_0():
    df = pd.DataFrame(["Italy", "Canada"], columns=[""])
    triplets = TripletTableSerializer._build_triplets(
        df,
        has_row_headers=False,
        has_col_headers=True,
    )
    assert triplets == ["col_0 = Italy", "col_0 = Canada"]


def test_triplet_nested_dataframe_skipped_at_max_depth():
    nested = pd.DataFrame([["x", "y"]], columns=["A", "B"])
    outer = pd.DataFrame([[nested, "plain"]], columns=["nested", "value"])
    triplets = TripletTableSerializer._build_triplets(
        outer,
        has_row_headers=False,
        has_col_headers=True,
        depth=0,
        max_depth=0,
    )
    assert triplets == ["row_0, value = plain"]


def test_triplet_single_column_header_skips_empty_values():
    df = pd.DataFrame(["Italy", None, "   ", "Canada"], columns=["Country"])
    triplets = TripletTableSerializer._build_triplets(
        df,
        has_row_headers=False,
        has_col_headers=True,
    )
    assert triplets == ["Country = Italy", "Country = Canada"]


def test_triplet_non_string_cell_uses_scalar_path():
    df = pd.DataFrame([[123, "ok"]], columns=["left", "right"])
    triplets = TripletTableSerializer._build_triplets(
        df,
        has_row_headers=False,
        has_col_headers=True,
    )
    assert triplets == ["row_0, left = 123", "row_0, right = ok"]


def test_triplet_nested_dataframe_emits_arrow_triplets():
    nested = pd.DataFrame([["x", "y"]], columns=["A", "B"])
    outer = pd.DataFrame([[nested, "plain"]], columns=["nested", "value"])
    triplets = TripletTableSerializer._build_triplets(
        outer,
        has_row_headers=False,
        has_col_headers=True,
        depth=0,
        max_depth=3,
    )
    assert triplets == [
        "row_0, nested -> row_0, A = x",
        "row_0, nested -> row_0, B = y",
        "row_0, value = plain",
    ]
