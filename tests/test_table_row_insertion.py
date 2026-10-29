# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

"""Regression coverage for row insertion into tables with non-dense cells."""

from copy import deepcopy

import pytest

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    ProvenanceItem,
    RichTableCell,
    TableCell,
    TableData,
)


def _cell(text: str, row: int, column: int, *, height: int = 1, width: int = 1) -> TableCell:
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + height,
        start_col_offset_idx=column,
        end_col_offset_idx=column + width,
        row_span=height,
        col_span=width,
    )


def _stock_table() -> TableData:
    return TableData(
        num_rows=3,
        num_cols=2,
        table_cells=[
            _cell("Stock", 0, 0, width=2),
            _cell("Bolt", 1, 0),
            _cell("10", 1, 1),
            _cell("Nut", 2, 0),
            _cell("20", 2, 1),
        ],
    )


@pytest.mark.parametrize(("row_index", "after"), [(0, False), (0, True), (1, False), (2, True)])
def test_insert_row_preserves_column_spans(row_index: int, after: bool) -> None:
    data = _stock_table()
    expected = [["Stock", "Stock"], ["Bolt", "10"], ["Nut", "20"]]
    expected.insert(row_index + int(after), ["Washer", "30"])

    data.insert_row(row_index, ["Washer", "30"], after=after)

    assert [[cell.text for cell in row] for row in data.grid] == expected
    assert data.num_rows == 4
    assert next(cell for cell in data.table_cells if cell.text == "Stock").col_span == 2


def test_insert_rows_preserves_sparse_unordered_cells() -> None:
    cells = [_cell("last", 2, 1), _cell("first", 0, 0), _cell("middle", 1, 1)]
    data = TableData(num_rows=3, num_cols=2, table_cells=cells)

    data.insert_rows(1, [["new-a", "1"], ["new-b", "2"]])

    assert [[cell.text for cell in row] for row in data.grid] == [
        ["first", ""],
        ["new-a", "1"],
        ["new-b", "2"],
        ["", "middle"],
        ["", "last"],
    ]
    assert all(any(cell is original for cell in data.table_cells) for original in cells)


@pytest.mark.parametrize("after", [False, True])
def test_insert_multiple_rows_keeps_input_order(after: bool) -> None:
    data = _stock_table()
    data.insert_rows(0 if after else 1, [["Washer", "30"], ["Screw", "40"]], after=after)

    assert [[cell.text for cell in row] for row in data.grid] == [
        ["Stock", "Stock"],
        ["Washer", "30"],
        ["Screw", "40"],
        ["Bolt", "10"],
        ["Nut", "20"],
    ]


@pytest.mark.parametrize("row_index", [0, 2, 3])
def test_insert_at_vertical_span_boundaries_preserves_span(row_index: int) -> None:
    spanning = _cell("group", 0, 0, height=2)
    data = TableData(
        num_rows=3,
        num_cols=2,
        table_cells=[spanning, _cell("A", 0, 1), _cell("B", 1, 1), _cell("C", 2, 0), _cell("D", 2, 1)],
    )
    expected = [["group", "A"], ["group", "B"], ["C", "D"]]
    expected.insert(row_index, ["new", "value"])

    data.insert_row(row_index, ["new", "value"])

    assert [[cell.text for cell in row] for row in data.grid] == expected
    assert spanning.row_span == 2
    assert spanning.end_row_offset_idx - spanning.start_row_offset_idx == 2


@pytest.mark.parametrize(("index", "after"), [(1, False), (0, True)])
def test_insert_through_vertical_span_is_rejected_without_mutation(index: int, after: bool) -> None:
    data = TableData(num_rows=2, num_cols=2, table_cells=[_cell("merged", 0, 0, height=2, width=2)])
    before = data.model_dump()

    with pytest.raises(ValueError, match="spanning cell"):
        data.insert_rows(index, [["new", "values"], ["more", "values"]], after=after)

    assert data.model_dump() == before


@pytest.mark.parametrize("rows", [[["wrong width"]], [["wrong width"], ["valid", "width"]]])
def test_insert_rows_validates_entire_batch_before_mutation(rows: list[list[str]]) -> None:
    data = _stock_table()
    before = data.model_dump()

    with pytest.raises(ValueError, match="Row length"):
        data.insert_rows(1, rows)

    assert data.model_dump() == before


@pytest.mark.parametrize(("index", "after"), [(-1, False), (-2, True), (4, False), (3, True)])
def test_insert_rows_rejects_out_of_range_without_mutation(index: int, after: bool) -> None:
    data = _stock_table()
    before = data.model_dump()

    with pytest.raises(IndexError):
        data.insert_rows(index, [["new", "row"]], after=after)

    assert data.model_dump() == before


def test_insert_preserves_rich_content_and_source_locations() -> None:
    doc = DoclingDocument(name="synthetic table with rich content")
    bbox = BoundingBox(l=10, t=30, r=90, b=50, coord_origin=CoordOrigin.TOPLEFT)
    table = doc.add_table(
        data=TableData(num_rows=1, num_cols=2),
        prov=ProvenanceItem(page_no=2, bbox=bbox, charspan=(0, 0)),
    )
    group = doc.add_group(parent=table)
    doc.add_text(label=DocItemLabel.TEXT, text="Source value", parent=group, formatting=Formatting(italic=True))
    rich = RichTableCell(
        text="",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=2,
        col_span=2,
        bbox=bbox,
        column_header=True,
        ref=group.get_ref(),
    )
    doc.add_table_cell(table_item=table, cell=rich)
    before = deepcopy(rich.model_dump())
    original_prov = deepcopy(table.prov)

    table.data.insert_row(0, ["new", "values"])

    after = rich.model_dump()
    assert after.pop("start_row_offset_idx") == 1
    assert after.pop("end_row_offset_idx") == 2
    before.pop("start_row_offset_idx")
    before.pop("end_row_offset_idx")
    assert after == before
    assert table.prov == original_prov
    assert rich.ref.resolve(doc) is group
    assert table.data.grid[1][0] is rich
    assert table.data.grid[1][1] is rich
    assert all(cell.bbox is None for cell in table.data.grid[0])
    assert "Source value" in doc.export_to_html()
    loaded = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert "Source value" in loaded.export_to_markdown()


def test_add_rows_preserves_existing_merged_cells() -> None:
    data = _stock_table()
    data.add_rows([["Washer", "30"], ["Screw", "40"]])

    assert [[cell.text for cell in row] for row in data.grid] == [
        ["Stock", "Stock"],
        ["Bolt", "10"],
        ["Nut", "20"],
        ["Washer", "30"],
        ["Screw", "40"],
    ]


def test_add_to_empty_table_and_empty_batch() -> None:
    data = TableData(num_rows=0, num_cols=2)
    data.add_row(["first", "row"])
    before = data.model_dump()
    data.insert_rows(100, [])

    assert [[cell.text for cell in row] for row in data.grid] == [["first", "row"]]
    assert data.model_dump() == before


def test_append_preserves_existing_empty_rows() -> None:
    data = TableData(num_rows=1, num_cols=1)
    data.add_row(["appended"])

    assert [[cell.text for cell in row] for row in data.grid] == [[""], ["appended"]]


def test_insert_row_rejects_vertical_span_without_mutation() -> None:
    data = TableData(num_rows=2, num_cols=1, table_cells=[_cell("source", 0, 0, height=2)])
    before = data.model_dump()

    with pytest.raises(ValueError, match="spanning cell"):
        data.insert_row(0, ["new"], after=True)

    assert data.model_dump() == before


@pytest.mark.parametrize("rows", [[[None, "invalid"], ["valid", "row"]]])
def test_insert_rows_cell_validation_failure_does_not_mutate(rows: list[list[str]]) -> None:
    data = _stock_table()
    before = data.model_dump()

    with pytest.raises(ValueError):
        data.insert_rows(1, rows)

    assert data.model_dump() == before


def test_add_row_preserves_vertical_span_and_source_fields() -> None:
    source = _cell("original", 0, 0, height=2)
    source.bbox = BoundingBox(l=10, t=20, r=30, b=40, coord_origin=CoordOrigin.TOPLEFT)
    source.row_header = True
    source.row_section = True
    source.fillable = True
    data = TableData(num_rows=2, num_cols=2, table_cells=[source, _cell("A", 0, 1), _cell("B", 1, 1)])
    before = [cell.model_dump() for cell in data.table_cells]

    data.add_row(["new", "values"])

    assert [[cell.text for cell in row] for row in data.grid] == [
        ["original", "A"],
        ["original", "B"],
        ["new", "values"],
    ]
    assert [cell.model_dump() for cell in data.table_cells[:3]] == before


def test_insert_preserves_offsets_before_boundary_and_shifts_both_offsets_below() -> None:
    data = _stock_table()
    originals = list(data.table_cells)

    data.insert_rows(1, [["A", "11"], ["B", "22"]])

    assert [(cell.start_row_offset_idx, cell.end_row_offset_idx) for cell in originals] == [
        (0, 1),
        (3, 4),
        (3, 4),
        (4, 5),
        (4, 5),
    ]


def test_insert_replaces_cell_list_without_replacing_source_cells() -> None:
    data = TableData(num_rows=1, num_cols=1, table_cells=[_cell("source", 0, 0)])
    previous_list = data.table_cells
    source = previous_list[0]

    data.insert_row(0, ["new"])

    assert data.table_cells is not previous_list
    assert len(previous_list) == 1
    assert previous_list[0] is source
    assert data.table_cells[1] is source
