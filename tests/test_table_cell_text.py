"""Test the TableCell text assembly from docling-ibm-models output."""

from docling_core.types.doc.items.table.table_data import TableCell


def test_multiline_cell_text_preserves_line_structure():
    """A cell that covers several text lines keeps one line per source line."""
    data = {
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 1,
        "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 9.0},
        "text_cell_bboxes": [
            {"token": "Expedited", "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 5.0}},
            {"token": "Overnight", "bbox": {"l": 1.0, "t": 6.0, "r": 3.0, "b": 9.0}},
        ],
    }
    cell = TableCell.model_validate(data)
    assert cell.text == "Expedited\nOvernight"


def test_single_line_cell_text_has_no_separator():
    """A one-line cell keeps its plain text without any line separator."""
    data = {
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 1,
        "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 5.0},
        "text_cell_bboxes": [
            {"token": "Expedited", "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 5.0}},
        ],
    }
    cell = TableCell.model_validate(data)
    assert cell.text == "Expedited"


def test_empty_text_cell_list_gives_empty_text():
    """Cell data with no matched text lines yields an empty cell text."""
    data = {
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 1,
        "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 5.0},
        "text_cell_bboxes": [],
    }
    cell = TableCell.model_validate(data)
    assert cell.text == ""


def test_bbox_token_takes_precedence_over_lines():
    """When the aggregate bbox provides a token, it stays the cell text."""
    data = {
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 1,
        "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 9.0, "token": "Expedited"},
        "text_cell_bboxes": [
            {"token": "Expedited", "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 5.0}},
        ],
    }
    cell = TableCell.model_validate(data)
    assert cell.text == "Expedited"
