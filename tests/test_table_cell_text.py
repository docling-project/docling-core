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


def test_word_level_entries_on_one_line_stay_space_separated():
    """Native PDF text arrives as word entries; one visual line joins with spaces."""
    data = {
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 1,
        "bbox": {"l": 1.0, "t": 2.0, "r": 8.0, "b": 5.0},
        # Flat docling-ibm-models entry format: b/l/r/t/token keys.
        "text_cell_bboxes": [
            {"token": "Tertiary", "b": 5.0, "l": 4.0, "r": 8.0, "t": 2.0},
            {"token": "Contingent", "b": 5.0, "l": 1.0, "r": 3.5, "t": 2.0},
        ],
    }
    cell = TableCell.model_validate(data)
    assert cell.text == "Contingent Tertiary"


def test_word_level_entries_rebuild_stacked_lines():
    """Stacked word entries form separate lines, top row first."""
    data = {
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 1,
        "bbox": {"l": 1.0, "t": 2.0, "r": 8.0, "b": 12.0},
        "text_cell_bboxes": [
            # Second PDF row, tokens out of horizontal order on purpose.
            {"token": "281", "b": 12.0, "l": 1.0, "r": 2.5, "t": 9.0},
            {"token": "K123", "b": 12.0, "l": 3.0, "r": 6.0, "t": 9.0},
            # First PDF row.
            {"token": "63", "b": 5.0, "l": 4.5, "r": 6.0, "t": 2.0},
            {"token": "A9", "b": 5.0, "l": 1.0, "r": 2.5, "t": 2.0},
        ],
    }
    cell = TableCell.model_validate(data)
    assert cell.text == "A9 63\n281 K123"


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
