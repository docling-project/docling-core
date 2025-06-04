from docling_core.types.doc.document import DoclingDocument, TableData

from .test_data_gen_flag import GEN_TEST_DATA


def _verify(act_data: str, exp_file: str):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{act_data}\n")
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            exp_data = f.read().rstrip()
        assert exp_data == act_data


def _construct_doc():
    num_cols = 3
    num_rows = 3

    # Construct a table by building data without cells first...
    doc = DoclingDocument(name="test_rich_table")
    data = TableData(num_rows=num_rows, num_cols=num_cols)
    tbl = doc.add_table(data=data)

    # Then add cells with the update_cell method.
    tbl.update_cell(
        text="AB",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=2,
        column_header=False,
        row_header=True,
    )

    tbl.update_cell(
        text="C",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=2,
        end_col_offset_idx=3,
        column_header=False,
        row_header=True,
    )

    tbl.update_cell(
        text="1",
        start_row_offset_idx=1,
        end_row_offset_idx=3,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        column_header=False,
        row_header=True,
    )

    tbl.update_cell(
        text="2",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=1,
        end_col_offset_idx=2,
        column_header=False,
        row_header=False,
    )

    rich_cell = tbl.update_cell(
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=2,
        end_col_offset_idx=3,
        column_header=False,
        row_header=False,
    )

    list_node = doc.add_unordered_list(name="inner list", parent=rich_cell)
    doc.add_list_item(text="foo", parent=list_node)
    doc.add_list_item(text="bar", parent=list_node)

    return doc


def test_rich_table():

    doc = _construct_doc()

    html_pred = doc.export_to_html()
    _verify(act_data=html_pred, exp_file="test/data/doc/rich_table_doc.html")
