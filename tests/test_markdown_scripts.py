"""Regression tests for script formatting in Markdown export."""

import pytest

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import DocItemLabel, DoclingDocument, Formatting, RichTableCell, Script, TableData


@pytest.mark.parametrize("script,tag", [(Script.SUPER, "sup"), (Script.SUB, "sub")])
@pytest.mark.parametrize("in_table", [False, True])
def test_markdown_script_formatting(script, tag, in_table):
    doc = DoclingDocument(name="scripts")
    table = doc.add_table(data=TableData()) if in_table else None
    group = doc.add_inline_group(parent=table)
    doc.add_text(label=DocItemLabel.TEXT, text="x", parent=group)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="a<b",
        formatting=Formatting(script=script),
        parent=group,
    )
    if table is not None:
        table.data = TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                RichTableCell(
                    text="x a<b",
                    ref=group.get_ref(),
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                )
            ],
        )
    markdown = doc.export_to_markdown()
    assert f"<{tag}>a&lt;b</{tag}>" in markdown
    unformatted = (
        MarkdownDocSerializer(
            doc=doc,
            params=MarkdownParams(include_formatting=False),
        )
        .serialize()
        .text
    )
    assert f"<{tag}>" not in unformatted
    assert "a&lt;b" in unformatted
    plain_text = doc.export_to_text()
    assert f"<{tag}>" not in plain_text
    assert "a<b" in plain_text
