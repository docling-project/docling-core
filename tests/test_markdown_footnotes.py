from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
from docling_core.types.doc.document import (
    CodeItem,
    FieldHeadingItem,
    FieldValueItem,
    FormulaItem,
    ListItem,
    SectionHeaderItem,
    TextItem,
    TitleItem,
)

# Mock footnotes for pre-serialization
numericFtnMock = "1 Note about data"
wordFtnMock = "ID Note about data"
idOnlyFtnMock = "ID"

# Mock footnotes for post-serialization
numericFtnSerialized = "[^1]: Note about data\n"
wordFtnSerialized = "[^ID]: Note about data\n"
idOnlyFtnSerialized = "[^ID]:\n"


def test_table_with_footnotes_markdown():
    doc = DoclingDocument(name="test")

    table = doc.add_table(data=TableData())

    # Test three types of footnotes on table
    footnote1: TextItem = doc.add_text(label=DocItemLabel.FOOTNOTE, text=numericFtnMock)
    footnote2: TextItem = doc.add_text(label=DocItemLabel.FOOTNOTE, text=wordFtnMock)
    footnote3: TextItem = doc.add_text(label=DocItemLabel.FOOTNOTE, text=idOnlyFtnMock)

    table.footnotes.append(footnote1.get_ref())
    table.footnotes.append(footnote2.get_ref())
    table.footnotes.append(footnote3.get_ref())

    serializer = MarkdownDocSerializer(doc=doc)

    result = serializer.serialize(item=table)

    # Verify serialization result has formatted footnotes
    assert numericFtnSerialized in result.text
    assert wordFtnSerialized in result.text
    assert idOnlyFtnSerialized in result.text


def test_picture_with_footnotes_markdown():
    doc = DoclingDocument(name="test")

    picture = doc.add_picture()

    # Test one footnote on picture
    footnote1: TextItem = doc.add_text(label=DocItemLabel.FOOTNOTE, text=numericFtnMock)

    picture.footnotes.append(footnote1.get_ref())

    serializer = MarkdownDocSerializer(doc=doc)
    result = serializer.serialize(item=picture)

    # Verify serialization result has formatted footnote
    assert numericFtnSerialized in result.text


def test_table_export_to_markdown_with_footnotes():
    doc = DoclingDocument(name="test")

    # Create a table
    table_data = TableData(
        num_rows=2,
        num_cols=2,
        table_cells=[
            TableCell(
                text="Header 1",
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=0,
                end_col_offset_idx=0,
                column_header=True,
            ),
            TableCell(
                text="Header 2",
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=1,
                end_col_offset_idx=1,
                column_header=True,
            ),
            TableCell(
                text="Data 1",
                row_span=1,
                col_span=1,
                start_row_offset_idx=1,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=0,
            ),
            TableCell(
                text="Data 2",
                row_span=1,
                col_span=1,
                start_row_offset_idx=1,
                end_row_offset_idx=1,
                start_col_offset_idx=1,
                end_col_offset_idx=1,
            ),
        ],
    )

    table = doc.add_table(data=table_data)

    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1: Sample Data")
    table.captions.append(caption.get_ref())

    # Test one footnote on picture
    footnote1 = doc.add_text(label=DocItemLabel.FOOTNOTE, text=numericFtnMock)

    table.footnotes.append(footnote1.get_ref())

    markdown = table.export_to_markdown(doc)

    # Test Table is in exported markdown
    assert "Table 1: Sample Data" in markdown

    # Test Footnote is in exported markdown
    assert numericFtnSerialized in markdown


if __name__ == "__main__":
    test_table_with_footnotes_markdown()
    test_picture_with_footnotes_markdown()
    test_table_export_to_markdown_with_footnotes()
