"""Define options across tests."""

from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import AnyUrl

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    CodeLanguageLabel,
    DocItemLabel,
    Formatting,
    GraphCell,
    GraphCellLabel,
    GraphData,
    GraphLink,
    GraphLinkLabel,
    GroupLabel,
    ImageRef,
    RichTableCell,
    Script,
    TableCell,
    TableData,
)


# factored out of fixture to simplify IDE-level debugging
def _construct_doc_impl() -> DoclingDocument:
    """Fixture for a DoclingDocument to be reused across a test session."""

    doc = DoclingDocument(name="Untitled 1")

    leading_list = doc.add_list_group(parent=None)
    doc.add_list_item(parent=leading_list, text="item of leading list", marker="■")

    title = doc.add_title(
        text="Title of the Document"
    )  # can be done if such information is present, or ommitted.

    # group, heading, paragraph, table, figure, title, list, provenance
    doc.add_text(parent=title, label=DocItemLabel.TEXT, text="Author 1\nAffiliation 1")
    doc.add_text(parent=title, label=DocItemLabel.TEXT, text="Author 2\nAffiliation 2")

    chapter1 = doc.add_group(
        label=GroupLabel.CHAPTER, name="Introduction"
    )  # can be done if such information is present, or ommitted.

    doc.add_heading(
        parent=chapter1,
        text="1. Introduction",
        level=1,
    )
    doc.add_text(
        parent=chapter1,
        label=DocItemLabel.TEXT,
        text="This paper introduces the biggest invention ever made. ...",
    )

    mylist_level_1 = doc.add_list_group(parent=chapter1)

    doc.add_list_item(parent=mylist_level_1, text="list item 1", marker="■")
    doc.add_list_item(parent=mylist_level_1, text="list item 2", marker="■")
    li3 = doc.add_list_item(parent=mylist_level_1, text="list item 3", marker="■")

    mylist_level_2 = doc.add_list_group(parent=li3)

    doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.a",
        enumerated=True,
    )
    doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.b",
        enumerated=True,
    )
    li3c = doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.c",
        enumerated=True,
    )

    mylist_level_3 = doc.add_list_group(parent=li3c)

    doc.add_list_item(
        parent=mylist_level_3,
        text="list item 3.c.i",
        enumerated=True,
    )

    doc.add_list_item(parent=mylist_level_1, text="list item 4", marker="■")

    tab_caption = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of table 1."
    )

    # Make some table cells
    table_cells = []
    table_cells.append(
        TableCell(
            row_span=2,
            start_row_offset_idx=0,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Product",
        )
    )
    table_cells.append(
        TableCell(
            col_span=2,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=3,
            text="Years",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="2016",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            text="2017",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Apple",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="49823",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            text="695944",
        )
    )
    table_data = TableData(num_rows=3, num_cols=3, table_cells=table_cells)
    doc.add_table(data=table_data, caption=tab_caption)

    fig_caption_1 = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of figure 1."
    )
    doc.add_picture(caption=fig_caption_1)

    size = (64, 64)
    fig2_image = PILImage.new("RGB", size, "black")

    # Draw a red disk touching the borders
    # draw = ImageDraw.Draw(fig2_image)
    # draw.ellipse((0, 0, size[0] - 1, size[1] - 1), fill="red")

    # Create a drawing object
    ImageDraw.Draw(fig2_image)

    # Define the coordinates of the red square (x1, y1, x2, y2)
    # square_size = 20  # Adjust as needed
    # x1, y1 = 22, 22  # Adjust position
    # x2, y2 = x1 + square_size, y1 + square_size

    # Draw the red square
    # draw.rectangle([x1, y1, x2, y2], fill="red")

    fig_caption_2 = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of figure 2."
    )
    doc.add_picture(
        image=ImageRef.from_pil(image=fig2_image, dpi=72), caption=fig_caption_2
    )

    g0 = doc.add_list_group(parent=None)
    doc.add_list_item(text="item 1 of list", parent=g0, marker="■")

    # an empty list
    doc.add_list_group(parent=None)

    g1 = doc.add_list_group(parent=None)
    doc.add_list_item(text="item 1 of list after empty list", parent=g1, marker="*")
    doc.add_list_item(text="item 2 of list after empty list", parent=g1, marker="")

    g2 = doc.add_list_group(parent=None)
    doc.add_list_item(text="item 1 of neighboring list", parent=g2, marker="■")
    nli2 = doc.add_list_item(text="item 2 of neighboring list", parent=g2, marker="■")

    g2_subgroup = doc.add_list_group(parent=nli2)
    doc.add_list_item(text="item 1 of sub list", parent=g2_subgroup, marker="□")

    g2_subgroup_li_1 = doc.add_list_item(text="", parent=g2_subgroup, marker="□")
    inline1 = doc.add_inline_group(parent=g2_subgroup_li_1)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a code snippet:",
        parent=inline1,
    )
    doc.add_code(
        text='print("Hello world")',
        parent=inline1,
        code_language=CodeLanguageLabel.PYTHON,
    )
    doc.add_text(
        label=DocItemLabel.TEXT, text="(to be displayed inline)", parent=inline1
    )

    g2_subgroup_li_2 = doc.add_list_item(text="", parent=g2_subgroup, marker="□")
    inline2 = doc.add_inline_group(parent=g2_subgroup_li_2)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a formula:",
        parent=inline2,
    )
    doc.add_text(label=DocItemLabel.FORMULA, text="E=mc^2", parent=inline2)
    doc.add_text(
        label=DocItemLabel.TEXT, text="(to be displayed inline)", parent=inline2
    )

    doc.add_text(label=DocItemLabel.TEXT, text="Here a code block:", parent=None)
    doc.add_code(
        text='print("Hello world")', parent=None, code_language=CodeLanguageLabel.PYTHON
    )

    doc.add_text(label=DocItemLabel.TEXT, text="Here a formula block:", parent=None)
    doc.add_text(label=DocItemLabel.FORMULA, text="E=mc^2", parent=None)

    graph = GraphData(
        cells=[
            GraphCell(
                label=GraphCellLabel.KEY,
                cell_id=0,
                text="number",
                orig="#",
            ),
            GraphCell(
                label=GraphCellLabel.VALUE,
                cell_id=1,
                text="1",
                orig="1",
            ),
        ],
        links=[
            GraphLink(
                label=GraphLinkLabel.TO_VALUE,
                source_cell_id=0,
                target_cell_id=1,
            ),
            GraphLink(label=GraphLinkLabel.TO_KEY, source_cell_id=1, target_cell_id=0),
        ],
    )

    doc.add_key_values(graph=graph)

    doc.add_form(graph=graph)

    inline_fmt = doc.add_inline_group()
    doc.add_text(
        label=DocItemLabel.TEXT, text="Some formatting chops:", parent=inline_fmt
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="bold",
        parent=inline_fmt,
        formatting=Formatting(bold=True),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="italic",
        parent=inline_fmt,
        formatting=Formatting(italic=True),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="underline",
        parent=inline_fmt,
        formatting=Formatting(underline=True),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="strikethrough",
        parent=inline_fmt,
        formatting=Formatting(strikethrough=True),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="subscript",
        orig="subscript",
        formatting=Formatting(script=Script.SUB),
        parent=inline_fmt,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="superscript",
        orig="superscript",
        formatting=Formatting(script=Script.SUPER),
        parent=inline_fmt,
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="hyperlink",
        parent=inline_fmt,
        hyperlink=Path("."),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="&", parent=inline_fmt)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="everything at the same time.",
        parent=inline_fmt,
        formatting=Formatting(
            bold=True,
            italic=True,
            underline=True,
            strikethrough=True,
        ),
        hyperlink=AnyUrl("https://github.com/DS4SD/docling"),
    )

    parent_A = doc.add_list_group(name="list A")
    doc.add_list_item(
        text="Item 1 in A", enumerated=True, marker="(i)", parent=parent_A
    )
    doc.add_list_item(
        text="Item 2 in A", enumerated=True, marker="(ii)", parent=parent_A
    )
    item_A_3 = doc.add_list_item(
        text="Item 3 in A", enumerated=True, marker="(iii)", parent=parent_A
    )

    parent_B = doc.add_list_group(parent=item_A_3, name="list B")
    doc.add_list_item(text="Item 1 in B", enumerated=True, parent=parent_B)
    item_B_2 = doc.add_list_item(
        text="Item 2 in B", enumerated=True, marker="42.", parent=parent_B
    )

    parent_C = doc.add_list_group(parent=item_B_2, name="list C")
    doc.add_list_item(text="Item 1 in C", enumerated=True, parent=parent_C)
    doc.add_list_item(text="Item 2 in C", enumerated=True, parent=parent_C)

    doc.add_list_item(text="Item 3 in B", enumerated=True, parent=parent_B)

    doc.add_list_item(
        text="Item 4 in A", enumerated=True, marker="(iv)", parent=parent_A
    )

    with pytest.warns(DeprecationWarning, match="list group"):
        doc.add_list_item(text="List item without parent list group")

    doc.add_text(label=DocItemLabel.TEXT, text="The end.", parent=None)

    return doc


@pytest.fixture(scope="session")
def _construct_doc() -> DoclingDocument:
    """Fixture for a DoclingDocument to be reused across a test session."""
    return _construct_doc_impl()


@pytest.fixture(scope="function")
def sample_doc(_construct_doc: DoclingDocument) -> DoclingDocument:
    """Copy of a DoclingDocument for each test function."""

    return _construct_doc.model_copy(deep=True)


@pytest.fixture(scope="session")
def _rich_table_doc() -> DoclingDocument:
    """Fixture for a rich table document to be reused across the test session."""

    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TITLE, text="Rich tables")

    table_item = doc.add_table(
        data=TableData(
            num_rows=5,
            num_cols=2,
        ),
    )

    rich_item_1 = doc.add_text(
        parent=table_item,
        text="text in italic",
        label=DocItemLabel.TEXT,
        formatting=Formatting(italic=True),
    )

    rich_item_2 = doc.add_list_group(parent=table_item)
    doc.add_list_item(parent=rich_item_2, text="list item 1")
    doc.add_list_item(parent=rich_item_2, text="list item 2")

    rich_item_3 = doc.add_table(
        data=TableData(num_rows=2, num_cols=3), parent=table_item
    )

    rich_item_4 = doc.add_group(parent=table_item, label=GroupLabel.UNSPECIFIED)
    doc.add_text(
        parent=rich_item_4,
        text="Some text in a generic group.",
        label=DocItemLabel.TEXT,
    )
    doc.add_text(
        parent=rich_item_4, text="More text in the group.", label=DocItemLabel.TEXT
    )

    for i in range(rich_item_3.data.num_rows):
        for j in range(rich_item_3.data.num_cols):
            cell = TableCell(
                text=f"inner cell {i},{j}",
                start_row_offset_idx=i,
                end_row_offset_idx=i + 1,
                start_col_offset_idx=j,
                end_col_offset_idx=j + 1,
            )
            doc.add_table_cell(table_item=rich_item_3, cell=cell)

    for i in range(table_item.data.num_rows):
        for j in range(table_item.data.num_cols):
            if i == 1 and j == 1:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item_1.get_ref(),
                    text=f"cell {i},{j}",
                )
            elif i == 2 and j == 0:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item_2.get_ref(),
                    text=f"cell {i},{j}",
                )
            elif i == 3 and j == 1:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item_3.get_ref(),
                    text=f"cell {i},{j}",
                )
            elif i == 4 and j == 0:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item_4.get_ref(),
                    text=f"cell {i},{j}",
                )
            else:
                cell = TableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    text=f"cell {i},{j}",
                )
            doc.add_table_cell(table_item=table_item, cell=cell)

    return doc


@pytest.fixture(scope="function")
def rich_table_doc(_rich_table_doc: DoclingDocument) -> DoclingDocument:
    """Copy of a rich table document for each test function."""

    return _rich_table_doc.model_copy(deep=True)
