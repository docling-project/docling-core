import os
from collections import deque
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock

import pytest
import yaml
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import AnyUrl, ValidationError

from docling_core.types.doc.base import BoundingBox, CoordOrigin, ImageRefMode, Size
from docling_core.types.doc.document import (  # BoundingBox,
    CURRENT_VERSION,
    CodeItem,
    ContentLayer,
    DocItem,
    DoclingDocument,
    DocumentOrigin,
    FloatingItem,
    Formatting,
    FormItem,
    FormulaItem,
    GraphCell,
    GraphData,
    GraphLink,
    ImageRef,
    KeyValueItem,
    ListItem,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    Size,
    TableCell,
    TableData,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import (
    DocItemLabel,
    GraphCellLabel,
    GraphLinkLabel,
    GroupLabel,
)

from .test_data_gen_flag import GEN_TEST_DATA


def test_doc_origin():
    doc_origin = DocumentOrigin(
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="myfile.pdf",
        binary_hash="50115d582a0897fe1dd520a6876ec3f9321690ed0f6cfdc99a8d09019be073e8",
    )


def test_overlaps_horizontally():
    # Overlapping horizontally
    bbox1 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2 = BoundingBox(l=5, t=5, r=15, b=15, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_horizontally(bbox2) is True

    # No overlap horizontally (disjoint on the right)
    bbox3 = BoundingBox(l=11, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_horizontally(bbox3) is False

    # No overlap horizontally (disjoint on the left)
    bbox4 = BoundingBox(l=-10, t=0, r=-1, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_horizontally(bbox4) is False

    # Full containment
    bbox5 = BoundingBox(l=2, t=2, r=8, b=8, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_horizontally(bbox5) is True

    # Edge touching (no overlap)
    bbox6 = BoundingBox(l=10, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_horizontally(bbox6) is False


def test_overlaps_vertically():

    page_height = 300

    # Same CoordOrigin (TOPLEFT)
    bbox1 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2 = BoundingBox(l=5, t=5, r=15, b=15, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_vertically(bbox2) is True

    bbox1_ = bbox1.to_bottom_left_origin(page_height=page_height)
    bbox2_ = bbox2.to_bottom_left_origin(page_height=page_height)
    assert bbox1_.overlaps_vertically(bbox2_) is True

    bbox3 = BoundingBox(l=0, t=11, r=10, b=20, coord_origin=CoordOrigin.TOPLEFT)
    assert bbox1.overlaps_vertically(bbox3) is False

    bbox3_ = bbox3.to_bottom_left_origin(page_height=page_height)
    assert bbox1_.overlaps_vertically(bbox3_) is False

    # Same CoordOrigin (BOTTOMLEFT)
    bbox4 = BoundingBox(l=0, b=20, r=10, t=30, coord_origin=CoordOrigin.BOTTOMLEFT)
    bbox5 = BoundingBox(l=5, b=15, r=15, t=25, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert bbox4.overlaps_vertically(bbox5) is True

    bbox4_ = bbox4.to_top_left_origin(page_height=page_height)
    bbox5_ = bbox5.to_top_left_origin(page_height=page_height)
    assert bbox4_.overlaps_vertically(bbox5_) is True

    bbox6 = BoundingBox(l=0, b=31, r=10, t=40, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert bbox4.overlaps_vertically(bbox6) is False

    bbox6_ = bbox6.to_top_left_origin(page_height=page_height)
    assert bbox4_.overlaps_vertically(bbox6_) is False

    # Different CoordOrigin
    with pytest.raises(ValueError):
        bbox1.overlaps_vertically(bbox4)


def test_intersection_area_with():
    page_height = 300

    # Overlapping bounding boxes (TOPLEFT)
    bbox1 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2 = BoundingBox(l=5, t=5, r=15, b=15, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.intersection_area_with(bbox2) - 25.0) < 1.0e-3

    bbox1_ = bbox1.to_bottom_left_origin(page_height=page_height)
    bbox2_ = bbox2.to_bottom_left_origin(page_height=page_height)
    assert abs(bbox1_.intersection_area_with(bbox2_) - 25.0) < 1.0e-3

    # Non-overlapping bounding boxes (TOPLEFT)
    bbox3 = BoundingBox(l=11, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.intersection_area_with(bbox3) - 0.0) < 1.0e-3

    # Touching edges (no intersection, TOPLEFT)
    bbox4 = BoundingBox(l=10, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.intersection_area_with(bbox4) - 0.0) < 1.0e-3

    # Fully contained (TOPLEFT)
    bbox5 = BoundingBox(l=2, t=2, r=8, b=8, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.intersection_area_with(bbox5) - 36.0) < 1.0e-3

    # Overlapping bounding boxes (BOTTOMLEFT)
    bbox6 = BoundingBox(l=0, t=10, r=10, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
    bbox7 = BoundingBox(l=5, t=15, r=15, b=5, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert abs(bbox6.intersection_area_with(bbox7) - 25.0) < 1.0e-3

    # Different CoordOrigins (raises ValueError)
    with pytest.raises(ValueError):
        bbox1.intersection_area_with(bbox6)


def test_orientation():

    page_height = 300

    # Same CoordOrigin (TOPLEFT)
    bbox1 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2 = BoundingBox(l=5, t=5, r=15, b=15, coord_origin=CoordOrigin.TOPLEFT)
    bbox3 = BoundingBox(l=11, t=5, r=15, b=15, coord_origin=CoordOrigin.TOPLEFT)
    bbox4 = BoundingBox(l=0, t=11, r=10, b=15, coord_origin=CoordOrigin.TOPLEFT)

    assert bbox1.is_left_of(bbox2) is True
    assert bbox1.is_strictly_left_of(bbox2) is False
    assert bbox1.is_strictly_left_of(bbox3) is True

    bbox1_ = bbox1.to_bottom_left_origin(page_height=page_height)
    bbox2_ = bbox2.to_bottom_left_origin(page_height=page_height)
    bbox3_ = bbox3.to_bottom_left_origin(page_height=page_height)
    bbox4_ = bbox4.to_bottom_left_origin(page_height=page_height)

    assert bbox1.is_above(bbox2) is True
    assert bbox1_.is_above(bbox2_) is True
    assert bbox1.is_strictly_above(bbox4) is True
    assert bbox1_.is_strictly_above(bbox4_) is True


def test_docitems():

    # Iterative function to find all subclasses
    def find_all_subclasses_iterative(base_class):
        subclasses = deque(
            [base_class]
        )  # Use a deque for efficient popping from the front
        all_subclasses = []

        while subclasses:
            current_class = subclasses.popleft()  # Get the next class to process
            for subclass in current_class.__subclasses__():
                all_subclasses.append(subclass)
                subclasses.append(subclass)  # Add the subclass for further exploration

        return all_subclasses

    def serialise(obj):
        return yaml.safe_dump(obj.model_dump(mode="json", by_alias=True))

    def write(name: str, serialisation: str):
        with open(
            f"./test/data/docling_document/unit/{name}.yaml", "w", encoding="utf-8"
        ) as fw:
            fw.write(serialisation)

    def read(name: str):
        with open(
            f"./test/data/docling_document/unit/{name}.yaml", "r", encoding="utf-8"
        ) as fr:
            gold = fr.read()
        return yaml.safe_load(gold)

    def verify(dc, obj):
        pred = serialise(obj).strip()

        if dc is KeyValueItem or dc is FormItem:
            write(dc.__name__, pred)

        pred = yaml.safe_load(pred)

        # print(f"\t{dc.__name__}:\n {pred}")
        gold = read(dc.__name__)

        assert pred == gold, f"pred!=gold for {dc.__name__}"

    # Iterate over the derived classes of the BaseClass
    derived_classes = find_all_subclasses_iterative(DocItem)
    for dc in derived_classes:

        if dc is TextItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                label=DocItemLabel.TEXT,
                self_ref="#",
            )
            verify(dc, obj)
        elif dc is ListItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                marker="(1)",
                enumerated=True,
                self_ref="#",
            )
            verify(dc, obj)
        elif dc is FloatingItem:
            obj = dc(
                label=DocItemLabel.TEXT,
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is KeyValueItem:

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
                    GraphLink(
                        label=GraphLinkLabel.TO_KEY, source_cell_id=1, target_cell_id=0
                    ),
                ],
            )

            obj = dc(
                label=DocItemLabel.KEY_VALUE_REGION,
                graph=graph,
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is FormItem:

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
                    GraphLink(
                        label=GraphLinkLabel.TO_KEY, source_cell_id=1, target_cell_id=0
                    ),
                ],
            )

            obj = dc(
                label=DocItemLabel.FORM,
                graph=graph,
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is TitleItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                label=DocItemLabel.TITLE,
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is SectionHeaderItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                label=DocItemLabel.SECTION_HEADER,
                self_ref="#",
                level=2,
            )
            verify(dc, obj)

        elif dc is PictureItem:
            obj = dc(
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is TableItem:
            obj = dc(
                self_ref="#",
                data=TableData(num_rows=3, num_cols=5, table_cells=[]),
            )
            verify(dc, obj)
        elif dc is CodeItem:
            obj = dc(
                self_ref="#",
                orig="whatever",
                text="print(Hello World!)",
                code_language="Python",
            )
            verify(dc, obj)
        elif dc is FormulaItem:
            obj = dc(
                self_ref="#",
                orig="whatever",
                text="E=mc^2",
            )
            verify(dc, obj)
        elif dc is GraphData:  # we skip this on purpose
            continue
        else:
            raise RuntimeError(f"New derived class detected {dc.__name__}")


def test_reference_doc():

    filename = "test/data/doc/dummy_doc.yaml"

    # Read YAML file of manual reference doc
    with open(filename, "r", encoding="utf-8") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    # Objects can be accessed
    text_item = doc.texts[0]

    # access members
    text_item.text
    text_item.prov[0].page_no

    # Objects that are references need explicit resolution for now:
    obj = doc.texts[2]  # Text item with parent
    parent = obj.parent.resolve(doc=doc)  # it is a figure

    obj2 = parent.children[0].resolve(
        doc=doc
    )  # Child of figure must be the same as obj

    assert obj == obj2
    assert obj is obj2

    # Iterate all elements

    for item, level in doc.iterate_items():
        _ = f"Item: {item} at level {level}"
        # print(f"Item: {item} at level {level}")

    # Serialize and reload
    _test_serialize_and_reload(doc)

    # Call Export methods
    _test_export_methods(doc, filename=filename)


def test_parse_doc():

    filename = "test/data/doc/2206.01062.yaml"

    with open(filename, "r", encoding="utf-8") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    page_break = "<!-- page break -->"
    _test_export_methods(doc, filename=filename, page_break_placeholder=page_break)
    _test_serialize_and_reload(doc)


def test_construct_doc():

    filename = "test/data/doc/constructed_document.yaml"

    doc = _construct_doc()

    assert doc.validate_tree(doc.body)
    assert doc.validate_tree(doc.furniture)

    _test_export_methods(doc, filename=filename)
    _test_serialize_and_reload(doc)


def test_construct_bad_doc():

    filename = "test/data/doc/bad_doc.yaml"

    doc = _construct_bad_doc()
    assert doc.validate_tree(doc.body) == False

    with pytest.raises(ValueError):
        _test_export_methods(doc, filename=filename)
    with pytest.raises(ValueError):
        _test_serialize_and_reload(doc)


def _test_serialize_and_reload(doc):
    ### Serialize and deserialize stuff
    yaml_dump = yaml.safe_dump(doc.model_dump(mode="json", by_alias=True))
    # print(f"\n\n{yaml_dump}")
    doc_reload = DoclingDocument.model_validate(yaml.safe_load(yaml_dump))

    yaml_dump_reload = yaml.safe_dump(doc_reload.model_dump(mode="json", by_alias=True))

    assert yaml_dump == yaml_dump_reload, "yaml_dump!=yaml_dump_reload"

    """
    for item, level in doc.iterate_items():
        if isinstance(item, PictureItem):
            _ = item.get_image(doc)

    assert doc_reload == doc  # must be equal
    """

    assert doc_reload is not doc  # can't be identical


def _verify_regression_test(pred: str, filename: str, ext: str):
    if os.path.exists(filename + f".{ext}") and not GEN_TEST_DATA:
        with open(filename + f".{ext}", "r", encoding="utf-8") as fr:
            gt_true = fr.read().rstrip()

        assert gt_true == pred, f"Does not pass regression-test for {filename}.{ext}"
    else:
        with open(filename + f".{ext}", "w", encoding="utf-8") as fw:
            fw.write(f"{pred}\n")


def _test_export_methods(
    doc: DoclingDocument, filename: str, page_break_placeholder: Optional[str] = None
):
    # Iterate all elements
    et_pred = doc.export_to_element_tree()
    _verify_regression_test(et_pred, filename=filename, ext="et")

    # Export stuff
    md_pred = doc.export_to_markdown()
    _verify_regression_test(md_pred, filename=filename, ext="md")

    if page_break_placeholder is not None:
        md_pred = doc.export_to_markdown(page_break_placeholder=page_break_placeholder)
        _verify_regression_test(md_pred, filename=filename, ext="paged.md")

    # Test sHTML export ...
    html_pred = doc.export_to_html()
    _verify_regression_test(html_pred, filename=filename, ext="html")

    # Test DocTags export ...
    dt_pred = doc.export_to_doctags()
    _verify_regression_test(dt_pred, filename=filename, ext="dt")

    dt_min_pred = doc.export_to_doctags(minified=True)
    _verify_regression_test(dt_min_pred, filename=filename, ext="min.dt")

    # Test Tables export ...
    for table in doc.tables:
        table.export_to_markdown()
        table.export_to_html(doc)
        table.export_to_dataframe()
        table.export_to_doctags(doc)

    # Test Images export ...

    for fig in doc.pictures:
        fig.export_to_doctags(doc)


def _construct_bad_doc():
    doc = DoclingDocument(name="Bad doc")

    title = doc.add_text(label=DocItemLabel.TITLE, text="This is the title")
    group = doc.add_group(parent=title, name="chapter 1")
    text = doc.add_text(
        parent=group,
        label=DocItemLabel.SECTION_HEADER,
        text="This is the first section",
    )

    # Bend the parent of an element to be another.
    text.parent = title.get_ref()

    return doc


def _construct_doc() -> DoclingDocument:

    doc = DoclingDocument(name="Untitled 1")

    leading_list = doc.add_group(parent=None, label=GroupLabel.LIST)
    doc.add_list_item(parent=leading_list, text="item of leading list")

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

    mylist_level_1 = doc.add_group(parent=chapter1, label=GroupLabel.LIST)

    doc.add_list_item(
        parent=mylist_level_1,
        text="list item 1",
    )
    doc.add_list_item(parent=mylist_level_1, text="list item 2")
    li3 = doc.add_list_item(
        parent=mylist_level_1,
        text="list item 3",
    )

    mylist_level_2 = doc.add_group(parent=li3, label=GroupLabel.ORDERED_LIST)

    doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.a",
    )
    doc.add_list_item(parent=mylist_level_2, text="list item 3.b")
    li3c = doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.c",
    )

    mylist_level_3 = doc.add_group(parent=li3c, label=GroupLabel.ORDERED_LIST)

    doc.add_list_item(
        parent=mylist_level_3,
        text="list item 3.c.i",
    )

    doc.add_list_item(
        parent=mylist_level_1,
        text="list item 4",
    )

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
    fig_item = doc.add_picture(caption=fig_caption_1)

    size = (64, 64)
    fig2_image = PILImage.new("RGB", size, "black")

    # Draw a red disk touching the borders
    # draw = ImageDraw.Draw(fig2_image)
    # draw.ellipse((0, 0, size[0] - 1, size[1] - 1), fill="red")

    # Create a drawing object
    ImageDraw.Draw(fig2_image)

    # Define the coordinates of the red square (x1, y1, x2, y2)
    square_size = 20  # Adjust as needed
    x1, y1 = 22, 22  # Adjust position
    x2, y2 = x1 + square_size, y1 + square_size

    # Draw the red square
    # draw.rectangle([x1, y1, x2, y2], fill="red")

    fig_caption_2 = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of figure 2."
    )
    fig2_item = doc.add_picture(
        image=ImageRef.from_pil(image=fig2_image, dpi=72), caption=fig_caption_2
    )

    g0 = doc.add_group(label=GroupLabel.LIST, parent=None)
    doc.add_list_item(text="item 1 of list", parent=g0)

    # an empty list
    doc.add_group(label=GroupLabel.LIST, parent=None)

    g1 = doc.add_group(label=GroupLabel.LIST, parent=None)
    doc.add_list_item(text="item 1 of list after empty list", parent=g1)
    doc.add_list_item(text="item 2 of list after empty list", parent=g1)

    g2 = doc.add_group(label=GroupLabel.LIST, parent=None)
    doc.add_list_item(text="item 1 of neighboring list", parent=g2)
    nli2 = doc.add_list_item(text="item 2 of neighboring list", parent=g2)

    g2_subgroup = doc.add_group(label=GroupLabel.LIST, parent=nli2)
    doc.add_list_item(text="item 1 of sub list", parent=g2_subgroup)

    inline1 = doc.add_group(label=GroupLabel.INLINE, parent=g2_subgroup)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here a code snippet:",
        parent=inline1,
    )
    doc.add_code(text="<p>Hello world</p>", parent=inline1)
    doc.add_text(
        label=DocItemLabel.TEXT, text="(to be displayed inline)", parent=inline1
    )

    inline2 = doc.add_group(label=GroupLabel.INLINE, parent=g2_subgroup)
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
    doc.add_code(text='print("Hello world")', parent=None)

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

    inline_fmt = doc.add_group(label=GroupLabel.INLINE)
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

    parent_A = doc.add_group(name="list A", label=GroupLabel.ORDERED_LIST)
    doc.add_list_item(text="Item 1 in A", enumerated=True, parent=parent_A)
    doc.add_list_item(text="Item 2 in A", enumerated=True, parent=parent_A)
    item_A_3 = doc.add_list_item(text="Item 3 in A", enumerated=True, parent=parent_A)

    parent_B = doc.add_group(
        parent=item_A_3, name="list B", label=GroupLabel.ORDERED_LIST
    )
    doc.add_list_item(text="Item 1 in B", enumerated=True, parent=parent_B)
    item_B_2 = doc.add_list_item(text="Item 2 in B", enumerated=True, parent=parent_B)

    parent_C = doc.add_group(
        parent=item_B_2, name="list C", label=GroupLabel.ORDERED_LIST
    )
    doc.add_list_item(text="Item 1 in C", enumerated=True, parent=parent_C)
    doc.add_list_item(text="Item 2 in C", enumerated=True, parent=parent_C)

    doc.add_list_item(text="Item 3 in B", enumerated=True, parent=parent_B)

    doc.add_list_item(text="Item 4 in A", enumerated=True, parent=parent_A)

    doc.add_text(label=DocItemLabel.TEXT, text="The end.", parent=None)

    return doc


def test_pil_image():
    doc = DoclingDocument(name="Untitled 1")

    fig_image = PILImage.new(mode="RGB", size=(2, 2), color=(0, 0, 0))
    fig_item = doc.add_picture(image=ImageRef.from_pil(image=fig_image, dpi=72))

    ### Serialize and deserialize the document
    yaml_dump = yaml.safe_dump(doc.model_dump(mode="json", by_alias=True))
    doc_reload = DoclingDocument.model_validate(yaml.safe_load(yaml_dump))
    reloaded_fig = doc_reload.pictures[0]
    reloaded_image = reloaded_fig.image.pil_image

    assert isinstance(reloaded_image, PILImage.Image)
    assert reloaded_image.size == fig_image.size
    assert reloaded_image.mode == fig_image.mode
    assert reloaded_image.tobytes() == fig_image.tobytes()


def test_image_ref():

    data_uri = {
        "dpi": 72,
        "mimetype": "image/png",
        "size": {"width": 10, "height": 11},
        "uri": "file:///tests/data/image.png",
    }
    image = ImageRef.model_validate(data_uri)
    assert isinstance(image.uri, AnyUrl)
    assert image.uri.scheme == "file"
    assert image.uri.path == "/tests/data/image.png"

    data_path = {
        "dpi": 72,
        "mimetype": "image/png",
        "size": {"width": 10, "height": 11},
        "uri": "./tests/data/image.png",
    }
    image = ImageRef.model_validate(data_path)
    assert isinstance(image.uri, Path)
    assert image.uri.name == "image.png"


def test_upgrade_content_layer_from_1_0_0():
    doc = DoclingDocument.load_from_json("test/data/doc/2206.01062-1.0.0.json")

    assert doc.version == CURRENT_VERSION
    assert doc.texts[0].content_layer == ContentLayer.FURNITURE


def test_version_doc():

    # default version
    doc = DoclingDocument(name="Untitled 1")
    assert doc.version == CURRENT_VERSION

    with open("test/data/doc/dummy_doc.yaml", encoding="utf-8") as fp:
        dict_from_yaml = yaml.safe_load(fp)
    doc = DoclingDocument.model_validate(dict_from_yaml)
    assert doc.version == CURRENT_VERSION

    # invalid version
    with pytest.raises(ValidationError, match="NoneType"):
        DoclingDocument(name="Untitled 1", version=None)
    with pytest.raises(ValidationError, match="pattern"):
        DoclingDocument(name="Untitled 1", version="abc")

    # incompatible version (major)
    major_split = CURRENT_VERSION.split(".", 1)
    new_version = f"{int(major_split[0]) + 1}.{major_split[1]}"
    with pytest.raises(ValidationError, match="incompatible"):
        DoclingDocument(name="Untitled 1", version=new_version)

    # incompatible version (minor)
    minor_split = major_split[1].split(".", 1)
    new_version = f"{major_split[0]}.{int(minor_split[0]) + 1}.{minor_split[1]}"
    with pytest.raises(ValidationError, match="incompatible"):
        DoclingDocument(name="Untitled 1", version=new_version)

    # compatible version (equal or lower minor)
    patch_split = minor_split[1].split(".", 1)
    comp_version = f"{major_split[0]}.{minor_split[0]}.{int(patch_split[0]) + 1}"
    doc = DoclingDocument(name="Untitled 1", version=comp_version)
    assert doc.version == CURRENT_VERSION


def test_formula_mathml():
    doc = DoclingDocument(name="Dummy")
    equation = "\\frac{1}{x}"
    doc.add_text(label=DocItemLabel.FORMULA, text=equation)

    doc_html = doc.export_to_html(formula_to_mathml=True, html_head="")

    gt_html = Path("test/data/docling_document/export/formula_mathml.html").read_text(
        encoding="utf8"
    )

    assert doc_html == gt_html


def test_formula_with_missing_fallback():
    doc = DoclingDocument(name="Dummy")
    bbox = BoundingBox.from_tuple((1, 2, 3, 4), origin=CoordOrigin.BOTTOMLEFT)
    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, 2))
    doc.add_text(label=DocItemLabel.FORMULA, text="", orig="(II.24) 2 Imar", prov=prov)

    actual = doc.export_to_html(
        formula_to_mathml=True, html_head="", image_mode=ImageRefMode.EMBEDDED
    )

    expected = """<!DOCTYPE html>
<html lang="en">

<div class="formula-not-decoded">Formula not decoded</div>
</html>"""

    assert actual == expected


def test_docitem_get_image():
    # Prepare the document
    doc = DoclingDocument(name="Dummy")

    page1_image = PILImage.new(mode="RGB", size=(200, 400), color=(0, 0, 0))
    doc_item_image = PILImage.new(mode="RGB", size=(20, 40), color=(255, 0, 0))
    page1_image.paste(doc_item_image, box=(20, 40))

    doc.add_page(  # With image
        page_no=1,
        size=Size(width=20, height=40),
        image=ImageRef.from_pil(page1_image, dpi=72),
    )
    doc.add_page(page_no=2, size=Size(width=20, height=40), image=None)  # Without image

    # DocItem with no provenance
    doc_item = DocItem(self_ref="#", label=DocItemLabel.TEXT, prov=[])
    assert doc_item.get_image(doc=doc) is None

    # DocItem on an invalid page
    doc_item = DocItem(
        self_ref="#",
        label=DocItemLabel.TEXT,
        prov=[ProvenanceItem(page_no=3, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
    )
    assert doc_item.get_image(doc=doc) is None

    # DocItem on a page without page image
    doc_item = DocItem(
        self_ref="#",
        label=DocItemLabel.TEXT,
        prov=[ProvenanceItem(page_no=2, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
    )
    assert doc_item.get_image(doc=doc) is None

    # DocItem on a page with valid page image
    doc_item = DocItem(
        self_ref="#",
        label=DocItemLabel.TEXT,
        prov=[
            ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=2, t=4, r=4, b=8), charspan=(1, 2)
            )
        ],
    )
    returned_doc_item_image = doc_item.get_image(doc=doc)
    assert (
        returned_doc_item_image is not None
        and returned_doc_item_image.tobytes() == doc_item_image.tobytes()
    )


def test_floatingitem_get_image():
    # Prepare the document
    doc = DoclingDocument(name="Dummy")

    page1_image = PILImage.new(mode="RGB", size=(200, 400), color=(0, 0, 0))
    floating_item_image = PILImage.new(mode="RGB", size=(20, 40), color=(255, 0, 0))
    page1_image.paste(floating_item_image, box=(20, 40))

    doc.add_page(  # With image
        page_no=1,
        size=Size(width=20, height=40),
        image=ImageRef.from_pil(page1_image, dpi=72),
    )
    doc.add_page(page_no=2, size=Size(width=20, height=40), image=None)  # Without image

    # FloatingItem with explicit image different from image based on provenance
    new_image = PILImage.new(mode="RGB", size=(40, 80), color=(0, 255, 0))
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[
            ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=2, t=4, r=6, b=12), charspan=(1, 2)
            )
        ],
        image=ImageRef.from_pil(image=new_image, dpi=72),
    )
    retured_image = floating_item.get_image(doc=doc)
    assert retured_image is not None and retured_image.tobytes() == new_image.tobytes()

    # FloatingItem without explicit image and no provenance
    floating_item = FloatingItem(
        self_ref="#", label=DocItemLabel.PICTURE, prov=[], image=None
    )
    assert floating_item.get_image(doc=doc) is None

    # FloatingItem without explicit image on invalid page
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[ProvenanceItem(page_no=3, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
        image=None,
    )
    assert floating_item.get_image(doc=doc) is None

    # FloatingItem without explicit image on a page without page image
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[ProvenanceItem(page_no=2, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
        image=None,
    )
    assert floating_item.get_image(doc=doc) is None

    # FloatingItem without explicit image on a page with page image
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[
            ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=2, t=4, r=4, b=8), charspan=(1, 2)
            )
        ],
        image=None,
    )
    retured_image = floating_item.get_image(doc=doc)
    assert (
        retured_image is not None
        and retured_image.tobytes() == floating_item_image.tobytes()
    )


def test_save_pictures():

    doc: DoclingDocument = _construct_doc()

    new_doc = doc._with_pictures_refs(image_dir=Path("./test/data/constructed_images/"))

    img_paths = new_doc._list_images_on_disk()
    assert len(img_paths) == 1, "len(img_paths)!=1"


def _normalise_string_wrt_filepaths(instr: str, paths: List[Path]):

    for p in paths:
        instr = instr.replace(str(p), str(p.name))

    return instr


def _verify_saved_output(filename: str, paths: List[Path]):

    pred = ""
    with open(filename, "r", encoding="utf-8") as fr:
        pred = fr.read()

    pred = _normalise_string_wrt_filepaths(pred, paths=paths)

    if GEN_TEST_DATA:
        with open(str(filename) + ".gt", "w", encoding="utf-8") as fw:
            fw.write(pred)
    else:
        gt = ""
        with open(str(filename) + ".gt", "r", encoding="utf-8") as fr:
            gt = fr.read()

        assert pred == gt, f"pred!=gt for {filename}"


def _verify_loaded_output(filename: str, pred=None):
    gt = DoclingDocument.load_from_json(Path(str(filename) + ".gt"))

    pred = pred or DoclingDocument.load_from_json(Path(filename))
    assert isinstance(pred, DoclingDocument)

    assert (
        pred.export_to_dict() == gt.export_to_dict()
    ), f"pred.export_to_dict() != gt.export_to_dict() for {filename}"
    assert pred == gt, f"pred!=gt for {filename}"


def test_save_to_disk():

    doc: DoclingDocument = _construct_doc()

    image_dir = Path("./test/data/doc/constructed_images/")

    doc_with_references = doc._with_pictures_refs(
        image_dir=image_dir  # Path("./test/data/constructed_images/")
    )

    # paths will be different on different machines, so needs to be kept!
    paths = doc_with_references._list_images_on_disk()
    assert len(paths) == 1, "len(paths)!=1"

    ### MarkDown

    filename = Path("test/data/doc/constructed_doc.placeholder.md")
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.PLACEHOLDER
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.embedded.md")
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.md")
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### HTML

    filename = Path("test/data/doc/constructed_doc.placeholder.html")
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.PLACEHOLDER
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.embedded.html")
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.html")
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### Document Tokens

    filename = Path("test/data/doc/constructed_doc.dt")
    doc.save_as_doctags(filename=filename)
    _verify_saved_output(filename=filename, paths=paths)

    ### JSON

    filename = Path("test/data/doc/constructed_doc.embedded.json")
    doc.save_as_json(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    doc_emb_loaded = DoclingDocument.load_from_json(filename)
    _verify_loaded_output(filename=filename, pred=doc_emb_loaded)

    filename = Path("test/data/doc/constructed_doc.referenced.json")
    doc.save_as_json(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    doc_ref_loaded = DoclingDocument.load_from_json(filename)
    _verify_loaded_output(filename=filename, pred=doc_ref_loaded)

    ### YAML

    filename = Path("test/data/doc/constructed_doc.embedded.yaml")
    doc.save_as_yaml(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.yaml")
    doc.save_as_yaml(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    assert True
