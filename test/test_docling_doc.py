import os
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union
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
    NodeItem,
    PictureItem,
    ProvenanceItem,
    RefItem,
    RichTableCell,
    Script,
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


def test_x_overlap_with():
    bbox1 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2 = BoundingBox(l=5, t=0, r=15, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_overlap_with(bbox2) - 5.0) < 1.0e-3

    # No overlap (disjoint right)
    bbox3 = BoundingBox(l=11, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_overlap_with(bbox3) - 0.0) < 1.0e-3

    # No overlap (disjoint left)
    bbox4 = BoundingBox(l=-10, t=0, r=-1, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_overlap_with(bbox4) - 0.0) < 1.0e-3

    # Touching edges
    bbox5 = BoundingBox(l=10, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_overlap_with(bbox5) - 0.0) < 1.0e-3

    # Full containment
    bbox6 = BoundingBox(l=2, t=0, r=8, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_overlap_with(bbox6) - 6.0) < 1.0e-3

    # Identical boxes
    bbox7 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_overlap_with(bbox7) - 10.0) < 1.0e-3

    # Different CoordOrigin
    bbox_bl = BoundingBox(l=0, t=10, r=10, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
    with pytest.raises(ValueError):
        bbox1.x_overlap_with(bbox_bl)


def test_y_overlap_with():
    # TOPLEFT origin
    bbox1_tl = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2_tl = BoundingBox(l=0, t=5, r=10, b=15, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1_tl.y_overlap_with(bbox2_tl) - 5.0) < 1.0e-3

    # No overlap (disjoint below)
    bbox3_tl = BoundingBox(l=0, t=11, r=10, b=20, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1_tl.y_overlap_with(bbox3_tl) - 0.0) < 1.0e-3

    # Touching edges
    bbox4_tl = BoundingBox(l=0, t=10, r=10, b=20, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1_tl.y_overlap_with(bbox4_tl) - 0.0) < 1.0e-3

    # Full containment
    bbox5_tl = BoundingBox(l=0, t=2, r=10, b=8, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1_tl.y_overlap_with(bbox5_tl) - 6.0) < 1.0e-3

    # BOTTOMLEFT origin
    bbox1_bl = BoundingBox(l=0, b=0, r=10, t=10, coord_origin=CoordOrigin.BOTTOMLEFT)
    bbox2_bl = BoundingBox(l=0, b=5, r=10, t=15, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert abs(bbox1_bl.y_overlap_with(bbox2_bl) - 5.0) < 1.0e-3

    # No overlap (disjoint above)
    bbox3_bl = BoundingBox(l=0, b=11, r=10, t=20, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert abs(bbox1_bl.y_overlap_with(bbox3_bl) - 0.0) < 1.0e-3

    # Touching edges
    bbox4_bl = BoundingBox(l=0, b=10, r=10, t=20, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert abs(bbox1_bl.y_overlap_with(bbox4_bl) - 0.0) < 1.0e-3

    # Full containment
    bbox5_bl = BoundingBox(l=0, b=2, r=10, t=8, coord_origin=CoordOrigin.BOTTOMLEFT)
    assert abs(bbox1_bl.y_overlap_with(bbox5_bl) - 6.0) < 1.0e-3

    # Different CoordOrigin
    with pytest.raises(ValueError):
        bbox1_tl.y_overlap_with(bbox1_bl)


def test_union_area_with():
    # Overlapping (TOPLEFT)
    bbox1 = BoundingBox(
        l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT
    )  # Area 100
    bbox2 = BoundingBox(
        l=5, t=5, r=15, b=15, coord_origin=CoordOrigin.TOPLEFT
    )  # Area 100
    # Intersection area 25
    # Union area = 100 + 100 - 25 = 175
    assert abs(bbox1.union_area_with(bbox2) - 175.0) < 1.0e-3

    # Non-overlapping (TOPLEFT)
    bbox3 = BoundingBox(
        l=20, t=0, r=30, b=10, coord_origin=CoordOrigin.TOPLEFT
    )  # Area 100
    # Union area = 100 + 100 - 0 = 200
    assert abs(bbox1.union_area_with(bbox3) - 200.0) < 1.0e-3

    # Touching edges (TOPLEFT)
    bbox4 = BoundingBox(
        l=10, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT
    )  # Area 100
    # Union area = 100 + 100 - 0 = 200
    assert abs(bbox1.union_area_with(bbox4) - 200.0) < 1.0e-3

    # Full containment (TOPLEFT)
    bbox5 = BoundingBox(l=2, t=2, r=8, b=8, coord_origin=CoordOrigin.TOPLEFT)  # Area 36
    # Union area = 100 + 36 - 36 = 100
    assert abs(bbox1.union_area_with(bbox5) - 100.0) < 1.0e-3

    # Overlapping (BOTTOMLEFT)
    bbox6 = BoundingBox(
        l=0, b=0, r=10, t=10, coord_origin=CoordOrigin.BOTTOMLEFT
    )  # Area 100
    bbox7 = BoundingBox(
        l=5, b=5, r=15, t=15, coord_origin=CoordOrigin.BOTTOMLEFT
    )  # Area 100
    # Intersection area 25
    # Union area = 100 + 100 - 25 = 175
    assert abs(bbox6.union_area_with(bbox7) - 175.0) < 1.0e-3

    # Different CoordOrigin
    with pytest.raises(ValueError):
        bbox1.union_area_with(bbox6)


def test_x_union_with():
    bbox1 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2 = BoundingBox(l=5, t=0, r=15, b=10, coord_origin=CoordOrigin.TOPLEFT)
    # x_union = max(10, 15) - min(0, 5) = 15 - 0 = 15
    assert abs(bbox1.x_union_with(bbox2) - 15.0) < 1.0e-3

    # No overlap (disjoint)
    bbox3 = BoundingBox(l=20, t=0, r=30, b=10, coord_origin=CoordOrigin.TOPLEFT)
    # x_union = max(10, 30) - min(0, 20) = 30 - 0 = 30
    assert abs(bbox1.x_union_with(bbox3) - 30.0) < 1.0e-3

    # Touching edges
    bbox4 = BoundingBox(l=10, t=0, r=20, b=10, coord_origin=CoordOrigin.TOPLEFT)
    # x_union = max(10, 20) - min(0, 10) = 20 - 0 = 20
    assert abs(bbox1.x_union_with(bbox4) - 20.0) < 1.0e-3

    # Full containment
    bbox5 = BoundingBox(l=2, t=0, r=8, b=10, coord_origin=CoordOrigin.TOPLEFT)
    # x_union = max(10, 8) - min(0, 2) = 10 - 0 = 10
    assert abs(bbox1.x_union_with(bbox5) - 10.0) < 1.0e-3

    # Identical boxes
    bbox6 = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert abs(bbox1.x_union_with(bbox6) - 10.0) < 1.0e-3

    # Different CoordOrigin
    bbox_bl = BoundingBox(l=0, t=10, r=10, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
    with pytest.raises(ValueError):
        bbox1.x_union_with(bbox_bl)


def test_y_union_with():

    bbox1_tl = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    bbox2_tl = BoundingBox(l=0, t=5, r=10, b=15, coord_origin=CoordOrigin.TOPLEFT)
    # y_union = max(10, 15) - min(0, 5) = 15 - 0 = 15
    assert abs(bbox1_tl.y_union_with(bbox2_tl) - 15.0) < 1.0e-3

    # No overlap (disjoint below)
    bbox3_tl = BoundingBox(l=0, t=20, r=10, b=30, coord_origin=CoordOrigin.TOPLEFT)
    # y_union = max(10, 30) - min(0, 20) = 30 - 0 = 30
    assert abs(bbox1_tl.y_union_with(bbox3_tl) - 30.0) < 1.0e-3

    # Touching edges
    bbox4_tl = BoundingBox(l=0, t=10, r=10, b=20, coord_origin=CoordOrigin.TOPLEFT)
    # y_union = max(10, 20) - min(0, 10) = 20 - 0 = 20
    assert abs(bbox1_tl.y_union_with(bbox4_tl) - 20.0) < 1.0e-3

    # Full containment
    bbox5_tl = BoundingBox(l=0, t=2, r=10, b=8, coord_origin=CoordOrigin.TOPLEFT)
    # y_union = max(10, 8) - min(0, 2) = 10 - 0 = 10
    assert abs(bbox1_tl.y_union_with(bbox5_tl) - 10.0) < 1.0e-3

    # BOTTOMLEFT origin
    bbox1_bl = BoundingBox(l=0, b=0, r=10, t=10, coord_origin=CoordOrigin.BOTTOMLEFT)
    bbox2_bl = BoundingBox(l=0, b=5, r=10, t=15, coord_origin=CoordOrigin.BOTTOMLEFT)
    # y_union = max(10, 15) - min(0, 5) = 15 - 0 = 15
    assert abs(bbox1_bl.y_union_with(bbox2_bl) - 15.0) < 1.0e-3

    # No overlap (disjoint above)
    bbox3_bl = BoundingBox(l=0, b=20, r=10, t=30, coord_origin=CoordOrigin.BOTTOMLEFT)
    # y_union = max(10, 30) - min(0, 20) = 30 - 0 = 30
    assert abs(bbox1_bl.y_union_with(bbox3_bl) - 30.0) < 1.0e-3

    # Touching edges
    bbox4_bl = BoundingBox(l=0, b=10, r=10, t=20, coord_origin=CoordOrigin.BOTTOMLEFT)
    # y_union = max(10, 20) - min(0, 10) = 20 - 0 = 20
    assert abs(bbox1_bl.y_union_with(bbox4_bl) - 20.0) < 1.0e-3

    # Full containment
    bbox5_bl = BoundingBox(l=0, b=2, r=10, t=8, coord_origin=CoordOrigin.BOTTOMLEFT)
    # y_union = max(10, 8) - min(0, 2) = 10 - 0 = 10
    assert abs(bbox1_bl.y_union_with(bbox5_bl) - 10.0) < 1.0e-3

    # Different CoordOrigin
    with pytest.raises(ValueError):
        bbox1_tl.y_union_with(bbox1_bl)


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

    # check that deprecation warning for furniture has been raised.
    with pytest.warns(DeprecationWarning, match="deprecated"):
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

        assert (
            gt_true == pred
        ), f"Does not pass regression-test for {filename}.{ext}\n\n{gt_true}\n\n{pred}"
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

    # Test pages parameter in DocTags export
    if doc.pages:  # Only test if document has pages
        first_page = min(doc.pages.keys())
        second_page = first_page + 1
        if second_page in doc.pages:  # Only test if document has at least 2 pages
            dt_pages_pred = doc.export_to_doctags(pages={first_page, second_page})
            print(dt_pages_pred)
            _verify_regression_test(dt_pages_pred, filename=filename, ext="pages.dt")

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
    doc.add_code(text='print("Hello world")', parent=inline1)
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

    file = "test/data/docling_document/export/formula_mathml.html"
    if GEN_TEST_DATA:
        with open(file, mode="w", encoding="utf8") as f:
            f.write(f"{doc_html}\n")
    else:
        with open(file, mode="r", encoding="utf8") as f:
            gt_html = f.read().rstrip()
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

    assert '<div class="formula-not-decoded">Formula not decoded</div>' in expected


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

    new_doc = doc._with_pictures_refs(
        image_dir=Path("./test/data/constructed_images/"), page_no=None
    )

    img_paths = new_doc._list_images_on_disk()
    assert len(img_paths) == 1, "len(img_paths)!=1"


def test_save_pictures_with_page():
    # Given
    doc = DoclingDocument(name="Dummy")

    doc.add_page(page_no=1, size=Size(width=2000, height=4000), image=None)
    doc.add_page(
        page_no=2,
        size=Size(width=2000, height=4000),
    )
    image = PILImage.new(mode="RGB", size=(200, 400), color=(0, 0, 0))
    doc.add_picture(
        image=ImageRef.from_pil(image=image, dpi=72),
        prov=ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(
                b=0, l=0, r=200, t=400, coord_origin=CoordOrigin.BOTTOMLEFT
            ),
            charspan=(1, 2),
        ),
    )

    # When
    with_ref = doc._with_pictures_refs(
        image_dir=Path("./test/data/constructed_images/"), page_no=1
    )
    # Then
    n_images = len(with_ref._list_images_on_disk())
    assert n_images == 0
    # When
    with_ref = with_ref._with_pictures_refs(
        image_dir=Path("./test/data/constructed_images/"), page_no=2
    )
    n_images = len(with_ref._list_images_on_disk())
    # Then
    assert n_images == 1


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


def _gt_filename(filename: Path) -> Path:
    return Path(str(filename) + ".gt")


def _verify_loaded_output(filename: Path, pred=None):
    # gt = DoclingDocument.load_from_json(Path(str(filename) + ".gt"))
    gt = DoclingDocument.load_from_json(_gt_filename(filename=filename))

    pred = pred or DoclingDocument.load_from_json(Path(filename))
    assert isinstance(pred, DoclingDocument)

    assert (
        pred.export_to_dict() == gt.export_to_dict()
    ), f"pred.export_to_dict() != gt.export_to_dict() for {filename}"
    assert pred == gt, f"pred!=gt for {filename}"


def test_save_to_disk():

    doc: DoclingDocument = _construct_doc()

    test_dir = Path("./test/data/doc")
    image_dir = Path("constructed_images/")  # will be relative to test_dir

    doc_with_references = doc._with_pictures_refs(
        image_dir=(test_dir / image_dir),
        page_no=None,
    )

    # paths will be different on different machines, so needs to be kept!
    paths = doc_with_references._list_images_on_disk()
    assert len(paths) == 1, "len(paths)!=1"

    ### MarkDown

    filename = test_dir / "constructed_doc.placeholder.md"
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.PLACEHOLDER
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = test_dir / "constructed_doc.embedded.md"
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = test_dir / "constructed_doc.referenced.md"
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### HTML

    filename = test_dir / "constructed_doc.placeholder.html"
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.PLACEHOLDER
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = test_dir / "constructed_doc.embedded.html"
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = test_dir / "constructed_doc.referenced.html"
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### Document Tokens

    filename = test_dir / "constructed_doc.dt"
    doc.save_as_doctags(filename=filename)
    _verify_saved_output(filename=filename, paths=paths)

    ### JSON

    filename = test_dir / "constructed_doc.embedded.json"
    doc.save_as_json(
        filename=filename,
        artifacts_dir=image_dir,
        image_mode=ImageRefMode.EMBEDDED,
    )
    _verify_saved_output(filename=filename, paths=paths)

    doc_emb_loaded = DoclingDocument.load_from_json(filename)
    _verify_loaded_output(filename=filename, pred=doc_emb_loaded)

    filename = test_dir / "constructed_doc.referenced.json"
    doc.save_as_json(
        filename=filename,
        artifacts_dir=image_dir,
        image_mode=ImageRefMode.REFERENCED,
    )
    _verify_saved_output(filename=filename, paths=paths)

    doc_ref_loaded = DoclingDocument.load_from_json(filename)
    _verify_loaded_output(filename=filename, pred=doc_ref_loaded)

    ### YAML

    filename = test_dir / "constructed_doc.embedded.yaml"
    doc.save_as_yaml(
        filename=filename,
        artifacts_dir=image_dir,
        image_mode=ImageRefMode.EMBEDDED,
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = test_dir / "constructed_doc.referenced.yaml"
    doc.save_as_yaml(
        filename=filename,
        artifacts_dir=image_dir,
        image_mode=ImageRefMode.REFERENCED,
    )
    _verify_saved_output(filename=filename, paths=paths)

    assert True


def test_document_stack_operations():

    doc: DoclingDocument = _construct_doc()

    # _print(document=doc)

    ref = RefItem(cref="#/texts/12")
    success, stack = doc._get_stack_of_refitem(ref=ref)

    assert success
    assert stack == [
        2,
        2,
        2,
        0,
        2,
        0,
        0,
    ], f"stack==[2, 2, 2, 0, 2, 0, 0] for stack: {stack}"


def test_document_manipulation():

    def _resolve(doc: DoclingDocument, cref: str) -> NodeItem:
        ref = RefItem(cref=cref)
        return ref.resolve(doc=doc)

    def _verify(
        filename: Path,
        document: DoclingDocument,
        generate: bool = False,
    ):
        if generate or (not os.path.exists(_gt_filename(filename=filename))):
            doc.save_as_json(
                filename=_gt_filename(filename=filename),
                artifacts_dir=image_dir,
                image_mode=ImageRefMode.EMBEDDED,
            )
        # test if the document is still model-validating
        DoclingDocument.load_from_json(filename=_gt_filename(filename=filename))

        # test if the document is the same as the stored GT
        _verify_loaded_output(
            filename=filename, pred=DoclingDocument.model_validate(document)
        )

    image_dir = Path("./test/data/doc/constructed_images/")

    doc: DoclingDocument = _construct_doc()

    text_item_1 = ListItem(
        self_ref="#",
        text="new list item (before)",
        orig="new list item (before)",
    )
    text_item_2 = ListItem(
        self_ref="#",
        text="new list item (after)",
        orig="new list item (after)",
    )

    node = _resolve(doc=doc, cref="#/texts/10")

    doc.insert_item_before_sibling(new_item=text_item_1, sibling=node)
    doc.insert_item_after_sibling(new_item=text_item_2, sibling=node)

    filename = Path("test/data/doc/constructed_doc.inserted_text.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    items = [_resolve(doc=doc, cref="#/texts/10")]
    doc.delete_items(node_items=items)

    filename = Path("test/data/doc/constructed_doc.deleted_text.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    items = [_resolve(doc=doc, cref="#/groups/1")]
    doc.delete_items(node_items=items)

    filename = Path("test/data/doc/constructed_doc.deleted_group.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    items = [_resolve(doc=doc, cref="#/pictures/1")]
    doc.delete_items(node_items=items)

    filename = Path("test/data/doc/constructed_doc.deleted_picture.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    text_item_3 = TextItem(
        self_ref="#",
        text="child text appended at body",
        orig="child text appended at body",
        label=DocItemLabel.TEXT,
    )
    doc.append_child_item(child=text_item_3)

    text_item_4 = ListItem(
        self_ref="#",
        text="child text appended at body",
        orig="child text appended at body",
        label=DocItemLabel.LIST_ITEM,
    )
    parent = _resolve(doc=doc, cref="#/groups/11")
    doc.append_child_item(child=text_item_4, parent=parent)

    # try to add a sibling to the root:
    with pytest.raises(ValueError):
        doc.insert_item_before_sibling(
            new_item=TextItem(
                self_ref="#",
                label=DocItemLabel.TEXT,
                text="foo",
                orig="foo",
            ),
            sibling=doc.body,
        )

    # try to append a child with children of its own:
    with pytest.raises(ValueError):
        doc.append_child_item(
            child=TextItem(
                self_ref="#",
                label=DocItemLabel.TEXT,
                text="foo",
                orig="foo",
                children=[
                    _resolve(doc=deepcopy(doc), cref=text_item_4.self_ref).get_ref()
                ],
            ),
            parent=doc.body,
        )

    filename = Path("test/data/doc/constructed_doc.appended_child.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    text_item_5 = TextItem(
        self_ref="#",
        text="new child",
        orig="new child",
        label=DocItemLabel.TEXT,
    )
    doc.replace_item(old_item=text_item_3, new_item=text_item_5)

    filename = Path("test/data/doc/constructed_doc.replaced_item.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    # Test insert_* methods with mixed insertion before/after sibling

    node = _resolve(doc=doc, cref="#/texts/45")

    last_node = doc.insert_list_group(
        sibling=node, name="Inserted List Group", after=True
    )
    group_node = doc.insert_inline_group(
        sibling=node, name="Inserted Inline Group", after=False
    )
    doc.insert_group(
        sibling=node,
        label=GroupLabel.LIST,
        name="Inserted Group w/ LIST Label",
        after=True,
    )
    doc.insert_group(
        sibling=node,
        label=GroupLabel.ORDERED_LIST,
        name="Inserted Group w/ ORDERED_LIST Label",
        after=False,
    )
    doc.insert_group(
        sibling=node,
        label=GroupLabel.INLINE,
        name="Inserted Group w/ INLINE Label",
        after=True,
    )
    doc.insert_group(
        sibling=node,
        label=GroupLabel.UNSPECIFIED,
        name="Inserted Group w/ UNSPECIFIED Label",
        after=False,
    )
    doc.insert_text(
        sibling=node,
        label=DocItemLabel.TITLE,
        text="Inserted Text w/ TITLE Label",
        after=True,
    )
    doc.insert_text(
        sibling=node,
        label=DocItemLabel.SECTION_HEADER,
        text="Inserted Text w/ SECTION_HEADER Label",
        after=False,
    )
    doc.insert_text(
        sibling=node,
        label=DocItemLabel.CODE,
        text="Inserted Text w/ CODE Label",
        after=True,
    )
    doc.insert_text(
        sibling=node,
        label=DocItemLabel.FORMULA,
        text="Inserted Text w/ FORMULA Label",
        after=False,
    )
    doc.insert_text(
        sibling=node,
        label=DocItemLabel.TEXT,
        text="Inserted Text w/ TEXT Label",
        after=True,
    )

    num_rows = 3
    num_cols = 3
    table_cells = []

    for i in range(num_rows):
        for j in range(num_cols):
            table_cells.append(
                TableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    text=str(i * num_rows + j),
                )
            )

    table_data = TableData(
        table_cells=table_cells, num_rows=num_rows, num_cols=num_cols
    )
    doc.insert_table(sibling=node, data=table_data, after=False)

    size = (64, 64)
    img = PILImage.new("RGB", size, "black")
    doc.insert_picture(
        sibling=node, image=ImageRef.from_pil(image=img, dpi=72), after=True
    )

    doc.insert_title(sibling=node, text="Inserted Title", after=False)
    doc.insert_code(sibling=node, text="Inserted Code", after=True)
    doc.insert_formula(sibling=node, text="Inserted Formula", after=False)
    doc.insert_heading(sibling=node, text="Inserted Heading", after=True)

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

    doc.insert_key_values(sibling=node, graph=graph, after=False)
    doc.insert_form(sibling=node, graph=graph, after=True)

    filename = Path("test/data/doc/constructed_doc.inserted_items_with_insert_*.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    # Test the handling of list items in insert_* methods, both with and without parent groups

    with pytest.warns(DeprecationWarning, match="ListItem parent must be a ListGroup"):
        li_sibling = doc.insert_list_item(
            sibling=node, text="Inserted List Item, Incorrect Parent", after=False
        )
    doc.insert_list_item(
        sibling=li_sibling, text="Inserted List Item, Correct Parent", after=True
    )
    doc.insert_text(
        sibling=li_sibling,
        label=DocItemLabel.LIST_ITEM,
        text="Inserted Text with LIST_ITEM Label, Correct Parent",
        after=False,
    )
    with pytest.warns(DeprecationWarning, match="ListItem parent must be a ListGroup"):
        doc.insert_text(
            sibling=node,
            label=DocItemLabel.LIST_ITEM,
            text="Inserted Text with LIST_ITEM Label, Incorrect Parent",
            after=True,
        )

    filename = Path(
        "test/data/doc/constructed_doc.inserted_list_items_with_insert_*.json"
    )
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    # Test the bulk addition of node items

    text_item_6 = TextItem(
        self_ref="#",
        text="Bulk Addition 1",
        orig="Bulk Addition 1",
        label=DocItemLabel.TEXT,
    )
    text_item_7 = TextItem(
        self_ref="#",
        text="Bulk Addition 2",
        orig="Bulk Addition 2",
        label=DocItemLabel.TEXT,
    )

    doc.add_node_items(
        node_items=[text_item_6, text_item_7], doc=doc, parent=group_node
    )

    filename = Path("test/data/doc/constructed_doc.bulk_item_addition.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    # Test the bulk insertion of node items

    text_item_8 = TextItem(
        self_ref="#",
        text="Bulk Insertion 1",
        orig="Bulk Insertion 1",
        label=DocItemLabel.TEXT,
    )
    text_item_9 = TextItem(
        self_ref="#",
        text="Bulk Insertion 2",
        orig="Bulk Insertion 2",
        label=DocItemLabel.TEXT,
    )

    doc.insert_node_items(
        sibling=node, node_items=[text_item_8, text_item_9], doc=doc, after=False
    )

    filename = Path("test/data/doc/constructed_doc.bulk_item_insertion.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    # Test table data manipulation methods

    table_data.add_row(["*"] * 3)
    table_data.add_rows([["a", "b", "c"], ["d", "e", "f"]])
    table_data.insert_row(1, ["*"] * 3)
    table_data.insert_rows(1, [["a", "b", "c"], ["d", "e", "f"]], after=True)
    table_data.pop_row()
    table_data.remove_row(3)
    table_data.remove_rows([1, 2, 5])

    # Try to remove a nonexistent row
    with pytest.raises(IndexError):
        table_data.remove_row(100)

    filename = Path("test/data/doc/constructed_doc.manipulated_table.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    # Test range manipulation methods

    # Try to extract a range where start is after end
    with pytest.raises(ValueError):
        extracted_doc = doc.extract_items_range(start=node, end=group_node)

    # Try to extract a range where start and end have different parents
    with pytest.raises(ValueError):
        extracted_doc = doc.extract_items_range(start=li_sibling, end=node)

    extracted_doc = doc.extract_items_range(
        start=group_node, end=node, end_inclusive=False, delete=True
    )

    filename = Path("test/data/doc/constructed_doc.extracted_with_deletion.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    doc.add_document(doc=extracted_doc, parent=last_node)

    filename = Path("test/data/doc/constructed_doc.added_extracted_doc.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    doc.insert_document(doc=extracted_doc, sibling=last_node, after=False)

    filename = Path("test/data/doc/constructed_doc.inserted_extracted_doc.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)

    doc.delete_items_range(start=node, end=last_node, start_inclusive=False)

    filename = Path("test/data/doc/constructed_doc.deleted_items_range.json")
    _verify(filename=filename, document=doc, generate=GEN_TEST_DATA)


def test_misplaced_list_items():
    filename = Path("test/data/doc/misplaced_list_items.yaml")
    doc = DoclingDocument.load_from_yaml(filename)

    dt_pred = doc.export_to_doctags()
    _verify_regression_test(dt_pred, filename=str(filename), ext="dt")

    exp_file = filename.parent / f"{filename.stem}.out.yaml"
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)
    else:
        exp_doc = DoclingDocument.load_from_yaml(exp_file)
        assert doc == exp_doc

    doc._normalize_references()
    exp_file = filename.parent / f"{filename.stem}.norm.out.yaml"
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)
    else:
        exp_doc = DoclingDocument.load_from_yaml(exp_file)
        assert doc == exp_doc


def test_export_with_precision():
    doc = DoclingDocument.load_from_yaml(filename="test/data/doc/dummy_doc_2.yaml")
    act_data = doc.export_to_dict(coord_precision=2, confid_precision=1)
    exp_file = Path("test/data/doc/dummy_doc_2_prec.yaml")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            yaml.dump(act_data, f, default_flow_style=False)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            exp_data = yaml.load(f, Loader=yaml.FullLoader)
        assert act_data == exp_data


def test_concatenate():
    files = [
        "test/data/doc/2501.17887v1.json",
        "test/data/doc/constructed_doc.embedded.json",
        "test/data/doc/2311.18481v1.json",
    ]
    docs = [DoclingDocument.load_from_json(filename=f) for f in files]
    doc = DoclingDocument.concatenate(docs=docs)

    html_data = doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED, split_page_view=True
    )

    exp_json_file = Path("test/data/doc/concatenated.json")
    exp_html_file = exp_json_file.with_suffix(".html")

    if GEN_TEST_DATA:
        doc.save_as_json(exp_json_file)
        with open(exp_html_file, "w", encoding="utf-8") as f:
            f.write(html_data)
    else:
        exp_doc = DoclingDocument.load_from_json(exp_json_file)
        assert doc == exp_doc

        with open(exp_html_file, "r", encoding="utf-8") as f:
            exp_html_data = f.read()
        assert html_data == exp_html_data


def test_list_group_with_list_items():
    good_doc = DoclingDocument(name="")
    l1 = good_doc.add_list_group()
    good_doc.add_list_item(text="ListItem 1", parent=l1)
    good_doc.add_list_item(text="ListItem 2", parent=l1)

    good_doc._validate_rules()


def test_list_group_with_non_list_items():
    bad_doc = DoclingDocument(name="")
    l1 = bad_doc.add_list_group()
    bad_doc.add_list_item(text="ListItem 1", parent=l1)
    bad_doc.add_text(
        text="non-ListItem in ListGroup", label=DocItemLabel.TEXT, parent=l1
    )

    with pytest.raises(ValueError):
        bad_doc._validate_rules()


def test_list_item_outside_list_group():
    def unsafe_add_list_item(
        doc: DoclingDocument,
        text: str,
        enumerated: bool = False,
        marker: Optional[str] = None,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        if not parent:
            parent = doc.body

        if not orig:
            orig = text

        text_index = len(doc.texts)
        cref = f"#/texts/{text_index}"
        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            enumerated=enumerated,
            marker=marker or "",
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            list_item.prov.append(prov)
        if content_layer:
            list_item.content_layer = content_layer

        doc.texts.append(list_item)
        parent.children.append(RefItem(cref=cref))

        return list_item

    bad_doc = DoclingDocument(name="")
    unsafe_add_list_item(doc=bad_doc, text="ListItem outside ListGroup")
    with pytest.raises(ValueError):
        bad_doc._validate_rules()


def test_list_item_inside_list_group():
    doc = DoclingDocument(name="")
    l1 = doc.add_list_group()
    doc.add_list_item(text="ListItem inside ListGroup", parent=l1)
    doc._validate_rules()


def test_group_with_children():
    good_doc = DoclingDocument(name="")
    grp = good_doc.add_group()
    good_doc.add_text(text="Text in group", label=DocItemLabel.TEXT, parent=grp)
    good_doc._validate_rules()


def test_group_without_children():
    bad_doc = DoclingDocument(name="")
    bad_doc.add_group()
    with pytest.raises(ValueError):
        bad_doc._validate_rules()


def _construct_rich_table_doc():

    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TITLE, text="Rich tables")

    table_item = doc.add_table(
        data=TableData(
            num_rows=4,
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
                )
            elif i == 2 and j == 0:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item_2.get_ref(),
                )
            elif i == 3 and j == 1:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item_3.get_ref(),
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


def test_rich_tables():
    doc = _construct_rich_table_doc()

    exp_file = Path("test/data/doc/rich_table.out.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)

    exp_doc = DoclingDocument.load_from_yaml(exp_file)
    assert doc == exp_doc


def test_doc_manipulation_with_rich_tables():
    doc = _construct_rich_table_doc()

    doc.delete_items(node_items=[doc.texts[0]])

    exp_file = Path("test/data/doc/rich_table_post_text_del.out.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)

    exp_doc = DoclingDocument.load_from_yaml(exp_file)
    assert doc == exp_doc


def test_invalid_rich_table_doc():
    doc = DoclingDocument(name="")
    table_item = doc.add_table(data=TableData(num_rows=2, num_cols=2))
    rich_item = doc.add_text(
        text="rich item",
        label=DocItemLabel.TEXT,
        parent=doc.body,  # not the table item
    )
    for i in range(table_item.data.num_rows):
        for j in range(table_item.data.num_cols):
            if i == 1 and j == 1:
                table_cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item.get_ref(),
                )

                # ensure add_table_cell() raises:
                with pytest.raises(ValueError):
                    doc.add_table_cell(table_item=table_item, cell=table_cell)
            else:
                table_cell = TableCell(
                    text=f"cell {i},{j}",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )

            # discouraged but technically possible:
            table_item.data.table_cells.append(table_cell)

    # ensure validate_document() raises:
    with pytest.raises(ValueError):
        DoclingDocument.validate_document(doc)


def test_rich_table_item_insertion_normalization():

    doc = DoclingDocument(name="")
    doc.add_text(label=DocItemLabel.TITLE, text="Rich tables")

    table_item = doc.add_table(
        data=TableData(
            num_rows=4,
            num_cols=2,
        ),
    )

    rich_item = doc.add_text(
        parent=table_item,
        text="text in italic",
        label=DocItemLabel.TEXT,
        formatting=Formatting(italic=True),
    )

    for i in range(table_item.data.num_rows):
        for j in range(table_item.data.num_cols):
            if i == 1 and j == 1:
                cell = RichTableCell(
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    ref=rich_item.get_ref(),
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

    # state before insert:
    exp_file = Path("test/data/doc/rich_table_item_ins_norm_1.out.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)
    exp_doc = DoclingDocument.load_from_yaml(exp_file)
    assert doc == exp_doc

    doc.insert_item_before_sibling(
        new_item=TextItem(
            self_ref="#",
            text="text before",
            orig="text before",
            label=DocItemLabel.TEXT,
        ),
        sibling=table_item,
    )

    # state after insert (prior to normalization):
    exp_file = Path("test/data/doc/rich_table_item_ins_norm_2.out.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)
    exp_doc = DoclingDocument.load_from_yaml(exp_file)
    assert doc == exp_doc

    doc._normalize_references()

    # state after insert (after normalization):
    exp_file = Path("test/data/doc/rich_table_item_ins_norm_3.out.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(exp_file)
    exp_doc = DoclingDocument.load_from_yaml(exp_file)
    assert doc == exp_doc


def test_filter_pages():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    orig_doc = DoclingDocument.load_from_json(src)
    doc = orig_doc.filter(page_nrs={2, 3, 5})

    html_data = doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED, split_page_view=True
    )

    exp_json_file = src.with_name(f"{src.stem}_p2_p3_p5.gt.json")
    exp_html_file = exp_json_file.with_suffix(".html")

    if GEN_TEST_DATA:
        doc.save_as_json(exp_json_file)
        with open(exp_html_file, "w", encoding="utf-8") as f:
            f.write(html_data)
    else:
        exp_doc = DoclingDocument.load_from_json(exp_json_file)
        assert doc == exp_doc

        with open(exp_html_file, "r", encoding="utf-8") as f:
            exp_html_data = f.read()
        assert html_data == exp_html_data
