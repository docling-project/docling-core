import math
from pathlib import Path

from docling_core.types.doc.base import (
    BoundingBox,
    BoundingRectangle,
    CoordOrigin,
    Size,
)
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
    Script,
    SectionHeaderItem,
    Size,
    TableCell,
    TableData,
    TableItem,
    TextItem,
    TitleItem,
)

from .test_data_gen_flag import GEN_TEST_DATA


class TestBoundingRectangle:
    """Test suite for BoundingRectangle class."""

    def test_init(self):
        """Test BoundingRectangle initialization."""
        rect = BoundingRectangle(
            r_x0=0,
            r_y0=0,
            r_x1=10,
            r_y1=0,
            r_x2=10,
            r_y2=10,
            r_x3=0,
            r_y3=10,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

        assert rect.r_x0 == 0
        assert rect.r_y0 == 0
        assert rect.coord_origin == CoordOrigin.BOTTOMLEFT

    def test_width_property(self):
        """Test width calculation for rectangle."""
        # Simple rectangle (aligned with axes)
        rect = BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=0, r_x2=10, r_y2=10, r_x3=0, r_y3=10
        )

        expected_width = math.sqrt((10 - 0) ** 2 + (0 - 0) ** 2)
        assert abs(rect.width - expected_width) < 1e-6

    def test_height_property(self):
        """Test height calculation for rectangle."""
        # simple rectangle aligned with axis
        rect = BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=0, r_x2=10, r_y2=10, r_x3=0, r_y3=10
        )

        expected_height = math.sqrt((0 - 0) ** 2 + (10 - 0) ** 2)
        assert abs(rect.height - expected_height) < 1e-6

    def test_angle_property(self):
        """Test angle calculation."""
        # Horizontal rectangle (angle should be 0)
        rect_horizontal = BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=0, r_x2=10, r_y2=10, r_x3=0, r_y3=10
        )
        assert abs(rect_horizontal.angle - 0) < 1e-3

        # 45-degree rotated 4 vertices polygon
        # The angle is always referred to the orientation of the segment 0-1
        rect_diagonal = BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=10, r_x2=0, r_y2=20, r_x3=-10, r_y3=10
        )
        expected_angle = math.pi / 4  # 45 degrees in radians
        assert abs(rect_diagonal.angle - expected_angle) < 1e-1

    def test_angle_360_property(self):
        """Test angle in degrees."""
        rect = BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=0, r_x2=10, r_y2=10, r_x3=0, r_y3=10
        )
        assert rect.angle_360 == 0

    def test_centre_property(self):
        """Test center calculation."""
        rect = BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=0, r_x2=10, r_y2=10, r_x3=0, r_y3=10
        )

        center_x, center_y = rect.centre
        assert center_x == 5.0  # (0+10+10+0)/4
        assert center_y == 5.0  # (0+0+10+10)/4

    def test_bounding_properties_bottomleft(self):
        """Test l, r, t, b properties (enclosing BBox) for bottom-left origin."""
        rect = BoundingRectangle(
            r_x0=2,
            r_y0=3,
            r_x1=8,
            r_y1=1,
            r_x2=10,
            r_y2=7,
            r_x3=4,
            r_y3=9,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

        assert rect.l == 2
        assert rect.r == 10
        assert rect.t == 9
        assert rect.b == 1

    def test_bounding_properties_topleft(self):
        """Test l, r, t, b properties (enclosing Bbox) for top-left origin."""
        rect = BoundingRectangle(
            r_x0=2,
            r_y0=3,
            r_x1=8,
            r_y1=1,
            r_x2=10,
            r_y2=7,
            r_x3=4,
            r_y3=9,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        assert rect.l == 2
        assert rect.r == 10
        assert rect.t == 1
        assert rect.b == 9

    def test_to_bounding_box(self):
        """Test conversion to BoundingBox."""
        rect = BoundingRectangle(
            r_x0=0,
            r_y0=0,
            r_x1=10,
            r_y1=0,
            r_x2=10,
            r_y2=10,
            r_x3=0,
            r_y3=10,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

        bbox = rect.to_bounding_box()

        assert bbox.l == 0
        assert bbox.r == 10
        assert bbox.t == 10
        assert bbox.b == 0
        assert bbox.coord_origin == CoordOrigin.BOTTOMLEFT

    def test_from_bounding_box(self):
        """Test creation from BoundingBox."""
        bbox = BoundingBox(l=0, t=10, r=10, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
        rect = BoundingRectangle.from_bounding_box(bbox)

        assert rect.r_x0 == 0
        assert rect.r_y0 == 0
        assert rect.r_x1 == 10
        assert rect.r_y1 == 0
        assert rect.r_x2 == 10
        assert rect.r_y2 == 10
        assert rect.r_x3 == 0
        assert rect.r_y3 == 10

    def test_intersection_over_union(self):
        """Test intersection over union with Shapely and BoundingRectangle"""
        rect_1 = BoundingRectangle(
            r_x0=0,
            r_y0=0,
            r_x1=1.5,
            r_y1=0,
            r_x2=1.5,
            r_y2=10,
            r_x3=0,
            r_y3=10,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

        rect_2 = BoundingRectangle(
            r_x0=0.5,
            r_y0=0,
            r_x1=2,
            r_y1=0,
            r_x2=2,
            r_y2=10,
            r_x3=0.5,
            r_y3=10,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

        assert (rect_1.intersection_over_union(rect_2) - 0.5) < 1e-3


class TestBRectangleDocument:

    def verify(self, exp_file: Path, actual: str):
        if GEN_TEST_DATA:
            with open(exp_file, "w", encoding="utf-8") as f:
                f.write(f"{actual}\n")
        else:
            with open(exp_file, "r", encoding="utf-8") as f:
                expected = f.read().rstrip()

            assert expected == actual

    def test_doctags_creation(self):
        src = Path("./test/data/doc/rect_doc_test.json")
        doc = DoclingDocument.load_from_json(src)

        actual = doc.export_to_doctags()

        self.verify(exp_file=src.parent / f"{src.stem}.gt.txt", actual=actual)
        import re

        # Split the string into lines
        lines = actual.splitlines()

        for i, line in enumerate(lines, 1):
            rec_tags = re.findall(r"<rec_\d+>", line)
            if len(rec_tags) > 0:
                assert len(rec_tags) == 8

    def test_rec_inside(self):
        src = Path("./test/data/doc/rect_doc_test.json")
        doc = DoclingDocument.load_from_json(src)
        rec1 = doc.texts[0].prov[0].bbox
        assert isinstance(rec1, BoundingRectangle)
        assert rec1.coord_origin == CoordOrigin.TOPLEFT
