import math

import numpy as np
import pytest
from pydantic import AnyUrl, ValidationError

from docling_core.types.doc import CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    Coord2D,
    PdfDestination,
    PdfDestinationKind,
    PdfHyperlink,
    PdfTableOfContents,
)

SQRT_2 = math.sqrt(2)

R_0_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=1,
    r_y1=0,
    r_x2=1,
    r_y2=1,
    r_x3=0,
    r_y3=1,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_0_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=1,
    r_y1=0,
    r_x2=1,
    r_y2=1,
    r_x3=0,
    r_y3=1,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_45_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=SQRT_2 / 2,
    r_y1=SQRT_2 / 2,
    r_x2=0,
    r_y2=SQRT_2,
    r_x3=-SQRT_2 / 2,
    r_y3=SQRT_2 / 2,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_45_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=SQRT_2 / 2,
    r_y1=-SQRT_2 / 2,
    r_x2=0,
    r_y2=-SQRT_2,
    r_x3=-SQRT_2 / 2,
    r_y3=-SQRT_2 / 2,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_90_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=0,
    r_y1=1,
    r_x2=-1,
    r_y2=1,
    r_x3=-1,
    r_y3=0,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_90_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=0,
    r_y1=-1,
    r_x2=-1,
    r_y2=-1,
    r_x3=-1,
    r_y3=0,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_135_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-SQRT_2 / 2,
    r_y1=SQRT_2 / 2,
    r_x2=-SQRT_2,
    r_y2=0,
    r_x3=-SQRT_2 / 2,
    r_y3=-SQRT_2 / 2,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_135_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-SQRT_2 / 2,
    r_y1=-SQRT_2 / 2,
    r_x2=-SQRT_2,
    r_y2=0,
    r_x3=-SQRT_2 / 2,
    r_y3=SQRT_2 / 2,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_180_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-0,
    r_y1=0,
    r_x2=-1,
    r_y2=-1,
    r_x3=0,
    r_y3=-1,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_180_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-0,
    r_y1=0,
    r_x2=-1,
    r_y2=1,
    r_x3=0,
    r_y3=1,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_225_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-SQRT_2 / 2,
    r_y1=-SQRT_2 / 2,
    r_x2=0,
    r_y2=-SQRT_2,
    r_x3=SQRT_2 / 2,
    r_y3=-SQRT_2 / 2,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_225_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-SQRT_2 / 2,
    r_y1=SQRT_2 / 2,
    r_x2=0,
    r_y2=SQRT_2,
    r_x3=SQRT_2 / 2,
    r_y3=SQRT_2 / 2,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_270_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=0,
    r_y1=-1,
    r_x2=1,
    r_y2=-1,
    r_x3=1,
    r_y3=0,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_270_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=0,
    r_y1=1,
    r_x2=1,
    r_y2=1,
    r_x3=1,
    r_y3=0,
    coord_origin=CoordOrigin.TOPLEFT,
)
R_315_BL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=SQRT_2 / 2,
    r_y1=-SQRT_2 / 2,
    r_x2=SQRT_2,
    r_y2=0,
    r_x3=SQRT_2 / 2,
    r_y3=SQRT_2 / 2,
    coord_origin=CoordOrigin.BOTTOMLEFT,
)
R_315_TL = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=SQRT_2 / 2,
    r_y1=SQRT_2 / 2,
    r_x2=SQRT_2,
    r_y2=0,
    r_x3=SQRT_2 / 2,
    r_y3=-SQRT_2 / 2,
    coord_origin=CoordOrigin.TOPLEFT,
)


@pytest.mark.parametrize(
    ("rectangle", "expected_angle", "expected_angle_360"),
    [
        (R_0_BL, 0, 0.0),
        (R_45_BL, np.pi / 4, 45),
        (R_90_BL, np.pi / 2, 90),
        (R_135_BL, 3 * np.pi / 4, 135),
        (R_180_BL, np.pi, 180),
        (R_225_BL, 5 * np.pi / 4, 225),
        (R_270_BL, 3 * np.pi / 2, 270),
        (R_315_BL, 7 * np.pi / 4, 315),
        (R_0_TL, 0, 0.0),
        (R_45_TL, np.pi / 4, 45),
        (R_90_TL, np.pi / 2, 90),
        (R_135_TL, 3 * np.pi / 4, 135),
        (R_180_TL, np.pi, 180),
        (R_225_TL, 5 * np.pi / 4, 225),
        (R_270_TL, 3 * np.pi / 2, 270),
        (R_315_TL, 7 * np.pi / 4, 315),
    ],
)
def test_bounding_rectangle_angle(rectangle: BoundingRectangle, expected_angle: float, expected_angle_360: int):
    assert pytest.approx(rectangle.angle, abs=1e-6) == expected_angle
    assert pytest.approx(rectangle.angle_360, abs=1e-6) == expected_angle_360


# -- PdfHyperlink URI validation tests --

RECT = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=1,
    r_y1=0,
    r_x2=1,
    r_y2=1,
    r_x3=0,
    r_y3=1,
    coord_origin=CoordOrigin.TOPLEFT,
)


class TestPdfHyperlinkUri:
    """PdfHyperlink.uri should accept any URI form found in real PDFs."""

    def test_absolute_url_parsed_as_anyurl(self):
        h = PdfHyperlink(rect=RECT, uri="https://example.com/page")
        assert isinstance(h.uri, AnyUrl)
        assert h.uri.scheme == "https"
        assert h.uri.host == "example.com"

    def test_mailto_parsed_as_anyurl(self):
        h = PdfHyperlink(rect=RECT, uri="mailto:user@example.com")
        assert isinstance(h.uri, AnyUrl)
        assert h.uri.scheme == "mailto"

    def test_relative_path_falls_back_to_str(self):
        h = PdfHyperlink(
            rect=RECT,
            uri="/wiki/pages/internal-document-link",
        )
        assert isinstance(h.uri, str)
        assert h.uri == "/wiki/pages/internal-document-link"

    def test_fragment_only_falls_back_to_str(self):
        h = PdfHyperlink(rect=RECT, uri="#internal-bookmark")
        assert isinstance(h.uri, str)
        assert h.uri == "#internal-bookmark"

    def test_relative_path_falls_back_to_str_dotdot(self):
        h = PdfHyperlink(rect=RECT, uri="../relative/path.html")
        assert isinstance(h.uri, str)
        assert h.uri == "../relative/path.html"

    def test_none_uri(self):
        h = PdfHyperlink(rect=RECT, uri=None)
        assert h.uri is None

    def test_omitted_uri(self):
        h = PdfHyperlink(rect=RECT)
        assert h.uri is None


def _dest(
    page_no: int = 1,
    *,
    kind: PdfDestinationKind = PdfDestinationKind.XYZ,
    point: Coord2D | None = Coord2D(x=72.0, y=690.0),
    coord_origin: CoordOrigin = CoordOrigin.BOTTOMLEFT,
    height: float = 800.0,
) -> PdfDestination:
    return PdfDestination(
        page_no=page_no,
        kind=kind,
        point=point,
        coord_origin=coord_origin,
        page_size=Size(width=600.0, height=height),
    )


def _toc(text: str, *children: PdfTableOfContents, destination: PdfDestination | None = None) -> PdfTableOfContents:
    return PdfTableOfContents(text=text, destination=destination, children=list(children))


class TestPdfTableOfContentsCompatibility:
    """A table of contents without destinations must behave exactly as it did before."""

    def test_destination_defaults_to_none(self):
        toc = PdfTableOfContents(text="Introduction")
        assert toc.destination is None

    def test_destination_omitted_from_export(self):
        toc = _toc("<root>", _toc("Introduction"))
        assert toc.export_to_dict() == {
            "text": "<root>",
            "orig": "",
            "marker": "",
            "children": [{"text": "Introduction", "orig": "", "marker": "", "children": []}],
        }

    def test_legacy_payload_still_validates(self):
        toc = PdfTableOfContents.model_validate(
            {"text": "<root>", "orig": "", "marker": "", "children": [{"text": "Introduction"}]}
        )
        assert toc.children[0].text == "Introduction"
        assert toc.children[0].destination is None

    def test_destination_round_trips(self):
        toc = _toc("<root>", _toc("Introduction", destination=_dest(3)))
        restored = PdfTableOfContents.model_validate(toc.export_to_dict())
        assert restored.children[0].destination == toc.children[0].destination


class TestPdfTableOfContentsIterate:
    """iterate() yields descendants pre-order, with the level derived from the tree depth."""

    def test_yields_document_order_and_levels(self):
        toc = _toc(
            "<root>",
            _toc("Introduction"),
            _toc("Model Architecture", _toc("Dense Models", _toc("Scaling")), _toc("Mixture-of-Expert models")),
            _toc("Conclusion"),
        )
        assert [(level, entry.text) for level, entry in toc.iterate()] == [
            (0, "Introduction"),
            (0, "Model Architecture"),
            (1, "Dense Models"),
            (2, "Scaling"),
            (1, "Mixture-of-Expert models"),
            (0, "Conclusion"),
        ]

    def test_root_itself_is_not_yielded(self):
        toc = _toc("<root>", _toc("Introduction"))
        assert [entry.text for _, entry in toc.iterate()] == ["Introduction"]

    def test_leaf_yields_nothing(self):
        assert list(_toc("Introduction").iterate()) == []

    def test_deep_chain_does_not_recurse(self):
        leaf = _toc("level-4999")
        for depth in reversed(range(4999)):
            leaf = _toc(f"level-{depth}", leaf)
        root = _toc("<root>", leaf)

        levels = [level for level, _ in root.iterate()]

        assert len(levels) == 5000
        assert levels == list(range(5000))


class TestPdfDestinationOrigin:
    """Changing the coordinate origin flips y against the target page height only."""

    def test_to_top_left_origin_flips_y(self):
        dest = _dest(point=Coord2D(x=72.0, y=690.0), height=800.0).to_top_left_origin()
        assert dest.coord_origin == CoordOrigin.TOPLEFT
        assert dest.point == Coord2D(x=72.0, y=110.0)

    def test_to_top_left_origin_preserves_everything_else(self):
        source = _dest(page_no=3, kind=PdfDestinationKind.FIT_R)
        dest = source.to_top_left_origin()
        assert (dest.page_no, dest.kind, dest.page_size) == (source.page_no, source.kind, source.page_size)
        assert dest.point is not None and source.point is not None
        assert dest.point.x == source.point.x

    def test_conversion_to_same_origin_is_a_no_op(self):
        source = _dest(coord_origin=CoordOrigin.TOPLEFT)
        dest = source.to_top_left_origin()
        assert dest == source
        assert dest is not source

    def test_round_trip(self):
        source = _dest()
        assert source.to_top_left_origin().to_bottom_left_origin() == source

    def test_missing_point_survives_conversion(self):
        dest = _dest(kind=PdfDestinationKind.FIT, point=None).to_top_left_origin()
        assert dest.point is None
        assert dest.coord_origin == CoordOrigin.TOPLEFT


class TestPdfDestinationValidation:
    """page_no is a PageNumber and kind is a closed enumeration."""

    @pytest.mark.parametrize("page_no", [0, -1])
    def test_page_no_must_be_positive(self, page_no: int):
        with pytest.raises(ValidationError):
            _dest(page_no=page_no)

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValidationError):
            PdfDestination.model_validate(
                {"page_no": 1, "kind": "FIT_XY", "page_size": {"width": 600.0, "height": 800.0}}
            )

    def test_unknown_kind_is_the_documented_fallback(self):
        dest = PdfDestination(page_no=1, page_size=Size(width=600.0, height=800.0))
        assert dest.kind == PdfDestinationKind.UNKNOWN
        assert dest.point is None
        assert dest.coord_origin == CoordOrigin.BOTTOMLEFT
