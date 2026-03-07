import math

import numpy as np
import pytest
from pydantic import AnyUrl

from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, PdfHyperlink

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
