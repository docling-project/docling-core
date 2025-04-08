import math

import numpy as np
import pytest

from docling_core.types.doc.page import BoundingRectangle

SQRT_2 = math.sqrt(2)

R_0 = BoundingRectangle(r_x0=0, r_y0=0, r_x1=1, r_y1=0, r_x2=1, r_y2=1, r_x3=0, r_y3=1)
R_45 = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=SQRT_2 / 2,
    r_y1=SQRT_2 / 2,
    r_x2=0,
    r_y2=SQRT_2,
    r_x3=-SQRT_2 / 2,
    r_y3=SQRT_2 / 2,
)
R_90 = BoundingRectangle(
    r_x0=0, r_y0=0, r_x1=0, r_y1=1, r_x2=-1, r_y2=1, r_x3=-1, r_y3=0
)
R_135 = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-SQRT_2 / 2,
    r_y1=SQRT_2 / 2,
    r_x2=-SQRT_2,
    r_y2=0,
    r_x3=-SQRT_2 / 2,
    r_y3=-SQRT_2 / 2,
)
R_180 = BoundingRectangle(
    r_x0=0, r_y0=0, r_x1=-0, r_y1=0, r_x2=-1, r_y2=-1, r_x3=0, r_y3=-1
)
R_MINUS_135 = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=-SQRT_2 / 2,
    r_y1=-SQRT_2 / 2,
    r_x2=0,
    r_y2=-SQRT_2,
    r_x3=SQRT_2 / 2,
    r_y3=-SQRT_2 / 2,
)
R_MINUS_90 = BoundingRectangle(
    r_x0=0, r_y0=0, r_x1=0, r_y1=-1, r_x2=1, r_y2=-1, r_x3=1, r_y3=0
)
R_MINUS_45 = BoundingRectangle(
    r_x0=0,
    r_y0=0,
    r_x1=SQRT_2 / 2,
    r_y1=-SQRT_2 / 2,
    r_x2=SQRT_2,
    r_y2=0,
    r_x3=SQRT_2 / 2,
    r_y3=SQRT_2 / 2,
)


@pytest.mark.parametrize(
    ("rectangle", "expected_angle", "expected_angle_360"),
    [
        (R_0, 0, 0.0),
        (R_45, np.pi / 4, 45),
        (R_90, np.pi / 2, 90),
        (R_135, 3 * np.pi / 4, 135),
        (R_180, np.pi, 180),
        (R_MINUS_135, 5 * np.pi / 4, 225),
        (R_MINUS_90, 3 * np.pi / 2, 270),
        (R_MINUS_45, 7 * np.pi / 4, 315),
    ],
)
def test_bounding_rectangle_angle(
    rectangle: BoundingRectangle, expected_angle: float, expected_angle_360: int
):
    assert pytest.approx(rectangle.angle, abs=1e-6) == expected_angle
    assert pytest.approx(rectangle.angle_360, abs=1e-6) == expected_angle_360
