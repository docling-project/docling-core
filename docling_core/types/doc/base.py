"""Models for the base data types."""

import copy
from enum import Enum
from typing import Tuple

from pydantic import BaseModel


class ImageRefMode(str, Enum):
    """ImageRefMode."""

    PLACEHOLDER = "placeholder"
    EMBEDDED = "embedded"


class CoordOrigin(str, Enum):
    """CoordOrigin."""

    TOPLEFT = "TOPLEFT"
    BOTTOMLEFT = "BOTTOMLEFT"


class Size(BaseModel):
    """Size."""

    width: float = 0.0
    height: float = 0.0

    def as_tuple(self):
        """as_tuple."""
        return (self.width, self.height)


class BoundingBox(BaseModel):
    """BoundingBox."""

    l: float  # left
    t: float  # top
    r: float  # right
    b: float  # bottom

    coord_origin: CoordOrigin = CoordOrigin.TOPLEFT

    @property
    def width(self):
        """width."""
        return self.r - self.l

    @property
    def height(self):
        """height."""
        return abs(self.t - self.b)

    def scaled(self, scale: float) -> "BoundingBox":
        """scaled.

        :param scale: float:

        """
        out_bbox = copy.deepcopy(self)
        out_bbox.l *= scale
        out_bbox.r *= scale
        out_bbox.t *= scale
        out_bbox.b *= scale

        return out_bbox

    def normalized(self, page_size: Size) -> "BoundingBox":
        """normalized.

        :param page_size: Size:

        """
        out_bbox = copy.deepcopy(self)
        out_bbox.l /= page_size.width
        out_bbox.r /= page_size.width
        out_bbox.t /= page_size.height
        out_bbox.b /= page_size.height

        return out_bbox

    def as_tuple(self):
        """as_tuple."""
        if self.coord_origin == CoordOrigin.TOPLEFT:
            return (self.l, self.t, self.r, self.b)
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return (self.l, self.b, self.r, self.t)

    @classmethod
    def from_tuple(cls, coord: Tuple[float, ...], origin: CoordOrigin):
        """from_tuple.

        :param coord: Tuple[float:
        :param ...]:
        :param origin: CoordOrigin:

        """
        if origin == CoordOrigin.TOPLEFT:
            l, t, r, b = coord[0], coord[1], coord[2], coord[3]
            if r < l:
                l, r = r, l
            if b < t:
                b, t = t, b

            return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=origin)
        elif origin == CoordOrigin.BOTTOMLEFT:
            l, b, r, t = coord[0], coord[1], coord[2], coord[3]
            if r < l:
                l, r = r, l
            if b > t:
                b, t = t, b

            return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=origin)

    def area(self) -> float:
        """area."""
        area = (self.r - self.l) * (self.b - self.t)
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            area = -area
        return area

    def intersection_area_with(self, other: "BoundingBox") -> float:
        """intersection_area_with.

        :param other: "BoundingBox":

        """
        # Calculate intersection coordinates
        left = max(self.l, other.l)
        top = max(self.t, other.t)
        right = min(self.r, other.r)
        bottom = min(self.b, other.b)

        # Calculate intersection dimensions
        width = right - left
        height = bottom - top

        # If the bounding boxes do not overlap, width or height will be negative
        if width <= 0 or height <= 0:
            return 0.0

        return width * height

    def to_bottom_left_origin(self, page_height) -> "BoundingBox":
        """to_bottom_left_origin.

        :param page_height:

        """
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return self
        elif self.coord_origin == CoordOrigin.TOPLEFT:
            return BoundingBox(
                l=self.l,
                r=self.r,
                t=page_height - self.t,
                b=page_height - self.b,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            )

    def to_top_left_origin(self, page_height):
        """to_top_left_origin.

        :param page_height:

        """
        if self.coord_origin == CoordOrigin.TOPLEFT:
            return self
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return BoundingBox(
                l=self.l,
                r=self.r,
                t=page_height - self.t,  # self.b
                b=page_height - self.b,  # self.t
                coord_origin=CoordOrigin.TOPLEFT,
            )
