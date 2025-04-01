"""Models for the base data types."""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel


class ImageRefMode(str, Enum):
    """ImageRefMode."""

    PLACEHOLDER = "placeholder"  # just a place-holder
    EMBEDDED = "embedded"  # embed the image as a base64
    REFERENCED = "referenced"  # reference the image via uri


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

    def resize_by_scale(self, x_scale: float, y_scale: float):
        """resize_by_scale."""
        return BoundingBox(
            l=self.l * x_scale,
            r=self.r * x_scale,
            t=self.t * y_scale,
            b=self.b * y_scale,
            coord_origin=self.coord_origin,
        )

    def scale_to_size(self, old_size: Size, new_size: Size):
        """scale_to_size."""
        return self.resize_by_scale(
            x_scale=new_size.width / old_size.width,
            y_scale=new_size.height / old_size.height,
        )

    # same as before, but using the implementation above
    def scaled(self, scale: float):
        """scaled."""
        return self.resize_by_scale(x_scale=scale, y_scale=scale)

    # same as before, but using the implementation above
    def normalized(self, page_size: Size):
        """normalized."""
        return self.scale_to_size(
            old_size=page_size, new_size=Size(height=1.0, width=1.0)
        )

    def expand_by_scale(self, x_scale: float, y_scale: float) -> "BoundingBox":
        """expand_to_size."""
        if self.coord_origin == CoordOrigin.TOPLEFT:
            return BoundingBox(
                l=self.l - self.width * x_scale,
                r=self.r + self.width * x_scale,
                t=self.t - self.height * y_scale,
                b=self.b + self.height * y_scale,
                coord_origin=self.coord_origin,
            )
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return BoundingBox(
                l=self.l - self.width * x_scale,
                r=self.r + self.width * x_scale,
                t=self.t + self.height * y_scale,
                b=self.b - self.height * y_scale,
                coord_origin=self.coord_origin,
            )

    def as_tuple(self) -> Tuple[float, float, float, float]:
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
        return abs(self.r - self.l) * abs(self.b - self.t)

    def intersection_area_with(self, other: "BoundingBox") -> float:
        """Calculate the intersection area with another bounding box."""
        if self.coord_origin != other.coord_origin:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        # Calculate intersection coordinates
        left = max(self.l, other.l)
        right = min(self.r, other.r)

        if self.coord_origin == CoordOrigin.TOPLEFT:
            bottom = max(self.t, other.t)
            top = min(self.b, other.b)
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            top = min(self.t, other.t)
            bottom = max(self.b, other.b)

        # Calculate intersection dimensions
        width = right - left
        height = top - bottom

        # If the bounding boxes do not overlap, width or height will be negative
        if width <= 0 or height <= 0:
            return 0.0

        return width * height

    def intersection_over_union(
        self, other: "BoundingBox", eps: float = 1.0e-6
    ) -> float:
        """intersection_over_union."""
        intersection_area = self.intersection_area_with(other=other)

        union_area = (
            abs(self.l - self.r) * abs(self.t - self.b)
            + abs(other.l - other.r) * abs(other.t - other.b)
            - intersection_area
        )

        return intersection_area / (union_area + eps)

    def intersection_over_self(
        self, other: "BoundingBox", eps: float = 1.0e-6
    ) -> float:
        """intersection_over_self."""
        intersection_area = self.intersection_area_with(other=other)
        if self.area() > 0:
            return intersection_area / self.area()
        else:
            return 0.0

    def to_bottom_left_origin(self, page_height: float) -> "BoundingBox":
        """to_bottom_left_origin.

        :param page_height:

        """
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return self.model_copy()
        elif self.coord_origin == CoordOrigin.TOPLEFT:
            return BoundingBox(
                l=self.l,
                r=self.r,
                t=page_height - self.t,
                b=page_height - self.b,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            )

    def to_top_left_origin(self, page_height: float) -> "BoundingBox":
        """to_top_left_origin.

        :param page_height:

        """
        if self.coord_origin == CoordOrigin.TOPLEFT:
            return self.model_copy()
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return BoundingBox(
                l=self.l,
                r=self.r,
                t=page_height - self.t,  # self.b
                b=page_height - self.b,  # self.t
                coord_origin=CoordOrigin.TOPLEFT,
            )

    def overlaps(self, other: "BoundingBox") -> bool:
        """overlaps."""
        return self.overlaps_horizontally(other=other) and self.overlaps_vertically(
            other=other
        )

    def overlaps_horizontally(self, other: "BoundingBox") -> bool:
        """Check if two bounding boxes overlap horizontally."""
        return not (self.r <= other.l or other.r <= self.l)

    def overlaps_vertically(self, other: "BoundingBox") -> bool:
        """Check if two bounding boxes overlap vertically."""
        if self.coord_origin != other.coord_origin:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        # Normalize coordinates if needed
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return not (self.t <= other.b or other.t <= self.b)
        elif self.coord_origin == CoordOrigin.TOPLEFT:
            return not (self.b <= other.t or other.b <= self.t)

    def overlaps_vertically_with_iou(self, other: "BoundingBox", iou: float) -> bool:
        """overlaps_y_with_iou."""
        if (
            self.coord_origin == CoordOrigin.BOTTOMLEFT
            and other.coord_origin == CoordOrigin.BOTTOMLEFT
        ):

            if self.overlaps_vertically(other=other):

                u0 = min(self.b, other.b)
                u1 = max(self.t, other.t)

                i0 = max(self.b, other.b)
                i1 = min(self.t, other.t)

                iou_ = float(i1 - i0) / float(u1 - u0)
                return (iou_) > iou

            return False

        elif (
            self.coord_origin == CoordOrigin.TOPLEFT
            and other.coord_origin == CoordOrigin.TOPLEFT
        ):
            if self.overlaps_vertically(other=other):
                u0 = min(self.t, other.t)
                u1 = max(self.b, other.b)

                i0 = max(self.t, other.t)
                i1 = min(self.b, other.b)

                iou_ = float(i1 - i0) / float(u1 - u0)
                return (iou_) > iou

            return False
        else:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        return False

    def is_left_of(self, other: "BoundingBox") -> bool:
        """is_left_of."""
        return self.l < other.l

    def is_strictly_left_of(self, other: "BoundingBox", eps: float = 0.001) -> bool:
        """is_strictly_left_of."""
        return (self.r + eps) < other.l

    def is_above(self, other: "BoundingBox") -> bool:
        """is_above."""
        if (
            self.coord_origin == CoordOrigin.BOTTOMLEFT
            and other.coord_origin == CoordOrigin.BOTTOMLEFT
        ):
            return self.t > other.t

        elif (
            self.coord_origin == CoordOrigin.TOPLEFT
            and other.coord_origin == CoordOrigin.TOPLEFT
        ):
            return self.t < other.t

        else:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        return False

    def is_strictly_above(self, other: "BoundingBox", eps: float = 1.0e-3) -> bool:
        """is_strictly_above."""
        if (
            self.coord_origin == CoordOrigin.BOTTOMLEFT
            and other.coord_origin == CoordOrigin.BOTTOMLEFT
        ):
            return (self.b + eps) > other.t

        elif (
            self.coord_origin == CoordOrigin.TOPLEFT
            and other.coord_origin == CoordOrigin.TOPLEFT
        ):
            return (self.b + eps) < other.t

        else:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        return False

    def is_horizontally_connected(
        self, elem_i: "BoundingBox", elem_j: "BoundingBox"
    ) -> bool:
        """is_horizontally_connected."""
        if (
            self.coord_origin == CoordOrigin.BOTTOMLEFT
            and elem_i.coord_origin == CoordOrigin.BOTTOMLEFT
            and elem_j.coord_origin == CoordOrigin.BOTTOMLEFT
        ):
            min_ij = min(elem_i.b, elem_j.b)
            max_ij = max(elem_i.t, elem_j.t)

            if self.b < max_ij and min_ij < self.t:  # overlap_y
                return False

            if self.l < elem_i.r and elem_j.l < self.r:
                return True

            return False

        elif (
            self.coord_origin == CoordOrigin.TOPLEFT
            and elem_i.coord_origin == CoordOrigin.TOPLEFT
            and elem_j.coord_origin == CoordOrigin.TOPLEFT
        ):
            min_ij = min(elem_i.t, elem_j.t)
            max_ij = max(elem_i.b, elem_j.b)

            if self.t < max_ij and min_ij < self.b:  # overlap_y
                return False

            if self.l < elem_i.r and elem_j.l < self.r:
                return True

            return False

        else:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        return False

    @classmethod
    def enclosing_bbox(cls, boxes: List["BoundingBox"]) -> "BoundingBox":
        """Create a bounding box that covers all of the given boxes."""
        if not boxes:
            raise ValueError("No bounding boxes provided for union.")

        origin = boxes[0].coord_origin
        if any(box.coord_origin != origin for box in boxes):
            raise ValueError(
                "All bounding boxes must have the same \
                CoordOrigin to compute their union."
            )

        left = min(box.l for box in boxes)
        right = max(box.r for box in boxes)

        if origin == CoordOrigin.TOPLEFT:
            top = min(box.t for box in boxes)
            bottom = max(box.b for box in boxes)
        elif origin == CoordOrigin.BOTTOMLEFT:
            top = max(box.t for box in boxes)
            bottom = min(box.b for box in boxes)
        else:
            raise ValueError("BoundingBoxes have different CoordOrigin")

        return cls(l=left, t=top, r=right, b=bottom, coord_origin=origin)
