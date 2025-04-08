"""Datastructures for PaginatedDocument."""

import copy
import json
import logging
import math
import re
import typing
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from PIL import Image as PILImage
from PIL import ImageColor, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from pydantic import AnyUrl, BaseModel, Field, model_validator

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ImageRef

_logger = logging.getLogger(__name__)

PageNumber = typing.Annotated[int, Field(ge=1)]


class TextCellUnit(str, Enum):
    """Enumeration of text cell units for segmented PDF page processing."""

    CHAR = "char"
    WORD = "word"
    LINE = "line"

    def __str__(self) -> str:
        """Return string representation of the enum value."""
        return str(self.value)


class PdfPageBoundaryType(str, Enum):
    """Enumeration of PDF page boundary types."""

    ART_BOX = "art_box"
    BLEED_BOX = "bleed_box"
    CROP_BOX = "crop_box"
    MEDIA_BOX = "media_box"
    TRIM_BOX = "trim_box"

    def __str__(self) -> str:
        """Return string representation of the enum value."""
        return str(self.value)


ColorChannelValue = Annotated[int, Field(ge=0, le=255)]


class ColorRGBA(BaseModel):
    """Model representing an RGBA color value."""

    r: ColorChannelValue
    g: ColorChannelValue
    b: ColorChannelValue
    a: ColorChannelValue = 255

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the color components as a tuple."""
        return (self.r, self.g, self.b, self.a)

    def __iter__(self):
        """Yield the color components for iteration."""
        yield from (self.r, self.g, self.b, self.a)


class Coord2D(NamedTuple):
    """A 2D coordinate with x and y components."""

    x: float
    y: float


class BoundingRectangle(BaseModel):
    """Model representing a rectangular boundary with four corner points."""

    r_x0: float
    r_y0: float

    r_x1: float
    r_y1: float

    r_x2: float
    r_y2: float

    r_x3: float
    r_y3: float

    coord_origin: CoordOrigin = CoordOrigin.BOTTOMLEFT

    @property
    def width(self) -> float:
        """Calculate the width of the rectangle."""
        return np.sqrt((self.r_x1 - self.r_x0) ** 2 + (self.r_y1 - self.r_y0) ** 2)

    @property
    def height(self) -> float:
        """Calculate the height of the rectangle."""
        return np.sqrt((self.r_x3 - self.r_x0) ** 2 + (self.r_y3 - self.r_y0) ** 2)

    @property
    def angle(self) -> float:
        """Calculate the angle of the rectangle in radians (0-2pi range)."""
        p_0 = ((self.r_x0 + self.r_x3) / 2.0, (self.r_y0 + self.r_y3) / 2.0)
        p_1 = ((self.r_x1 + self.r_x2) / 2.0, (self.r_y1 + self.r_y2) / 2.0)

        delta_x, delta_y = p_1[0] - p_0[0], p_1[1] - p_0[1]

        if abs(delta_y) < 1.0e-3:
            angle = 0.0
        elif abs(delta_x) < 1.0e-3:
            angle = np.pi / 2.0 if delta_y > 0 else -np.pi / 2.0
        else:
            angle = math.atan(delta_y / delta_x)
        if delta_x < 0:
            angle += np.pi
        if angle < 0:
            angle += 2 * np.pi
        return angle

    @property
    def angle_360(self) -> int:
        """Calculate the angle of the rectangle in degrees (0-360 range)."""
        return round(self.angle / np.pi * 180)

    @property
    def centre(self):
        """Calculate the center point of the rectangle."""
        return (self.r_x0 + self.r_x1 + self.r_x2 + self.r_x3) / 4.0, (
            self.r_y0 + self.r_y1 + self.r_y2 + self.r_y3
        ) / 4.0

    def to_bounding_box(self) -> BoundingBox:
        """Convert to a BoundingBox representation."""
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            top = max([self.r_y0, self.r_y1, self.r_y2, self.r_y3])
            bottom = min([self.r_y0, self.r_y1, self.r_y2, self.r_y3])
        else:
            top = min([self.r_y0, self.r_y1, self.r_y2, self.r_y3])
            bottom = max([self.r_y0, self.r_y1, self.r_y2, self.r_y3])

        left = min([self.r_x0, self.r_x1, self.r_x2, self.r_x3])
        right = max([self.r_x0, self.r_x1, self.r_x2, self.r_x3])

        return BoundingBox(
            l=left,
            b=bottom,
            r=right,
            t=top,
            coord_origin=self.coord_origin,
        )

    @classmethod
    def from_bounding_box(cls, bbox: BoundingBox) -> "BoundingRectangle":
        """Convert a BoundingBox into a BoundingRectangle."""
        return cls(
            r_x0=bbox.l,
            r_y0=bbox.b,
            r_x2=bbox.r,
            r_y2=bbox.t,
            r_x1=bbox.r,
            r_y1=bbox.b,
            r_x3=bbox.l,
            r_y3=bbox.t,
            coord_origin=bbox.coord_origin,
        )

    def to_polygon(self) -> List[Coord2D]:
        """Convert to a list of point coordinates forming a polygon."""
        return [
            Coord2D(self.r_x0, self.r_y0),
            Coord2D(self.r_x1, self.r_y1),
            Coord2D(self.r_x2, self.r_y2),
            Coord2D(self.r_x3, self.r_y3),
        ]

    def to_bottom_left_origin(self, page_height: float) -> "BoundingRectangle":
        """Convert coordinates to use bottom-left origin.

        Args:
            page_height: The height of the page

        Returns:
            BoundingRectangle with bottom-left origin
        """
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return self
        elif self.coord_origin == CoordOrigin.TOPLEFT:
            return BoundingRectangle(
                r_x0=self.r_x0,
                r_x1=self.r_x1,
                r_x2=self.r_x2,
                r_x3=self.r_x3,
                r_y0=page_height - self.r_y0,
                r_y1=page_height - self.r_y1,
                r_y2=page_height - self.r_y2,
                r_y3=page_height - self.r_y3,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            )

    def to_top_left_origin(self, page_height: float) -> "BoundingRectangle":
        """Convert coordinates to use top-left origin.

        Args:
            page_height: The height of the page

        Returns:
            BoundingRectangle with top-left origin
        """
        if self.coord_origin == CoordOrigin.TOPLEFT:
            return self
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return BoundingRectangle(
                r_x0=self.r_x0,
                r_x1=self.r_x1,
                r_x2=self.r_x2,
                r_x3=self.r_x3,
                r_y0=page_height - self.r_y0,
                r_y1=page_height - self.r_y1,
                r_y2=page_height - self.r_y2,
                r_y3=page_height - self.r_y3,
                coord_origin=CoordOrigin.TOPLEFT,
            )


class OrderedElement(BaseModel):
    """Base model for elements that have an ordering index."""

    index: int = -1


class ColorMixin(BaseModel):
    """Mixin class that adds color attributes to a model."""

    rgba: ColorRGBA = ColorRGBA(r=0, g=0, b=0, a=255)


class TextDirection(str, Enum):
    """Enumeration for text direction options."""

    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    UNSPECIFIED = "unspecified"


class TextCell(ColorMixin, OrderedElement):
    """Model representing a text cell with positioning and content information."""

    rect: BoundingRectangle

    text: str
    orig: str

    text_direction: TextDirection = TextDirection.LEFT_TO_RIGHT

    confidence: float = 1.0
    from_ocr: bool

    def to_bounding_box(self) -> BoundingBox:
        """Convert the cell rectangle to a BoundingBox."""
        return self.rect.to_bounding_box()

    def to_bottom_left_origin(self, page_height: float):
        """Convert the cell's coordinates to use bottom-left origin.

        Args:
            page_height: The height of the page
        """
        self.rect = self.rect.to_bottom_left_origin(page_height=page_height)

    def to_top_left_origin(self, page_height: float):
        """Convert the cell's coordinates to use top-left origin.

        Args:
            page_height: The height of the page
        """
        self.rect = self.rect.to_top_left_origin(page_height=page_height)


class PdfCellRenderingMode(int, Enum):
    """Text Rendering Mode, according to PDF32000."""

    FILL_TEXT = 0
    STROKE_TEXT = 1
    FILL_THEN_STROKE = 2
    INVISIBLE = 3
    FILL_AND_CLIPPING = 4
    STROKE_AND_CLIPPING = 5
    FILL_THEN_STROKE_AND_CLIPPING = 6
    ONLY_CLIPPING = 7
    UNKNOWN = -1


class PdfTextCell(TextCell):
    """Specialized text cell for PDF documents with font information."""

    rendering_mode: (
        PdfCellRenderingMode  # Turn into enum (PDF32000 Text Rendering Mode)
    )
    widget: bool  # Determines if this belongs to fillable PDF field.

    font_key: str
    font_name: str

    from_ocr: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def update_ltr_property(cls, data: dict) -> dict:
        """Update text direction property from left_to_right flag."""
        if "left_to_right" in data:
            data["text_direction"] = (
                "left_to_right" if data["left_to_right"] else "right_to_left"
            )
        # if "ordering" in data:
        #    data["index"] = data["ordering"]
        return data


class BitmapResource(OrderedElement):
    """Model representing a bitmap resource with positioning and URI information."""

    rect: BoundingRectangle
    uri: Optional[AnyUrl] = None

    def to_bottom_left_origin(self, page_height: float):
        """Convert the resource's coordinates to use bottom-left origin.

        Args:
            page_height: The height of the page
        """
        self.rect = self.rect.to_bottom_left_origin(page_height=page_height)

    def to_top_left_origin(self, page_height: float):
        """Convert the resource's coordinates to use top-left origin.

        Args:
            page_height: The height of the page
        """
        self.rect = self.rect.to_top_left_origin(page_height=page_height)


class PdfLine(ColorMixin, OrderedElement):
    """Model representing a line in a PDF document."""

    parent_id: int
    points: List[Coord2D]
    width: float = 1.0

    coord_origin: CoordOrigin = CoordOrigin.BOTTOMLEFT

    def __len__(self) -> int:
        """Return the number of points in the line."""
        return len(self.points)

    def iterate_segments(
        self,
    ) -> Iterator[Tuple[Coord2D, Coord2D]]:
        """Iterate through line segments defined by consecutive point pairs."""
        for k in range(0, len(self.points) - 1):
            yield (self.points[k], self.points[k + 1])

    def to_bottom_left_origin(self, page_height: float):
        """Convert the line's coordinates to use bottom-left origin.

        Args:
            page_height: The height of the page
        """
        if self.coord_origin == CoordOrigin.BOTTOMLEFT:
            return self
        elif self.coord_origin == CoordOrigin.TOPLEFT:
            for i, point in enumerate(self.points):
                self.points[i] = Coord2D(point[0], page_height - point[1])

            self.coord_origin = CoordOrigin.BOTTOMLEFT

    def to_top_left_origin(self, page_height: float):
        """Convert the line's coordinates to use top-left origin.

        Args:
            page_height: The height of the page
        """
        if self.coord_origin == CoordOrigin.TOPLEFT:
            return self
        elif self.coord_origin == CoordOrigin.BOTTOMLEFT:
            for i, point in enumerate(self.points):
                self.points[i] = Coord2D(point[0], page_height - point[1])

            self.coord_origin = CoordOrigin.TOPLEFT


class PageGeometry(BaseModel):
    """Model representing dimensions of a page."""

    angle: float
    rect: BoundingRectangle

    @property
    def width(self):
        """Get the width of the page."""
        # FIXME: think about angle, boundary_type and coord_origin ...
        return self.rect.width

    @property
    def height(self):
        """Get the height of the page."""
        # FIXME: think about angle, boundary_type and coord_origin ...
        return self.rect.height

    @property
    def origin(self):
        """Get the origin point of the page."""
        # FIXME: think about angle, boundary_type and coord_origin ...
        return (self.rect.to_bounding_box().l, self.rect.to_bounding_box().b)


class PdfPageGeometry(PageGeometry):
    """Extended dimensions model specific to PDF pages with boundary types."""

    boundary_type: PdfPageBoundaryType

    art_bbox: BoundingBox
    bleed_bbox: BoundingBox
    crop_bbox: BoundingBox
    media_bbox: BoundingBox
    trim_bbox: BoundingBox

    @property
    def width(self):
        """Get the width of the PDF page based on crop box."""
        # FIXME: think about angle, boundary_type and coord_origin ...
        return self.crop_bbox.width

    @property
    def height(self):
        """Get the height of the PDF page based on crop box."""
        # FIXME: think about angle, boundary_type and coord_origin ...
        return self.crop_bbox.height

    @property
    def origin(self):
        """Get the origin point of the PDF page based on crop box."""
        # FIXME: think about angle, boundary_type and coord_origin ...
        return (self.crop_bbox.l, self.crop_bbox.b)


class SegmentedPage(BaseModel):
    """Model representing a segmented page with text cells and resources."""

    dimension: PageGeometry

    bitmap_resources: List[BitmapResource] = []

    char_cells: List[TextCell] = []
    word_cells: List[TextCell] = []
    textline_cells: List[TextCell] = []

    image: Optional[ImageRef] = None

    def iterate_cells(self, unit_type: TextCellUnit) -> Iterator[TextCell]:
        """Iterate through text cells of the specified unit type.

        Args:
            unit_type: Type of text unit to iterate through

        Returns:
            Iterator of text cells

        Raises:
            ValueError: If an incompatible unit type is provided
        """
        if unit_type == TextCellUnit.CHAR:
            yield from self.char_cells

        elif unit_type == TextCellUnit.WORD:
            yield from self.word_cells

        elif unit_type == TextCellUnit.LINE:
            yield from self.textline_cells

        else:
            raise ValueError(f"incompatible {unit_type}")


class SegmentedPdfPage(SegmentedPage):
    """Extended segmented page model specific to PDF documents."""

    # Redefine typing to use PdfPageDimensions
    dimension: PdfPageGeometry

    lines: List[PdfLine] = []

    # Redefine typing of elements to include PdfTextCell
    char_cells: List[Union[PdfTextCell, TextCell]]
    word_cells: List[Union[PdfTextCell, TextCell]]
    textline_cells: List[Union[PdfTextCell, TextCell]]

    def get_cells_in_bbox(
        self, cell_unit: TextCellUnit, bbox: BoundingBox, ios: float = 0.8
    ) -> List[Union[PdfTextCell, TextCell]]:
        """Get text cells that are within the specified bounding box.

        Args:
            cell_unit: Type of text unit to check
            bbox: Bounding box to check against
            ios: Minimum intersection over self ratio

        Returns:
            List of text cells within the bounding box
        """
        cells = []
        for page_cell in self.iterate_cells(cell_unit):
            pc = copy.deepcopy(page_cell)
            # Bring cell_bbox coord origin to the same as input bbox.coord_origin:
            if page_cell.rect.coord_origin != bbox.coord_origin:
                if bbox.coord_origin == CoordOrigin.TOPLEFT:
                    pc.rect = pc.rect.to_top_left_origin(self.dimension.height)
                elif bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                    pc.rect = pc.rect.to_bottom_left_origin(self.dimension.height)
            cell_bbox = pc.to_bounding_box()
            if cell_bbox.intersection_over_self(bbox) > ios:
                cells.append(pc)
        return cells

    def export_to_dict(self) -> Dict:
        """Export the page data to a dictionary.

        Returns:
            Dictionary representation of the page
        """
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def save_as_json(
        self,
        filename: Union[str, Path],
        indent: int = 2,
    ):
        """Save the page data as a JSON file.

        Args:
            filename: Path to save the JSON file
            indent: Indentation level for JSON formatting
        """
        if isinstance(filename, str):
            filename = Path(filename)
        out = self.export_to_dict()
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Union[str, Path]) -> "SegmentedPdfPage":
        """Load page data from a JSON file.

        Args:
            filename: Path to the JSON file

        Returns:
            Instantiated SegmentedPdfPage object
        """
        if isinstance(filename, str):
            filename = Path(filename)
        with open(filename, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def crop_text(self, cell_unit: TextCellUnit, bbox: BoundingBox, eps: float = 1.0):
        """Extract text from cells within the specified bounding box.

        Args:
            cell_unit: Type of text unit to extract
            bbox: Bounding box to extract from
            eps: Epsilon value for position comparison
        """
        selection = []
        for page_cell in self.iterate_cells(cell_unit):
            cell_bbox = page_cell.rect.to_bottom_left_origin(
                page_height=self.dimension.height
            ).to_bounding_box()

            if (
                bbox.l <= cell_bbox.l
                and cell_bbox.r <= bbox.r
                and bbox.b <= cell_bbox.b
                and cell_bbox.t <= bbox.t
            ):
                selection.append(page_cell.copy())

        selection = sorted(selection, key=lambda x: x.index)

        text = ""
        for i, cell in enumerate(selection):

            if i == 0:
                text += cell.text
            else:
                prev = selection[i - 1]

                if (
                    abs(cell.rect.r_x0 - prev.rect.r_x1) < eps
                    and abs(cell.rect.r_y0 - prev.rect.r_y1) < eps
                ):
                    text += cell.text
                else:
                    text += " "
                    text += cell.text

    def export_to_textlines(
        self,
        cell_unit: TextCellUnit,
        add_location: bool = True,
        add_fontkey: bool = False,
        add_fontname: bool = True,
    ) -> List[str]:
        """Export text cells as formatted text lines.

        Args:
            cell_unit: Type of text unit to export
            add_location: Whether to include position information
            add_fontkey: Whether to include font key information
            add_fontname: Whether to include font name information

        Returns:
            List of formatted text lines
        """
        lines: List[str] = []
        for cell in self.iterate_cells(cell_unit):

            line = ""
            if add_location:
                line += f"({cell.rect.r_x0:06.02f}, {cell.rect.r_y0:06.02f}) "
                line += f"({cell.rect.r_x1:06.02f}, {cell.rect.r_y1:06.02f}) "
                line += f"({cell.rect.r_x2:06.02f}, {cell.rect.r_y2:06.02f}) "
                line += f"({cell.rect.r_x3:06.02f}, {cell.rect.r_y3:06.02f}) "

            if add_fontkey and isinstance(cell, PdfTextCell):
                line += f"{cell.font_key:>10} "

            if add_fontname and isinstance(cell, PdfTextCell):
                line += f"{cell.font_name:>10} "

            line += f"{cell.text}"
            lines.append(line)

        return lines

    def render_as_image(
        self,
        cell_unit: TextCellUnit,
        boundary_type: PdfPageBoundaryType = PdfPageBoundaryType.CROP_BOX,  # media_box
        draw_cells_bbox: bool = False,
        draw_cells_text: bool = True,
        draw_cells_bl: bool = False,
        draw_cells_tr: bool = False,
        cell_outline: str = "black",
        cell_color: str = "cyan",
        cell_alpha: float = 1.0,
        cell_bl_color: str = "red",
        cell_bl_outline: str = "red",
        cell_bl_alpha: float = 1.0,
        cell_bl_radius: float = 3.0,
        cell_tr_color: str = "green",
        cell_tr_outline: str = "green",
        cell_tr_alpha: float = 1.0,
        cell_tr_radius: float = 3.0,
        draw_bitmap_resources: bool = True,
        bitmap_resources_outline: str = "black",
        bitmap_resources_fill: str = "yellow",
        bitmap_resources_alpha: float = 1.0,
        draw_lines: bool = True,
        line_color: str = "black",
        line_width: int = 1,
        line_alpha: float = 1.0,
        draw_annotations: bool = True,
        annotations_outline: str = "white",
        annotations_color: str = "green",
        annotations_alpha: float = 0.5,
        draw_crop_box: bool = True,
        cropbox_outline: str = "red",
        cropbox_width: int = 3,
        cropbox_alpha: float = 1.0,
    ) -> PILImage.Image:
        """Render the page as an image with various visualization options.

        Args:
            cell_unit: Type of text unit to render
            boundary_type: Type of page boundary to use
            draw_cells_bbox: Whether to draw bounding boxes for cells
            draw_cells_text: Whether to draw text content of cells
            draw_cells_bl: Whether to draw bottom left points of cells
            draw_cells_tr: Whether to draw top right points of cells
            cell_outline: Color for cell outlines
            cell_color: Fill color for cells
            cell_alpha: Alpha value for cell visualization
            cell_bl_color: Color for bottom left points
            cell_bl_outline: Outline color for bottom left points
            cell_bl_alpha: Alpha value for bottom left points
            cell_bl_radius: Radius for bottom left points
            cell_tr_color: Color for top right points
            cell_tr_outline: Outline color for top right points
            cell_tr_alpha: Alpha value for top right points
            cell_tr_radius: Radius for top right points
            draw_bitmap_resources: Whether to draw bitmap resources
            bitmap_resources_outline: Outline color for bitmap resources
            bitmap_resources_fill: Fill color for bitmap resources
            bitmap_resources_alpha: Alpha value for bitmap resources
            draw_lines: Whether to draw lines
            line_color: Color for lines
            line_width: Width for lines
            line_alpha: Alpha value for lines
            draw_annotations: Whether to draw annotations
            annotations_outline: Outline color for annotations
            annotations_color: Fill color for annotations
            annotations_alpha: Alpha value for annotations
            draw_crop_box: Whether to draw crop box
            cropbox_outline: Color for crop box outline
            cropbox_width: Width for crop box outline
            cropbox_alpha: Alpha value for crop box

        Returns:
            PIL Image of the rendered page
        """
        for _ in [
            cell_alpha,
            cell_bl_alpha,
            cell_tr_alpha,
            bitmap_resources_alpha,
            line_alpha,
            annotations_alpha,
            cropbox_alpha,
        ]:
            if _ < 0 or 1.0 < _:
                logging.error(f"alpha value {_} needs to be in [0, 1]")
                _ = max(0, min(1.0, _))

        page_bbox = self.dimension.crop_bbox

        page_width = page_bbox.width
        page_height = page_bbox.height

        # Create a blank white image with RGBA mode
        result = PILImage.new(
            "RGBA", (round(page_width), round(page_height)), (255, 255, 255, 255)
        )
        draw = ImageDraw.Draw(result)

        # Draw each rectangle by connecting its four points
        if draw_bitmap_resources:
            draw = self._render_bitmap_resources(
                draw=draw,
                page_height=page_height,
                bitmap_resources_fill=bitmap_resources_fill,
                bitmap_resources_outline=bitmap_resources_outline,
                bitmap_resources_alpha=bitmap_resources_alpha,
            )

        if draw_cells_text:
            result = self._render_cells_text(
                cell_unit=cell_unit, img=result, page_height=page_height
            )

        elif draw_cells_bbox:
            self._render_cells_bbox(
                cell_unit=cell_unit,
                draw=draw,
                page_height=page_height,
                cell_fill=cell_color,
                cell_outline=cell_outline,
                cell_alpha=cell_alpha,
            )

        if draw_cells_bl:
            self._draw_cells_bl(
                cell_unit=cell_unit,
                draw=draw,
                page_height=page_height,
                cell_bl_color=cell_bl_color,
                cell_bl_outline=cell_bl_outline,
                cell_bl_alpha=cell_bl_alpha,
                cell_bl_radius=cell_bl_radius,
            )

        if draw_cells_tr:
            self._draw_cells_tr(
                cell_unit=cell_unit,
                draw=draw,
                page_height=page_height,
                cell_tr_color=cell_tr_color,
                cell_tr_outline=cell_tr_outline,
                cell_tr_alpha=cell_tr_alpha,
                cell_tr_radius=cell_tr_radius,
            )

        if draw_lines:
            draw = self._render_lines(
                draw=draw,
                page_height=page_height,
                line_color=line_color,
                line_alpha=line_alpha,
                line_width=line_width,
            )

        return result

    def _get_rgba(self, name: str, alpha: float):
        """Get RGBA tuple from color name and alpha value.

        Args:
            name: Color name
            alpha: Alpha value between 0 and 1

        Returns:
            RGBA tuple

        Raises:
            AssertionError: If alpha is out of range
        """
        assert 0.0 <= alpha and alpha <= 1.0, "0.0 <= alpha and alpha <= 1.0"
        rgba = ImageColor.getrgb(name) + (int(alpha * 255),)
        return rgba

    def _render_bitmap_resources(
        self,
        draw: ImageDraw.ImageDraw,
        page_height: float,
        bitmap_resources_fill: str,
        bitmap_resources_outline: str,
        bitmap_resources_alpha: float,
    ) -> ImageDraw.ImageDraw:
        """Render bitmap resources on the page.

        Args:
            draw: PIL ImageDraw object
            page_height: Height of the page
            bitmap_resources_fill: Fill color for bitmap resources
            bitmap_resources_outline: Outline color for bitmap resources
            bitmap_resources_alpha: Alpha value for bitmap resources

        Returns:
            Updated ImageDraw object
        """
        for bitmap_resource in self.bitmap_resources:
            poly = bitmap_resource.rect.to_top_left_origin(
                page_height=page_height
            ).to_polygon()

            fill = self._get_rgba(
                name=bitmap_resources_fill, alpha=bitmap_resources_alpha
            )
            outline = self._get_rgba(
                name=bitmap_resources_outline, alpha=bitmap_resources_alpha
            )

            draw.polygon(poly, outline=outline, fill=fill)

        return draw

    def _render_cells_bbox(
        self,
        cell_unit: TextCellUnit,
        draw: ImageDraw.ImageDraw,
        page_height: float,
        cell_fill: str,
        cell_outline: str,
        cell_alpha: float,
    ) -> ImageDraw.ImageDraw:
        """Render bounding boxes for text cells.

        Args:
            cell_unit: Type of text unit to render
            draw: PIL ImageDraw object
            page_height: Height of the page
            cell_fill: Fill color for cells
            cell_outline: Outline color for cells
            cell_alpha: Alpha value for cells

        Returns:
            Updated ImageDraw object
        """
        fill = self._get_rgba(name=cell_fill, alpha=cell_alpha)
        outline = self._get_rgba(name=cell_outline, alpha=cell_alpha)

        # Draw each rectangle by connecting its four points
        for page_cell in self.iterate_cells(unit_type=cell_unit):
            poly = page_cell.rect.to_top_left_origin(
                page_height=page_height
            ).to_polygon()
            draw.polygon(poly, outline=outline, fill=fill)

        return draw

    def _draw_text_in_rectangle(
        self,
        img: PILImage.Image,
        rect: BoundingRectangle,
        text: str,
        font: Optional[Union[FreeTypeFont, ImageFont.ImageFont]] = None,
        fill: str = "black",
    ) -> PILImage.Image:
        """Draw text within a rectangular boundary with rotation.

        Args:
            img: PIL Image to draw on
            rect: Rectangle defining the text boundary
            text: Text content to draw
            font: Font to use for drawing text
            fill: Text color

        Returns:
            Updated PIL Image
        """
        width = round(rect.width)
        height = round(rect.height)
        rot_angle = rect.angle_360

        centre = rect.centre
        centre_x, centre_y = round(centre[0]), round(centre[1])

        # print(f"width: {width}, height: {height}, angle: {rot_angle}, text: {text}")

        if width <= 2 or height <= 2:
            # logging.warning(f"skipping to draw text
            # (width: {x1-x0}, height: {y1-y0}): {text}")
            return img

        # Use the default font if no font is provided
        if font is None:
            font = ImageFont.load_default()

        # Create a temporary image for the text
        tmp_img = PILImage.new("RGBA", (1, 1), (255, 255, 255, 0))  # Dummy size
        tmp_draw = ImageDraw.Draw(tmp_img)
        _, _, text_width, text_height = tmp_draw.textbbox((0, 0), text=text, font=font)

        # Create a properly sized temporary image
        text_img = PILImage.new(
            "RGBA", (round(text_width), round(text_height)), (255, 255, 255, 255)
        )
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), text, font=font, fill=(0, 0, 0, 255))

        # Resize image
        text_img = text_img.resize((width, height), PILImage.Resampling.LANCZOS)

        # Rotate img_1
        rotated_img = text_img.rotate(rot_angle, expand=True)

        # Compute new position for pasting
        rotated_w, rotated_h = rotated_img.size
        paste_x = centre_x - rotated_w // 2
        paste_y = centre_y - rotated_h // 2

        # Paste rotated image onto img_2
        img.paste(rotated_img, (paste_x, paste_y), rotated_img)

        return img

    def _render_cells_text(
        self, cell_unit: TextCellUnit, img: PILImage.Image, page_height: float
    ) -> PILImage.Image:
        """Render text content of cells on the image.

        Args:
            cell_unit: Type of text unit to render
            img: PIL Image to draw on
            page_height: Height of the page

        Returns:
            Updated PIL Image
        """
        # Draw each rectangle by connecting its four points
        for page_cell in self.iterate_cells(unit_type=cell_unit):
            rect = page_cell.rect.to_top_left_origin(page_height=page_height)
            img = self._draw_text_in_rectangle(
                img=img,
                rect=rect,
                text=page_cell.text,
            )

        return img

    def _draw_cells_bl(
        self,
        cell_unit: TextCellUnit,
        draw: ImageDraw.ImageDraw,
        page_height: float,
        cell_bl_color: str,
        cell_bl_outline: str,
        cell_bl_alpha: float,
        cell_bl_radius: float,
    ) -> ImageDraw.ImageDraw:
        """Draw bottom-left points of text cells.

        Args:
            cell_unit: Type of text unit to render
            draw: PIL ImageDraw object
            page_height: Height of the page
            cell_bl_color: Fill color for bottom-left points
            cell_bl_outline: Outline color for bottom-left points
            cell_bl_alpha: Alpha value for bottom-left points
            cell_bl_radius: Radius for bottom-left points

        Returns:
            Updated ImageDraw object
        """
        fill = self._get_rgba(name=cell_bl_color, alpha=cell_bl_alpha)
        outline = self._get_rgba(name=cell_bl_outline, alpha=cell_bl_alpha)

        # Draw each rectangle by connecting its four points
        for page_cell in self.iterate_cells(unit_type=cell_unit):
            poly = page_cell.rect.to_top_left_origin(
                page_height=page_height
            ).to_polygon()
            # Define the bounding box for the dot
            dot_bbox = [
                (poly[0][0] - cell_bl_radius, poly[0][1] - cell_bl_radius),
                (poly[0][0] + cell_bl_radius, poly[0][1] + cell_bl_radius),
            ]

            # Draw the red dot
            draw.ellipse(dot_bbox, fill=fill, outline=outline)

        return draw

    def _draw_cells_tr(
        self,
        cell_unit: TextCellUnit,
        draw: ImageDraw.ImageDraw,
        page_height: float,
        cell_tr_color: str,
        cell_tr_outline: str,
        cell_tr_alpha: float,
        cell_tr_radius: float,
    ) -> ImageDraw.ImageDraw:
        """Draw top-right points of text cells.

        Args:
            cell_unit: Type of text unit to render
            draw: PIL ImageDraw object
            page_height: Height of the page
            cell_tr_color: Fill color for top-right points
            cell_tr_outline: Outline color for top-right points
            cell_tr_alpha: Alpha value for top-right points
            cell_tr_radius: Radius for top-right points

        Returns:
            Updated ImageDraw object
        """
        fill = self._get_rgba(name=cell_tr_color, alpha=cell_tr_alpha)
        outline = self._get_rgba(name=cell_tr_outline, alpha=cell_tr_alpha)

        # Draw each rectangle by connecting its four points
        for page_cell in self.iterate_cells(unit_type=cell_unit):
            poly = page_cell.rect.to_top_left_origin(
                page_height=page_height
            ).to_polygon()
            # Define the bounding box for the dot
            dot_bbox = [
                (poly[0][0] - cell_tr_radius, poly[0][1] - cell_tr_radius),
                (poly[0][0] + cell_tr_radius, poly[0][1] + cell_tr_radius),
            ]

            # Draw the red dot
            draw.ellipse(dot_bbox, fill=fill, outline=outline)

        return draw

    def _render_lines(
        self,
        draw: ImageDraw.ImageDraw,
        page_height: float,
        line_color: str,
        line_alpha: float,
        line_width: float,
    ) -> ImageDraw.ImageDraw:
        """Render lines on the page.

        Args:
            draw: PIL ImageDraw object
            page_height: Height of the page
            line_color: Color for lines
            line_alpha: Alpha value for lines
            line_width: Width for lines

        Returns:
            Updated ImageDraw object
        """
        fill = self._get_rgba(name=line_color, alpha=line_alpha)

        # Draw each rectangle by connecting its four points
        for line in self.lines:

            line.to_top_left_origin(page_height=page_height)
            for segment in line.iterate_segments():
                draw.line(
                    (segment[0][0], segment[0][1], segment[1][0], segment[1][1]),
                    fill=fill,
                    width=max(1, round(line.width)),
                )

        return draw


class PdfMetaData(BaseModel):
    """Model representing PDF metadata extracted from XML."""

    xml: str = ""

    data: Dict[str, str] = {}

    def initialise(self):
        """Initialize metadata by parsing the XML content."""
        # Define the regex pattern
        pattern = r"\<([a-zA-Z]+)\:([a-zA-Z]+)\>(.+?)\<\/([a-zA-Z]+)\:([a-zA-Z]+)\>"

        # Find all matches
        matches = re.findall(pattern, self.xml)

        # Process matches
        for _ in matches:
            namespace_open, tag_open, content, namespace_close, tag_close = _
            if namespace_open == namespace_close and tag_open == tag_close:
                _logger.debug(
                    f"Namespace: {namespace_open}, Tag: {tag_open}, Content: {content}"
                )
                self.data[tag_open] = content


class PdfTableOfContents(BaseModel):
    """Model representing a PDF table of contents entry with hierarchical structure."""

    text: str
    orig: str = ""

    marker: str = ""

    children: List["PdfTableOfContents"] = []

    def export_to_dict(self, mode: str = "json") -> Dict:
        """Export the table of contents to a dictionary.

        Args:
            mode: Serialization mode

        Returns:
            Dictionary representation of the table of contents
        """
        return self.model_dump(mode=mode, by_alias=True, exclude_none=True)

    def save_as_json(self, filename: Union[str, Path], indent: int = 2):
        """Save the table of contents as a JSON file.

        Args:
            filename: Path to save the JSON file
            indent: Indentation level for JSON formatting
        """
        if isinstance(filename, str):
            filename = Path(filename)
        out = self.export_to_dict()
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Union[str, Path]) -> "PdfTableOfContents":
        """Load table of contents from a JSON file.

        Args:
            filename: Path to the JSON file

        Returns:
            Instantiated PdfTableOfContents object
        """
        if isinstance(filename, str):
            filename = Path(filename)
        with open(filename, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


class ParsedPdfDocument(BaseModel):
    """Model representing a completely parsed PDF document with all components."""

    pages: Dict[PageNumber, SegmentedPdfPage] = {}

    meta_data: Optional[PdfMetaData] = None
    table_of_contents: Optional[PdfTableOfContents] = None

    def iterate_pages(
        self,
    ) -> Iterator[Tuple[int, SegmentedPdfPage]]:
        """Iterate through all pages in the document.

        Returns:
            Iterator of (page number, page) tuples
        """
        for page_no, page in self.pages.items():
            yield (page_no, page)

    def export_to_dict(
        self,
        mode: str = "json",
    ) -> Dict:
        """Export the document to a dictionary.

        Args:
            mode: Serialization mode

        Returns:
            Dictionary representation of the document
        """
        return self.model_dump(mode=mode, by_alias=True, exclude_none=True)

    def save_as_json(self, filename: Union[str, Path], indent: int = 2):
        """Save the document as a JSON file.

        Args:
            filename: Path to save the JSON file
            indent: Indentation level for JSON formatting
        """
        if isinstance(filename, str):
            filename = Path(filename)
        out = self.export_to_dict()
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Union[str, Path]) -> "ParsedPdfDocument":
        """Load document from a JSON file.

        Args:
            filename: Path to the JSON file

        Returns:
            Instantiated ParsedPdfDocument object
        """
        if isinstance(filename, str):
            filename = Path(filename)
        with open(filename, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
