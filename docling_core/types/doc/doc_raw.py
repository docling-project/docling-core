#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models for CCS objects in raw format."""
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from docling_core.types.doc.base import BoundingBox
from docling_core.utils.alias import AliasModel

FontDifferences = dict[str, Any]
NamedWidths = dict[str, Any]
IgnoredCell = Any


class Box(BaseModel):
    """Box."""

    baseline: BoundingBox
    device: BoundingBox


class Content(BaseModel):
    """Content."""

    rnormalized: str


class Enumeration(BaseModel):
    """Enumeration."""

    match: int
    type: int


class Font(BaseModel):
    """Font."""

    color: Annotated[List[float], Field(min_length=3, max_length=4)]
    name: str
    size: float


class Cell(AliasModel):
    """Cell."""

    see_cell: bool = Field(..., alias="SEE_cell")
    see_confidence: float = Field(..., alias="SEE_confidence")
    angle: float
    box: Box
    content: Content
    enumeration: Enumeration
    font: Font


class PageDimensions(BaseModel):
    """PageDimensions."""

    bbox: BoundingBox
    height: float
    width: float


class Path(AliasModel):
    """Path."""

    bbox: BoundingBox
    sub_paths: list[float] = Field(..., alias="sub-paths")
    type: str
    x_values: list[float] = Field(..., alias="x-values")
    y_values: list[float] = Field(..., alias="y-values")


class VerticalLine(BaseModel):
    """Vertical line."""

    y0: int
    y1: int
    x: int


class HorizontalLine(BaseModel):
    """Horizontal line."""

    x0: int
    x1: int
    y: int


class Image(BaseModel):
    """Image."""

    box: BoundingBox
    height: float
    width: float


class FontRange(BaseModel):
    """Font range."""

    first: int
    second: int


class FontCmap(BaseModel):
    """Font cmap."""

    cmap: dict[str, str]
    name: str
    range: FontRange
    type: int


class FontMetrics(AliasModel):
    """Font metrics."""

    stem_h: float = Field(..., alias="StemH")
    stem_v: float = Field(..., alias="StemV")
    ascent: float
    average_width: float = Field(..., alias="average-width")
    bbox: BoundingBox
    cap_height: float
    default_width: float = Field(..., alias="default-width")
    descent: float
    file: str
    italic_angle: float = Field(..., alias="italic-angle")
    max_width: float = Field(..., alias="max-width")
    missing_width: float = Field(..., alias="missing-width")
    name: str
    named_widths: NamedWidths = Field(..., alias="named-widths")
    weight: str
    widths: dict[str, float]
    x_height: float


class FontInfo(AliasModel):
    """Font info."""

    font_cmap: FontCmap = Field(..., alias="font-cmap")
    font_differences: FontDifferences = Field(..., alias="font-differences")
    font_metrics: FontMetrics = Field(..., alias="font-metrics")
    name: str
    internal_name: str = Field(..., alias="name (internal)")
    subtype: str


class Page(AliasModel):
    """Page."""

    height: float
    width: float
    dimensions: PageDimensions
    cells: list[Cell]
    paths: list[Path]
    vertical_lines: Optional[list[VerticalLine]] = Field(..., alias="vertical-lines")
    horizontal_lines: Optional[list[HorizontalLine]] = Field(
        ..., alias="horizontal-lines"
    )
    ignored_cells: list[IgnoredCell] = Field(..., alias="ignored-cells")
    images: list[Image]
    fonts: dict[str, FontInfo]


class Histograms(AliasModel):
    """Histogram."""

    mean_char_height: dict[str, float] = Field(..., alias="mean-char-height")
    mean_char_width: dict[str, float] = Field(..., alias="mean-char-width")
    number_of_chars: dict[str, int] = Field(..., alias="number-of-chars")


class PdfInfo(BaseModel):
    """PDF info."""

    histograms: Histograms
    styles: list[str]


class RawPdf(BaseModel):
    """Raw PDF."""

    info: PdfInfo
    pages: list[Page]
