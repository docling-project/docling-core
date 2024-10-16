#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models for CCS objects with OCR."""
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from docling_core.types.legacy_doc.base import BoundingBox
from docling_core.utils.alias import AliasModel

CoordsOrder = Literal["x1", "y1", "x2", "y2"]

CoordsOrigin = Literal["top-left"]  # TODO

Info = Dict[str, Any]  # TODO


class Page(BaseModel):
    """Page."""

    width: float
    height: float


class Meta(AliasModel):
    """Meta."""

    page: Page
    coords_order: List[CoordsOrder] = Field(..., alias="coords-order")
    coords_origin: CoordsOrigin = Field(..., alias="coords-origin")


class Dimension(BaseModel):
    """Dimension."""

    width: float
    height: float


class Word(BaseModel):
    """Word."""

    confidence: float
    bbox: BoundingBox
    content: str


class Cell(BaseModel):
    """Cell."""

    confidence: float
    bbox: BoundingBox
    content: str


class Box(BaseModel):
    """Box."""

    confidence: float
    bbox: BoundingBox
    content: str


class Path(BaseModel):
    """Path."""

    x: List[float]
    y: List[float]


class OcrOutput(AliasModel):
    """OCR output."""

    meta: Meta = Field(..., alias="_meta")
    info: Info
    dimension: Dimension
    words: List[Word]
    cells: List[Cell]
    boxes: List[Box]
    paths: List[Path]
