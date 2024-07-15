#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define common models across CCS objects."""
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, StrictStr

from docling_core.search.mapping import es_field
from docling_core.utils.alias import AliasModel

CellData = tuple[float, float, float, float, str, str]

CellHeader = tuple[
    Literal["x0"],
    Literal["y0"],
    Literal["x1"],
    Literal["y1"],
    Literal["font"],
    Literal["text"],
]

BoundingBox = Annotated[list[float], Field(min_length=4, max_length=4)]

Span = Annotated[list[int], Field(min_length=2, max_length=2)]


class CellsContainer(BaseModel):
    """Cell container."""

    data: Optional[list[CellData]] = None
    header: CellHeader = ("x0", "y0", "x1", "y1", "font", "text")


class S3Resource(BaseModel):
    """Resource in a cloud object storage."""

    mime: str
    path: str
    page: Optional[int] = None


class S3Data(AliasModel):
    """Data object in a cloud object storage."""

    pdf_document: Optional[list[S3Resource]] = Field(default=None, alias="pdf-document")
    pdf_pages: Optional[list[S3Resource]] = Field(default=None, alias="pdf-pages")
    pdf_images: Optional[list[S3Resource]] = Field(default=None, alias="pdf-images")
    json_document: Optional[S3Resource] = Field(default=None, alias="json-document")
    json_meta: Optional[S3Resource] = Field(default=None, alias="json-meta")
    glm_json_document: Optional[S3Resource] = Field(
        default=None, alias="glm-json-document"
    )
    figures: Optional[list[S3Resource]] = None


class S3Reference(AliasModel):
    """References an s3 resource."""

    ref_s3_data: StrictStr = Field(
        alias="__ref_s3_data", examples=["#/_s3_data/figures/0"]
    )


class Prov(AliasModel):
    """Provenance."""

    bbox: BoundingBox
    page: int
    span: Span
    ref_s3_data: Optional[StrictStr] = Field(
        default=None, alias="__ref_s3_data", json_schema_extra=es_field(suppress=True)
    )


class BoundingBoxContainer(BaseModel):
    """Bounding box container."""

    min: BoundingBox
    max: BoundingBox


class BitmapObject(AliasModel):
    """Bitmap object."""

    obj_type: str = Field(alias="type")
    bounding_box: BoundingBoxContainer = Field(
        json_schema_extra=es_field(suppress=True)
    )
    prov: Prov


class PageDimensions(BaseModel):
    """Page dimensions."""

    height: float
    page: int
    width: float


class TableCell(AliasModel):
    """Table cell."""

    bbox: Optional[BoundingBox] = None
    spans: Optional[list[Span]] = None
    text: str = Field(json_schema_extra=es_field(term_vector="with_positions_offsets"))
    obj_type: str = Field(alias="type")


class GlmTableCell(TableCell):
    """Glm Table cell."""

    col: Optional[int] = Field(default=None, json_schema_extra=es_field(suppress=True))
    col_header: bool = Field(
        default=False, alias="col-header", json_schema_extra=es_field(suppress=True)
    )
    col_span: Optional[Span] = Field(
        default=None, alias="col-span", json_schema_extra=es_field(suppress=True)
    )
    row: Optional[int] = Field(default=None, json_schema_extra=es_field(suppress=True))
    row_header: bool = Field(
        default=False, alias="row-header", json_schema_extra=es_field(suppress=True)
    )
    row_span: Optional[Span] = Field(
        default=None, alias="row-span", json_schema_extra=es_field(suppress=True)
    )


class Table(AliasModel):
    """Table."""

    num_cols: int = Field(alias="#-cols")
    num_rows: int = Field(alias="#-rows")
    bounding_box: Optional[BoundingBoxContainer] = Field(
        default=None, alias="bounding-box", json_schema_extra=es_field(suppress=True)
    )
    data: Optional[list[list[Union[GlmTableCell, TableCell]]]] = None
    model: Optional[str] = None
    prov: Optional[list[Prov]] = None
    text: Optional[str] = Field(
        default=None, json_schema_extra=es_field(term_vector="with_positions_offsets")
    )
    obj_type: str = Field(
        alias="type",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )


class BaseCell(AliasModel):
    """Base cell."""

    bounding_box: Optional[BoundingBoxContainer] = Field(
        default=None, alias="bounding-box", json_schema_extra=es_field(suppress=True)
    )
    prov: Optional[list[Prov]] = None
    text: Optional[str] = None
    obj_type: str = Field(
        alias="type", json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )


class BaseText(AliasModel):
    """Base model for text objects."""

    text: StrictStr = Field(
        json_schema_extra=es_field(term_vector="with_positions_offsets")
    )
    obj_type: StrictStr = Field(
        alias="type", json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    name: Optional[StrictStr] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    font: Optional[str] = None
    prov: Optional[list[Prov]] = None


class ListItem(BaseText):
    """List item."""

    identifier: str


class Ref(AliasModel):
    """Reference."""

    name: str
    obj_type: str = Field(alias="type")
    ref: str = Field(alias="$ref")


class PageReference(BaseModel):
    """Page reference."""

    hash: str = Field(json_schema_extra=es_field(type="keyword", ignore_above=8191))
    model: str = Field(json_schema_extra=es_field(suppress=True))
    page: int = Field(json_schema_extra=es_field(type="short"))
