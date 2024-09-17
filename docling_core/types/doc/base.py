#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define common models across CCS objects."""
from typing import Annotated, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, PositiveInt, StrictStr

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
    page: Optional[PositiveInt] = None


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
    page: PositiveInt
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
    page: PositiveInt
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


class BaseCell(AliasModel):
    """Base cell."""

    # FIXME: we need to check why we have bounding_box (this should be in prov)
    bounding_box: Optional[BoundingBoxContainer] = Field(
        default=None, alias="bounding-box", json_schema_extra=es_field(suppress=True)
    )
    prov: Optional[list[Prov]] = None
    text: Optional[str] = Field(
        default=None, json_schema_extra=es_field(term_vector="with_positions_offsets")
    )
    obj_type: str = Field(
        alias="type", json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )


class Table(BaseCell):
    """Table."""

    num_cols: int = Field(alias="#-cols")
    num_rows: int = Field(alias="#-rows")
    data: Optional[list[list[Union[GlmTableCell, TableCell]]]] = None
    model: Optional[str] = None

    def _get_tablecell_span(self, cell: TableCell, ix: int):
        if cell.spans is None:
            span = set()
        else:
            span = set([s[ix] for s in cell.spans])
        if len(span) == 0:
            return 1, None, None
        return len(span), min(span), max(span)

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export the table as a Pandas DataFrame."""
        if self.data is None or self.num_rows == 0 or self.num_cols == 0:
            return pd.DataFrame()

        # Count how many rows are column headers
        num_headers = 0
        for i, row in enumerate(self.data):
            if len(row) == 0:
                raise RuntimeError(f"Invalid table. {len(row)=} but {self.num_cols=}.")

            any_header = False
            for cell in row:
                if cell.obj_type == "col_header":
                    any_header = True
                    break

            if any_header:
                num_headers += 1
            else:
                break

        # Create the column names from all col_headers
        columns: Optional[List[str]] = None
        if num_headers > 0:
            columns = ["" for _ in range(self.num_cols)]
            for i in range(num_headers):
                for j, cell in enumerate(self.data[i]):
                    col_name = cell.text
                    if columns[j] != "":
                        col_name = f".{col_name}"
                    columns[j] += col_name

        # Create table data
        table_data = [[cell.text for cell in row] for row in self.data[num_headers:]]

        # Create DataFrame
        df = pd.DataFrame(table_data, columns=columns)

        return df

    def export_to_html(self) -> str:
        """Export the table as html."""
        body = ""
        nrows = self.num_rows
        ncols = self.num_cols

        if self.data is None:
            return ""
        for i in range(nrows):
            body += "<tr>"
            for j in range(ncols):
                cell: TableCell = self.data[i][j]

                rowspan, rowstart, rowend = self._get_tablecell_span(cell, 0)
                colspan, colstart, colend = self._get_tablecell_span(cell, 1)

                if rowstart is not None and rowstart != i:
                    continue
                if colstart is not None and colstart != j:
                    continue

                if rowstart is None:
                    rowstart = i
                if colstart is None:
                    colstart = j

                content = cell.text.strip()
                label = cell.obj_type
                celltag = "td"
                if label in ["row_header", "row_multi_header", "row_title"]:
                    pass
                elif label in ["col_header", "col_multi_header"]:
                    celltag = "th"

                opening_tag = f"{celltag}"
                if rowspan > 1:
                    opening_tag += f' rowspan="{rowspan}"'
                if colspan > 1:
                    opening_tag += f' colspan="{colspan}"'

                body += f"<{opening_tag}>{content}</{celltag}>"
            body += "</tr>"
        body = f"<table>{body}</table>"

        return body


# FIXME: let's add some figure specific data-types later
class Figure(BaseCell):
    """Figure."""


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
    page: PositiveInt = Field(json_schema_extra=es_field(type="short"))
