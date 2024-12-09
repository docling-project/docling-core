#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define common models across CCS objects."""
from typing import Annotated, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, PositiveInt, StrictStr

from docling_core.search.mapping import es_field
from docling_core.types.legacy_doc.tokens import DocumentToken
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

    prov: Optional[list[Prov]] = None
    text: Optional[str] = Field(
        default=None, json_schema_extra=es_field(term_vector="with_positions_offsets")
    )
    obj_type: str = Field(
        alias="type", json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    payload: Optional[dict] = None

    def get_location_tokens(
        self,
        new_line: str,
        page_w: float,
        page_h: float,
        xsize: int = 100,
        ysize: int = 100,
        add_page_index: bool = True,
    ) -> str:
        """Get the location string for the BaseCell."""
        if self.prov is None:
            return ""

        location = ""
        for prov in self.prov:

            page_i = -1
            if add_page_index:
                page_i = prov.page

            loc_str = DocumentToken.get_location(
                bbox=prov.bbox,
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
                page_i=page_i,
            )
            location += f"{loc_str}{new_line}"

        return location


class Table(BaseCell):
    """Table."""

    num_cols: int = Field(alias="#-cols")
    num_rows: int = Field(alias="#-rows")
    data: Optional[list[list[Union[GlmTableCell, TableCell]]]] = None
    model: Optional[str] = None

    # FIXME: we need to check why we have bounding_box (this should be in prov)
    bounding_box: Optional[BoundingBoxContainer] = Field(
        default=None, alias="bounding-box", json_schema_extra=es_field(suppress=True)
    )

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

    def export_to_document_tokens(
        self,
        new_line: str = "\n",
        page_w: float = 0.0,
        page_h: float = 0.0,
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,
        add_cell_location: bool = True,
        add_cell_label: bool = True,
        add_cell_text: bool = True,
        add_page_index: bool = True,
    ):
        """Export table to document tokens format."""
        body = f"{DocumentToken.BEG_TABLE.value}{new_line}"

        if add_location:
            body += self.get_location_tokens(
                new_line=new_line,
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_caption and self.text is not None and len(self.text) > 0:
            body += f"{DocumentToken.BEG_CAPTION.value}"
            body += f"{self.text.strip()}"
            body += f"{DocumentToken.END_CAPTION.value}"
            body += f"{new_line}"

        if add_content and self.data is not None and len(self.data) > 0:
            for i, row in enumerate(self.data):
                body += f"<row_{i}>"
                for j, col in enumerate(row):

                    text = ""
                    if add_cell_text:
                        text = col.text.strip()

                    cell_loc = ""
                    if (
                        col.bbox is not None
                        and add_cell_location
                        and add_page_index
                        and self.prov is not None
                        and len(self.prov) > 0
                    ):
                        cell_loc = DocumentToken.get_location(
                            bbox=col.bbox,
                            page_w=page_w,
                            page_h=page_h,
                            xsize=xsize,
                            ysize=ysize,
                            page_i=self.prov[0].page,
                        )
                    elif (
                        col.bbox is not None
                        and add_cell_location
                        and not add_page_index
                    ):
                        cell_loc = DocumentToken.get_location(
                            bbox=col.bbox,
                            page_w=page_w,
                            page_h=page_h,
                            xsize=xsize,
                            ysize=ysize,
                            page_i=-1,
                        )

                    cell_label = ""
                    if (
                        add_cell_label
                        and col.obj_type is not None
                        and len(col.obj_type) > 0
                    ):
                        cell_label = f"<{col.obj_type}>"

                    body += f"<col_{j}>{cell_loc}{cell_label}{text}</col_{j}>"

                body += f"</row_{i}>{new_line}"

        body += f"{DocumentToken.END_TABLE.value}{new_line}"

        return body


# FIXME: let's add some figure specific data-types later
class Figure(BaseCell):
    """Figure."""

    # FIXME: we need to check why we have bounding_box (this should be in prov)
    bounding_box: Optional[BoundingBoxContainer] = Field(
        default=None, alias="bounding-box", json_schema_extra=es_field(suppress=True)
    )

    def export_to_document_tokens(
        self,
        new_line: str = "\n",
        page_w: float = 0.0,
        page_h: float = 0.0,
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,  # not used at the moment
        add_page_index: bool = True,
    ):
        """Export figure to document tokens format."""
        body = f"{DocumentToken.BEG_FIGURE.value}{new_line}"

        if add_location:
            body += self.get_location_tokens(
                new_line=new_line,
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_caption and self.text is not None and len(self.text) > 0:
            body += f"{DocumentToken.BEG_CAPTION.value}"
            body += f"{self.text.strip()}"
            body += f"{DocumentToken.END_CAPTION.value}"
            body += f"{new_line}"

        body += f"{DocumentToken.END_FIGURE.value}{new_line}"

        return body


class BaseText(BaseCell):
    """Base model for text objects."""

    # FIXME: do we need these ???
    name: Optional[StrictStr] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    font: Optional[str] = None

    def export_to_document_tokens(
        self,
        new_line: str = "\n",
        page_w: float = 0.0,
        page_h: float = 0.0,
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
    ):
        """Export text element to document tokens format."""
        body = f"<{self.obj_type}>"

        assert DocumentToken.is_known_token(
            body
        ), f"failed DocumentToken.is_known_token({body})"

        if add_location:
            body += self.get_location_tokens(
                new_line="",
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_content and self.text is not None:
            body += self.text.strip()

        body += f"</{self.obj_type}>{new_line}"

        return body


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
