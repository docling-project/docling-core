"""Models for the Docling Document data type."""

import base64
import copy
import hashlib
import json
import logging
import mimetypes
import os
import re
import sys
import typing
import warnings
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Sequence, Tuple, Union
from urllib.parse import unquote

import pandas as pd
import yaml
from PIL import Image as PILImage
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    StringConstraints,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
    validate_call,
)
from tabulate import tabulate
from typing_extensions import Annotated, Self, deprecated, override

from docling_core.search.package import VERSION_PATTERN
from docling_core.types.base import _JSON_POINTER_REGEX
from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.base import (
    CoordOrigin,
    ImageRefMode,
    PydanticSerCtxKey,
    round_pydantic_float,
)
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    GraphCellLabel,
    GraphLinkLabel,
    GroupLabel,
    PictureClassificationLabel,
)
from docling_core.types.doc.tokens import DocumentToken, TableToken
from docling_core.types.doc.utils import parse_otsl_table_content, relative_path

_logger = logging.getLogger(__name__)

Uint64 = typing.Annotated[int, Field(ge=0, le=(2**64 - 1))]
LevelNumber = typing.Annotated[int, Field(ge=1, le=100)]
CURRENT_VERSION: Final = "1.6.0"

DEFAULT_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.KEY_VALUE_REGION,
}

DOCUMENT_TOKENS_EXPORT_LABELS = DEFAULT_EXPORT_LABELS.copy()
DOCUMENT_TOKENS_EXPORT_LABELS.update(
    [
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.FORM,
    ]
)


class BaseAnnotation(BaseModel):
    """Base class for all annotation types."""

    kind: str


class PictureClassificationClass(BaseModel):
    """PictureClassificationData."""

    class_name: str
    confidence: float

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)


class PictureClassificationData(BaseAnnotation):
    """PictureClassificationData."""

    kind: Literal["classification"] = "classification"
    provenance: str
    predicted_classes: List[PictureClassificationClass]


class DescriptionAnnotation(BaseAnnotation):
    """DescriptionAnnotation."""

    kind: Literal["description"] = "description"
    text: str
    provenance: str


class PictureMoleculeData(BaseAnnotation):
    """PictureMoleculeData."""

    kind: Literal["molecule_data"] = "molecule_data"
    smi: str
    confidence: float
    class_name: str
    segmentation: List[Tuple[float, float]]
    provenance: str

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)


class MiscAnnotation(BaseAnnotation):
    """MiscAnnotation."""

    kind: Literal["misc"] = "misc"
    content: Dict[str, Any]


class ChartLine(BaseModel):
    """Represents a line in a line chart.

    Attributes:
        label (str): The label for the line.
        values (List[Tuple[float, float]]): A list of (x, y) coordinate pairs
            representing the line's data points.
    """

    label: str
    values: List[Tuple[float, float]]


class ChartBar(BaseModel):
    """Represents a bar in a bar chart.

    Attributes:
        label (str): The label for the bar.
        values (float): The value associated with the bar.
    """

    label: str
    values: float


class ChartStackedBar(BaseModel):
    """Represents a stacked bar in a stacked bar chart.

    Attributes:
        label (List[str]): The labels for the stacked bars. Multiple values are stored
            in cases where the chart is "double stacked," meaning bars are stacked both
            horizontally and vertically.
        values (List[Tuple[str, int]]): A list of values representing different segments
            of the stacked bar along with their label.
    """

    label: List[str]
    values: List[Tuple[str, int]]


class ChartSlice(BaseModel):
    """Represents a slice in a pie chart.

    Attributes:
        label (str): The label for the slice.
        value (float): The value represented by the slice.
    """

    label: str
    value: float


class ChartPoint(BaseModel):
    """Represents a point in a scatter chart.

    Attributes:
        value (Tuple[float, float]): A (x, y) coordinate pair representing a point in a
            chart.
    """

    value: Tuple[float, float]


class PictureChartData(BaseAnnotation):
    """Base class for picture chart data.

    Attributes:
        title (str): The title of the chart.
    """

    title: str


class PictureLineChartData(PictureChartData):
    """Represents data of a line chart.

    Attributes:
        kind (Literal["line_chart_data"]): The type of the chart.
        x_axis_label (str): The label for the x-axis.
        y_axis_label (str): The label for the y-axis.
        lines (List[ChartLine]): A list of lines in the chart.
    """

    kind: Literal["line_chart_data"] = "line_chart_data"
    x_axis_label: str
    y_axis_label: str
    lines: List[ChartLine]


class PictureBarChartData(PictureChartData):
    """Represents data of a bar chart.

    Attributes:
        kind (Literal["bar_chart_data"]): The type of the chart.
        x_axis_label (str): The label for the x-axis.
        y_axis_label (str): The label for the y-axis.
        bars (List[ChartBar]): A list of bars in the chart.
    """

    kind: Literal["bar_chart_data"] = "bar_chart_data"
    x_axis_label: str
    y_axis_label: str
    bars: List[ChartBar]


class PictureStackedBarChartData(PictureChartData):
    """Represents data of a stacked bar chart.

    Attributes:
        kind (Literal["stacked_bar_chart_data"]): The type of the chart.
        x_axis_label (str): The label for the x-axis.
        y_axis_label (str): The label for the y-axis.
        stacked_bars (List[ChartStackedBar]): A list of stacked bars in the chart.
    """

    kind: Literal["stacked_bar_chart_data"] = "stacked_bar_chart_data"
    x_axis_label: str
    y_axis_label: str
    stacked_bars: List[ChartStackedBar]


class PicturePieChartData(PictureChartData):
    """Represents data of a pie chart.

    Attributes:
        kind (Literal["pie_chart_data"]): The type of the chart.
        slices (List[ChartSlice]): A list of slices in the pie chart.
    """

    kind: Literal["pie_chart_data"] = "pie_chart_data"
    slices: List[ChartSlice]


class PictureScatterChartData(PictureChartData):
    """Represents data of a scatter chart.

    Attributes:
        kind (Literal["scatter_chart_data"]): The type of the chart.
        x_axis_label (str): The label for the x-axis.
        y_axis_label (str): The label for the y-axis.
        points (List[ChartPoint]): A list of points in the scatter chart.
    """

    kind: Literal["scatter_chart_data"] = "scatter_chart_data"
    x_axis_label: str
    y_axis_label: str
    points: List[ChartPoint]


class TableCell(BaseModel):
    """TableCell."""

    bbox: Optional[BoundingBox] = None
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """from_dict_format."""
        if isinstance(data, dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                # "bbox" not in data
                # or data["bbox"] is None
                # or isinstance(data["bbox"], BoundingBox)
                "text"
                in data
            ):
                return data
            text = data.get("bbox", {}).get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "

                text = text.strip()
            data["text"] = text

        return data

    def _get_text(self, doc: Optional["DoclingDocument"] = None, **kwargs: Any) -> str:
        return self.text


class RichTableCell(TableCell):
    """RichTableCell."""

    ref: "RefItem"

    @override
    def _get_text(self, doc: Optional["DoclingDocument"] = None, **kwargs: Any) -> str:
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        if doc is not None:
            doc_serializer = kwargs.pop(
                "doc_serializer", MarkdownDocSerializer(doc=doc)
            )
            ser_res = doc_serializer.serialize(item=self.ref.resolve(doc=doc), **kwargs)
            return ser_res.text
        else:
            return "<!-- rich cell -->"


AnyTableCell = Annotated[
    Union[RichTableCell, TableCell],
    Field(union_mode="left_to_right"),
]


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: List[AnyTableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field  # type: ignore
    @property
    def grid(
        self,
    ) -> List[List[TableCell]]:
        """grid."""
        # Initialise empty table data grid (only empty cells)
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # Overwrite cells in table data for which there is actual cell content.
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data

    def remove_rows(
        self, indices: List[int], doc: Optional["DoclingDocument"] = None
    ) -> List[List[TableCell]]:
        """Remove rows from the table by their indices.

        :param indices: List[int]: A list of indices of the rows to remove. (Starting from 0)

        :return: List[List[TableCell]]: A list representation of the removed rows as lists of TableCell objects.
        """
        if not indices:
            return []

        indices = sorted(indices, reverse=True)

        refs_to_remove = []
        all_removed_cells = []
        for row_index in indices:
            if row_index < 0 or row_index >= self.num_rows:
                raise IndexError(
                    f"Row index {row_index} is out of bounds for the current number of rows {self.num_rows}."
                )

            start_idx = row_index * self.num_cols
            end_idx = start_idx + self.num_cols
            removed_cells = self.table_cells[start_idx:end_idx]

            for cell in removed_cells:
                if isinstance(cell, RichTableCell):
                    refs_to_remove.append(cell.ref)

            # Remove the cells from the table
            self.table_cells = self.table_cells[:start_idx] + self.table_cells[end_idx:]

            # Update the number of rows
            self.num_rows -= 1

            # Reassign row offset indices for existing cells
            for index, cell in enumerate(self.table_cells):
                new_index = index // self.num_cols
                cell.start_row_offset_idx = new_index
                cell.end_row_offset_idx = new_index + 1

            all_removed_cells.append(removed_cells)

        if refs_to_remove:
            if doc is None:
                _logger.warning(
                    "When table contains rich cells, `doc` argument must be provided, "
                    "otherwise rich cell content will be left dangling."
                )
            else:
                doc._delete_items(refs_to_remove)

        return all_removed_cells

    def pop_row(self, doc: Optional["DoclingDocument"] = None) -> List[TableCell]:
        """Remove and return the last row from the table.

        :returns: List[TableCell]: A list of TableCell objects representing the popped row.
        """
        if self.num_rows == 0:
            raise IndexError("Cannot pop from an empty table.")

        return self.remove_row(self.num_rows - 1, doc=doc)

    def remove_row(
        self, row_index: int, doc: Optional["DoclingDocument"] = None
    ) -> List[TableCell]:
        """Remove a row from the table by its index.

        :param row_index: int: The index of the row to remove. (Starting from 0)

        :returns: List[TableCell]: A list of TableCell objects representing the removed row.
        """
        return self.remove_rows([row_index], doc=doc)[0]

    def insert_rows(
        self, row_index: int, rows: List[List[str]], after: bool = False
    ) -> None:
        """Insert multiple new rows from a list of lists of strings before/after a specific index in the table.

        :param row_index: int: The index at which to insert the new rows. (Starting from 0)
        :param rows: List[List[str]]: A list of lists, where each inner list represents the content of a new row.
        :param after: bool: If True, insert the rows after the specified index, otherwise before it. (Default is False)

        :returns: None
        """
        effective_rows = rows[::-1]

        for row in effective_rows:
            self.insert_row(row_index, row, after)

    def insert_row(self, row_index: int, row: List[str], after: bool = False) -> None:
        """Insert a new row from a list of strings before/after a specific index in the table.

        :param row_index: int: The index at which to insert the new row. (Starting from 0)
        :param row: List[str]: A list of strings representing the content of the new row.
        :param after: bool: If True, insert the row after the specified index, otherwise before it. (Default is False)

        :returns: None
        """
        if len(row) != self.num_cols:
            raise ValueError(
                f"Row length {len(row)} does not match the number of columns {self.num_cols}."
            )

        effective_index = row_index + (1 if after else 0)

        if effective_index < 0 or effective_index > self.num_rows:
            raise IndexError(
                f"Row index {row_index} is out of bounds for the current number of rows {self.num_rows}."
            )

        new_row_cells = [
            TableCell(
                text=text,
                start_row_offset_idx=effective_index,
                end_row_offset_idx=effective_index + 1,
                start_col_offset_idx=j,
                end_col_offset_idx=j + 1,
            )
            for j, text in enumerate(row)
        ]

        self.table_cells = (
            self.table_cells[: effective_index * self.num_cols]
            + new_row_cells
            + self.table_cells[effective_index * self.num_cols :]
        )

        # Reassign row offset indices for existing cells
        for index, cell in enumerate(self.table_cells):
            new_index = index // self.num_cols
            cell.start_row_offset_idx = new_index
            cell.end_row_offset_idx = new_index + 1

        self.num_rows += 1

    def add_rows(self, rows: List[List[str]]) -> None:
        """Add multiple new rows to the table from a list of lists of strings.

        :param rows: List[List[str]]: A list of lists, where each inner list represents the content of a new row.

        :returns: None
        """
        for row in rows:
            self.add_row(row)

    def add_row(self, row: List[str]) -> None:
        """Add a new row to the table from a list of strings.

        :param row: List[str]: A list of strings representing the content of the new row.

        :returns: None
        """
        self.insert_row(row_index=self.num_rows - 1, row=row, after=True)

    def get_row_bounding_boxes(self) -> dict[int, BoundingBox]:
        """Get the minimal bounding box for each row in the table.

        Returns:
        List[Optional[BoundingBox]]: A list where each element is the minimal
        bounding box that encompasses all cells in that row, or None if no
        cells in the row have bounding boxes.
        """
        coords = []
        for cell in self.table_cells:
            if cell.bbox is not None:
                coords.append(cell.bbox.coord_origin)

        if len(set(coords)) > 1:
            raise ValueError(
                "All bounding boxes must have the same \
                CoordOrigin to compute their union."
            )

        row_bboxes: dict[int, BoundingBox] = {}

        for row_idx in range(self.num_rows):
            row_cells_with_bbox: dict[int, list[BoundingBox]] = {}

            # Collect all cells in this row that have bounding boxes
            for cell in self.table_cells:

                if (
                    cell.bbox is not None
                    and cell.start_row_offset_idx <= row_idx < cell.end_row_offset_idx
                ):

                    row_span = cell.end_row_offset_idx - cell.start_row_offset_idx
                    if row_span in row_cells_with_bbox:
                        row_cells_with_bbox[row_span].append(cell.bbox)
                    else:
                        row_cells_with_bbox[row_span] = [cell.bbox]

            # Calculate the enclosing bounding box for this row
            if len(row_cells_with_bbox) > 0:
                min_row_span = min(row_cells_with_bbox.keys())
                row_bbox: BoundingBox = BoundingBox.enclosing_bbox(
                    row_cells_with_bbox[min_row_span]
                )

                for rspan, bboxs in row_cells_with_bbox.items():
                    for bbox in bboxs:
                        row_bbox.l = min(row_bbox.l, bbox.l)
                        row_bbox.r = max(row_bbox.r, bbox.r)

                row_bboxes[row_idx] = row_bbox

        return row_bboxes

    def get_column_bounding_boxes(self) -> dict[int, BoundingBox]:
        """Get the minimal bounding box for each column in the table.

        Returns:
            List[Optional[BoundingBox]]: A list where each element is the minimal
            bounding box that encompasses all cells in that column, or None if no
            cells in the column have bounding boxes.
        """
        coords = []
        for cell in self.table_cells:
            if cell.bbox is not None:
                coords.append(cell.bbox.coord_origin)

        if len(set(coords)) > 1:
            raise ValueError(
                "All bounding boxes must have the same \
                CoordOrigin to compute their union."
            )

        col_bboxes: dict[int, BoundingBox] = {}

        for col_idx in range(self.num_cols):
            col_cells_with_bbox: dict[int, list[BoundingBox]] = {}

            # Collect all cells in this row that have bounding boxes
            for cell in self.table_cells:

                if (
                    cell.bbox is not None
                    and cell.start_col_offset_idx <= col_idx < cell.end_col_offset_idx
                ):

                    col_span = cell.end_col_offset_idx - cell.start_col_offset_idx
                    if col_span in col_cells_with_bbox:
                        col_cells_with_bbox[col_span].append(cell.bbox)
                    else:
                        col_cells_with_bbox[col_span] = [cell.bbox]

            # Calculate the enclosing bounding box for this row
            if len(col_cells_with_bbox) > 0:
                min_col_span = min(col_cells_with_bbox.keys())
                col_bbox: BoundingBox = BoundingBox.enclosing_bbox(
                    col_cells_with_bbox[min_col_span]
                )

                for rspan, bboxs in col_cells_with_bbox.items():
                    for bbox in bboxs:
                        if bbox.coord_origin == CoordOrigin.TOPLEFT:
                            col_bbox.b = max(col_bbox.b, bbox.b)
                            col_bbox.t = min(col_bbox.t, bbox.t)

                        elif bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                            col_bbox.b = min(col_bbox.b, bbox.b)
                            col_bbox.t = max(col_bbox.t, bbox.t)

                col_bboxes[col_idx] = col_bbox

        return col_bboxes


class PictureTabularChartData(PictureChartData):
    """Base class for picture chart data.

    Attributes:
        title (str): The title of the chart.
        chart_data (TableData): Chart data in the table format.
    """

    kind: Literal["tabular_chart_data"] = "tabular_chart_data"
    chart_data: TableData


PictureDataType = Annotated[
    Union[
        DescriptionAnnotation,
        MiscAnnotation,
        PictureClassificationData,
        PictureMoleculeData,
        PictureTabularChartData,
        PictureLineChartData,
        PictureBarChartData,
        PictureStackedBarChartData,
        PicturePieChartData,
        PictureScatterChartData,
    ],
    Field(discriminator="kind"),
]


class DocumentOrigin(BaseModel):
    """FileSource."""

    mimetype: str  # the mimetype of the original file
    binary_hash: Uint64  # the binary hash of the original file.
    # TODO: Change to be Uint64 and provide utility method to generate

    filename: str  # The name of the original file, including extension, without path.
    # Could stem from filesystem, source URI, Content-Disposition header, ...

    uri: Optional[AnyUrl] = (
        None  # any possible reference to a source file,
        # from any file handler protocol (e.g. https://, file://, s3://)
    )

    _extra_mimetypes: typing.ClassVar[List[str]] = [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/asciidoc",
        "text/markdown",
        "text/csv",
        "audio/x-wav",
        "audio/wav",
        "audio/mp3",
    ]

    @field_validator("binary_hash", mode="before")
    @classmethod
    def parse_hex_string(cls, value):
        """parse_hex_string."""
        if isinstance(value, str):
            try:
                # Convert hex string to an integer
                hash_int = Uint64(value, 16)
                # Mask to fit within 64 bits (unsigned)
                return (
                    hash_int & 0xFFFFFFFFFFFFFFFF
                )  # TODO be sure it doesn't clip uint64 max
            except ValueError:
                raise ValueError(f"Invalid sha256 hexdigest: {value}")
        return value  # If already an int, return it as is.

    @field_validator("mimetype")
    @classmethod
    def validate_mimetype(cls, v):
        """validate_mimetype."""
        # Check if the provided MIME type is valid using mimetypes module
        if v not in mimetypes.types_map.values() and v not in cls._extra_mimetypes:
            raise ValueError(f"'{v}' is not a valid MIME type")
        return v


class RefItem(BaseModel):
    """RefItem."""

    cref: str = Field(alias="$ref", pattern=_JSON_POINTER_REGEX)

    # This method makes RefItem compatible with DocItem
    def get_ref(self):
        """get_ref."""
        return self

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def _split_ref_to_path(self):
        """Get the path of the reference."""
        return self.cref.split("/")

    def resolve(self, doc: "DoclingDocument"):
        """Resolve the path in the document."""
        path_components = self.cref.split("/")
        if (num_comps := len(path_components)) == 3:
            _, path, index_str = path_components
            index = int(index_str)
            obj = doc.__getattribute__(path)[index]
        elif num_comps == 2:
            _, path = path_components
            obj = doc.__getattribute__(path)
        else:
            raise RuntimeError(f"Unsupported number of path components: {num_comps}")
        return obj


class ImageRef(BaseModel):
    """ImageRef."""

    mimetype: str
    dpi: int
    size: Size
    uri: Union[AnyUrl, Path] = Field(union_mode="left_to_right")
    _pil: Optional[PILImage.Image] = None

    @property
    def pil_image(self) -> Optional[PILImage.Image]:
        """Return the PIL Image."""
        if self._pil is not None:
            return self._pil

        if isinstance(self.uri, AnyUrl):
            if self.uri.scheme == "data":
                encoded_img = str(self.uri).split(",")[1]
                decoded_img = base64.b64decode(encoded_img)
                self._pil = PILImage.open(BytesIO(decoded_img))
            elif self.uri.scheme == "file":
                self._pil = PILImage.open(unquote(str(self.uri.path)))
            # else: Handle http request or other protocols...
        elif isinstance(self.uri, Path):
            self._pil = PILImage.open(self.uri)

        return self._pil

    @field_validator("mimetype")
    @classmethod
    def validate_mimetype(cls, v):
        """validate_mimetype."""
        # Check if the provided MIME type is valid using mimetypes module
        if v not in mimetypes.types_map.values():
            raise ValueError(f"'{v}' is not a valid MIME type")
        return v

    @classmethod
    def from_pil(cls, image: PILImage.Image, dpi: int) -> Self:
        """Construct ImageRef from a PIL Image."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_uri = f"data:image/png;base64,{img_str}"
        return cls(
            mimetype="image/png",
            dpi=dpi,
            size=Size(width=image.width, height=image.height),
            uri=img_uri,
            _pil=image,
        )


class DocTagsPage(BaseModel):
    """DocTagsPage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: str
    image: Optional[PILImage.Image] = None


class DocTagsDocument(BaseModel):
    """DocTagsDocument."""

    pages: List[DocTagsPage] = []

    @classmethod
    def from_doctags_and_image_pairs(
        cls,
        doctags: typing.Sequence[Union[Path, str]],
        images: Optional[List[Union[Path, PILImage.Image]]],
    ):
        """from_doctags_and_image_pairs."""
        if images is not None and len(doctags) != len(images):
            raise ValueError("Number of page doctags must be equal to page images!")
        doctags_doc = cls()

        pages = []

        for ix, dt in enumerate(doctags):
            if isinstance(dt, Path):
                with dt.open("r") as fp:
                    dt = fp.read()
            elif isinstance(dt, str):
                pass

            img = None
            if images is not None:
                img = images[ix]

                if isinstance(img, Path):
                    img = PILImage.open(img)
                elif isinstance(img, PILImage.Image):
                    pass

            page = DocTagsPage(tokens=dt, image=img)
            pages.append(page)

        doctags_doc.pages = pages
        return doctags_doc

    @classmethod
    def from_multipage_doctags_and_images(
        cls,
        doctags: Union[Path, str],
        images: Optional[List[Union[Path, PILImage.Image]]],
    ):
        """From doctags with `<page_break>` and corresponding list of page images."""
        if isinstance(doctags, Path):
            with doctags.open("r") as fp:
                doctags = fp.read()
        dt_list = (
            doctags.removeprefix(f"<{DocumentToken.DOCUMENT.value}>")
            .removesuffix(f"</{DocumentToken.DOCUMENT.value}>")
            .split(f"<{DocumentToken.PAGE_BREAK.value}>")
        )
        dt_list = [el.strip() for el in dt_list]

        return cls.from_doctags_and_image_pairs(dt_list, images)


class ProvenanceItem(BaseModel):
    """ProvenanceItem."""

    page_no: int
    bbox: BoundingBox
    charspan: Tuple[int, int]


class ContentLayer(str, Enum):
    """ContentLayer."""

    BODY = "body"  # main content of the document
    FURNITURE = "furniture"  # eg page-headers/footers
    BACKGROUND = "background"  # eg watermarks
    INVISIBLE = "invisible"  # hidden or invisible text
    NOTES = "notes"  # author/speaker notes, corrections, etc


DEFAULT_CONTENT_LAYERS = {ContentLayer.BODY}


class NodeItem(BaseModel):
    """NodeItem."""

    self_ref: str = Field(pattern=_JSON_POINTER_REGEX)
    parent: Optional[RefItem] = None
    children: List[RefItem] = []

    content_layer: ContentLayer = ContentLayer.BODY

    model_config = ConfigDict(extra="forbid")

    def get_ref(self) -> RefItem:
        """get_ref."""
        return RefItem(cref=self.self_ref)

    def _get_parent_ref(
        self, doc: "DoclingDocument", stack: list[int]
    ) -> Optional[RefItem]:
        """get_parent_ref."""
        if len(stack) == 0:
            return self.parent
        elif len(stack) > 0 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._get_parent_ref(doc=doc, stack=stack[1:])

        return None

    def _delete_child(self, doc: "DoclingDocument", stack: list[int]) -> bool:
        """Delete child node in tree."""
        if len(stack) == 1 and stack[0] < len(self.children):
            del self.children[stack[0]]
            return True
        elif len(stack) > 1 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._delete_child(doc=doc, stack=stack[1:])

        return False

    def _update_child(
        self, doc: "DoclingDocument", stack: list[int], new_ref: RefItem
    ) -> bool:
        """Update child node in tree."""
        if len(stack) == 1 and stack[0] < len(self.children):
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children[stack[0]] = new_ref
            return True
        elif len(stack) > 1 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._update_child(doc=doc, stack=stack[1:], new_ref=new_ref)

        return False

    def _add_child(
        self, doc: "DoclingDocument", stack: list[int], new_ref: RefItem
    ) -> bool:
        """Append child to node identified by stack."""
        if len(stack) == 0:

            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children.append(new_ref)
            return True
        elif len(stack) > 0 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._add_child(doc=doc, stack=stack[1:], new_ref=new_ref)

        return False

    def _add_sibling(
        self,
        doc: "DoclingDocument",
        stack: list[int],
        new_ref: RefItem,
        after: bool = True,
    ) -> bool:
        """Add sibling node in tree."""
        if len(stack) == 1 and stack[0] <= len(self.children) and (not after):
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children.insert(stack[0], new_ref)
            return True
        elif len(stack) == 1 and stack[0] < len(self.children) and (after):
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children.insert(stack[0] + 1, new_ref)
            return True
        elif len(stack) > 1 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._add_sibling(
                doc=doc, stack=stack[1:], new_ref=new_ref, after=after
            )

        return False


class GroupItem(NodeItem):  # Container type, can't be a leaf node
    """GroupItem."""

    name: str = (
        "group"  # Name of the group, e.g. "Introduction Chapter",
        # "Slide 5", "Navigation menu list", ...
    )
    # TODO narrow down to allowed values, i.e. excluding those used for subtypes
    label: GroupLabel = GroupLabel.UNSPECIFIED


class ListGroup(GroupItem):
    """ListGroup."""

    label: typing.Literal[GroupLabel.LIST] = GroupLabel.LIST  # type: ignore[assignment]

    @field_validator("label", mode="before")
    @classmethod
    def patch_ordered(cls, value):
        """patch_ordered."""
        return GroupLabel.LIST if value == GroupLabel.ORDERED_LIST else value

    def first_item_is_enumerated(self, doc: "DoclingDocument"):
        """Whether the first list item is enumerated."""
        return (
            len(self.children) > 0
            and isinstance(first_child := self.children[0].resolve(doc), ListItem)
            and first_child.enumerated
        )


@deprecated("Use ListGroup instead.")
class OrderedList(GroupItem):
    """OrderedList."""

    label: typing.Literal[GroupLabel.ORDERED_LIST] = (
        GroupLabel.ORDERED_LIST  # type: ignore[assignment]
    )


class InlineGroup(GroupItem):
    """InlineGroup."""

    label: typing.Literal[GroupLabel.INLINE] = GroupLabel.INLINE


class DocItem(
    NodeItem
):  # Base type for any element that carries content, can be a leaf node
    """DocItem."""

    label: DocItemLabel
    prov: List[ProvenanceItem] = []

    def get_location_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
    ) -> str:
        """Get the location string for the BaseCell."""
        if not len(self.prov):
            return ""

        location = ""
        for prov in self.prov:
            page_w, page_h = doc.pages[prov.page_no].size.as_tuple()

            loc_str = DocumentToken.get_location(
                bbox=prov.bbox.to_top_left_origin(page_h).as_tuple(),
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
            )
            location += loc_str

        return location

    def get_image(
        self, doc: "DoclingDocument", prov_index: int = 0
    ) -> Optional[PILImage.Image]:
        """Returns the image of this DocItem.

        The function returns None if this DocItem has no valid provenance or
        if a valid image of the page containing this DocItem is not available
        in doc.
        """
        if not len(self.prov):
            return None

        page = doc.pages.get(self.prov[prov_index].page_no)
        if page is None or page.size is None or page.image is None:
            return None

        page_image = page.image.pil_image
        if not page_image:
            return None
        crop_bbox = (
            self.prov[prov_index]
            .bbox.to_top_left_origin(page_height=page.size.height)
            .scale_to_size(old_size=page.size, new_size=page.image.size)
            # .scaled(scale=page_image.height / page.size.height)
        )
        return page_image.crop(crop_bbox.as_tuple())

    def get_annotations(self) -> Sequence[BaseAnnotation]:
        """Get the annotations of this DocItem."""
        return []


class Script(str, Enum):
    """Text script position."""

    BASELINE = "baseline"
    SUB = "sub"
    SUPER = "super"


class Formatting(BaseModel):
    """Formatting."""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    script: Script = Script.BASELINE


class TextItem(DocItem):
    """TextItem."""

    label: typing.Literal[
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.REFERENCE,
        DocItemLabel.TEXT,
        DocItemLabel.EMPTY_VALUE,
    ]

    orig: str  # untreated representation
    text: str  # sanitized representation

    formatting: Optional[Formatting] = None
    hyperlink: Optional[Union[AnyUrl, Path]] = Field(
        union_mode="left_to_right", default=None
    )

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class TitleItem(TextItem):
    """TitleItem."""

    label: typing.Literal[DocItemLabel.TITLE] = (
        DocItemLabel.TITLE  # type: ignore[assignment]
    )


class SectionHeaderItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.SECTION_HEADER] = (
        DocItemLabel.SECTION_HEADER  # type: ignore[assignment]
    )
    level: LevelNumber = 1

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class ListItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.LIST_ITEM] = (
        DocItemLabel.LIST_ITEM  # type: ignore[assignment]
    )
    enumerated: bool = False
    marker: str = "-"  # The bullet or number symbol that prefixes this list item


class FloatingItem(DocItem):
    """FloatingItem."""

    captions: List[RefItem] = []
    references: List[RefItem] = []
    footnotes: List[RefItem] = []
    image: Optional[ImageRef] = None

    def caption_text(self, doc: "DoclingDocument") -> str:
        """Computes the caption as a single text."""
        text = ""
        for cap in self.captions:
            text += cap.resolve(doc).text
        return text

    def get_image(
        self, doc: "DoclingDocument", prov_index: int = 0
    ) -> Optional[PILImage.Image]:
        """Returns the image corresponding to this FloatingItem.

        This function returns the PIL image from self.image if one is available.
        Otherwise, it uses DocItem.get_image to get an image of this FloatingItem.

        In particular, when self.image is None, the function returns None if this
        FloatingItem has no valid provenance or the doc does not contain a valid image
        for the required page.
        """
        if self.image is not None:
            return self.image.pil_image
        return super().get_image(doc=doc, prov_index=prov_index)


class CodeItem(FloatingItem, TextItem):
    """CodeItem."""

    label: typing.Literal[DocItemLabel.CODE] = (
        DocItemLabel.CODE  # type: ignore[assignment]
    )
    code_language: CodeLanguageLabel = CodeLanguageLabel.UNKNOWN

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class FormulaItem(TextItem):
    """FormulaItem."""

    label: typing.Literal[DocItemLabel.FORMULA] = (
        DocItemLabel.FORMULA  # type: ignore[assignment]
    )


class PictureItem(FloatingItem):
    """PictureItem."""

    label: typing.Literal[DocItemLabel.PICTURE, DocItemLabel.CHART] = (
        DocItemLabel.PICTURE
    )

    annotations: List[PictureDataType] = []

    # Convert the image to Base64
    def _image_to_base64(self, pil_image, format="PNG"):
        """Base64 representation of the image."""
        buffered = BytesIO()
        pil_image.save(buffered, format=format)  # Save the image to the byte stream
        img_bytes = buffered.getvalue()  # Get the byte data
        img_base64 = base64.b64encode(img_bytes).decode(
            "utf-8"
        )  # Encode to Base64 and decode to string
        return img_base64

    @staticmethod
    def _image_to_hexhash(img: Optional[PILImage.Image]) -> Optional[str]:
        """Hexash from the image."""
        if img is not None:
            # Convert the image to raw bytes
            image_bytes = img.tobytes()

            # Create a hash object (e.g., SHA-256)
            hasher = hashlib.sha256(usedforsecurity=False)

            # Feed the image bytes into the hash object
            hasher.update(image_bytes)

            # Get the hexadecimal representation of the hash
            return hasher.hexdigest()

        return None

    def export_to_markdown(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,  # deprecated
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        image_placeholder: str = "<!-- image -->",
    ) -> str:
        """Export picture to Markdown format."""
        from docling_core.transforms.serializer.markdown import (
            MarkdownDocSerializer,
            MarkdownParams,
        )

        if not add_caption:
            _logger.warning(
                "Argument `add_caption` is deprecated and will be ignored.",
            )

        serializer = MarkdownDocSerializer(
            doc=doc,
            params=MarkdownParams(
                image_mode=image_mode,
                image_placeholder=image_placeholder,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    def export_to_html(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
    ) -> str:
        """Export picture to HTML format."""
        from docling_core.transforms.serializer.html import (
            HTMLDocSerializer,
            HTMLParams,
        )

        serializer = HTMLDocSerializer(
            doc=doc,
            params=HTMLParams(
                image_mode=image_mode,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,  # not used at the moment
    ):
        r"""Export picture to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param # not used at the moment

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
                add_caption=add_caption,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    def get_annotations(self) -> Sequence[BaseAnnotation]:
        """Get the annotations of this PictureItem."""
        return self.annotations


TableAnnotationType = Annotated[
    Union[
        DescriptionAnnotation,
        MiscAnnotation,
    ],
    Field(discriminator="kind"),
]


class TableItem(FloatingItem):
    """TableItem."""

    data: TableData
    label: typing.Literal[
        DocItemLabel.DOCUMENT_INDEX,
        DocItemLabel.TABLE,
    ] = DocItemLabel.TABLE

    annotations: List[TableAnnotationType] = []

    def export_to_dataframe(
        self, doc: Optional["DoclingDocument"] = None
    ) -> pd.DataFrame:
        """Export the table as a Pandas DataFrame."""
        if doc is None:
            _logger.warning(
                "Usage of TableItem.export_to_dataframe() without `doc` argument is deprecated."
            )

        if self.data.num_rows == 0 or self.data.num_cols == 0:
            return pd.DataFrame()

        # Count how many rows are column headers
        num_headers = 0
        for i, row in enumerate(self.data.grid):
            if len(row) == 0:
                raise RuntimeError(
                    f"Invalid table. {len(row)=} but {self.data.num_cols=}."
                )

            any_header = False
            for cell in row:
                if cell.column_header:
                    any_header = True
                    break

            if any_header:
                num_headers += 1
            else:
                break

        # Create the column names from all col_headers
        columns: Optional[List[str]] = None
        if num_headers > 0:
            columns = ["" for _ in range(self.data.num_cols)]
            for i in range(num_headers):
                for j, cell in enumerate(self.data.grid[i]):
                    col_name = cell._get_text(doc=doc)
                    if columns[j] != "":
                        col_name = f".{col_name}"
                    columns[j] += col_name

        # Create table data
        table_data = [
            [cell._get_text(doc=doc) for cell in row]
            for row in self.data.grid[num_headers:]
        ]

        # Create DataFrame
        df = pd.DataFrame(table_data, columns=columns)

        return df

    def export_to_markdown(self, doc: Optional["DoclingDocument"] = None) -> str:
        """Export the table as markdown."""
        if doc is not None:
            from docling_core.transforms.serializer.markdown import (
                MarkdownDocSerializer,
            )

            serializer = MarkdownDocSerializer(doc=doc)
            text = serializer.serialize(item=self).text
            return text
        else:
            _logger.warning(
                "Usage of TableItem.export_to_markdown() without `doc` argument is "
                "deprecated.",
            )

            table = []
            for row in self.data.grid:
                tmp = []
                for col in row:

                    # make sure that md tables are not broken
                    # due to newline chars in the text
                    text = col._get_text(doc=doc)
                    text = text.replace("\n", " ")
                    tmp.append(text)

                table.append(tmp)

            res = ""
            if len(table) > 1 and len(table[0]) > 0:
                try:
                    res = tabulate(table[1:], headers=table[0], tablefmt="github")
                except ValueError:
                    res = tabulate(
                        table[1:],
                        headers=table[0],
                        tablefmt="github",
                        disable_numparse=True,
                    )

        return res

    def export_to_html(
        self,
        doc: Optional["DoclingDocument"] = None,
        add_caption: bool = True,
    ) -> str:
        """Export the table as html."""
        if doc is not None:
            from docling_core.transforms.serializer.html import HTMLDocSerializer

            serializer = HTMLDocSerializer(doc=doc)
            text = serializer.serialize(item=self).text
            return text
        else:
            _logger.error(
                "Usage of TableItem.export_to_html() without `doc` argument is "
                "deprecated.",
            )
            return ""

    def export_to_otsl(
        self,
        doc: "DoclingDocument",
        add_cell_location: bool = True,
        add_cell_text: bool = True,
        xsize: int = 500,
        ysize: int = 500,
        **kwargs: Any,
    ) -> str:
        """Export the table as OTSL."""
        # Possible OTSL tokens...
        #
        # Empty and full cells:
        # "ecel", "fcel"
        #
        # Cell spans (horisontal, vertical, 2d):
        # "lcel", "ucel", "xcel"
        #
        # New line:
        # "nl"
        #
        # Headers (column, row, section row):
        # "ched", "rhed", "srow"

        from docling_core.transforms.serializer.doctags import DocTagsDocSerializer

        doc_serializer = DocTagsDocSerializer(doc=doc)
        body = []
        nrows = self.data.num_rows
        ncols = self.data.num_cols
        if len(self.data.table_cells) == 0:
            return ""

        page_no = 0
        if len(self.prov) > 0:
            page_no = self.prov[0].page_no

        for i in range(nrows):
            for j in range(ncols):
                cell: TableCell = self.data.grid[i][j]
                content = cell._get_text(
                    doc=doc, doc_serializer=doc_serializer, **kwargs
                ).strip()
                rowspan, rowstart = (
                    cell.row_span,
                    cell.start_row_offset_idx,
                )
                colspan, colstart = (
                    cell.col_span,
                    cell.start_col_offset_idx,
                )

                if len(doc.pages.keys()):
                    page_w, page_h = doc.pages[page_no].size.as_tuple()
                cell_loc = ""
                if cell.bbox is not None:
                    cell_loc = DocumentToken.get_location(
                        bbox=cell.bbox.to_bottom_left_origin(page_h).as_tuple(),
                        page_w=page_w,
                        page_h=page_h,
                        xsize=xsize,
                        ysize=ysize,
                    )

                if rowstart == i and colstart == j:
                    if len(content) > 0:
                        if cell.column_header:
                            body.append(str(TableToken.OTSL_CHED.value))
                        elif cell.row_header:
                            body.append(str(TableToken.OTSL_RHED.value))
                        elif cell.row_section:
                            body.append(str(TableToken.OTSL_SROW.value))
                        else:
                            body.append(str(TableToken.OTSL_FCEL.value))
                        if add_cell_location:
                            body.append(str(cell_loc))
                        if add_cell_text:
                            body.append(str(content))
                    else:
                        body.append(str(TableToken.OTSL_ECEL.value))
                else:
                    add_cross_cell = False
                    if rowstart != i:
                        if colspan == 1:
                            body.append(str(TableToken.OTSL_UCEL.value))
                        else:
                            add_cross_cell = True
                    if colstart != j:
                        if rowspan == 1:
                            body.append(str(TableToken.OTSL_LCEL.value))
                        else:
                            add_cross_cell = True
                    if add_cross_cell:
                        body.append(str(TableToken.OTSL_XCEL.value))
            body.append(str(TableToken.OTSL_NL.value))
        body_str = "".join(body)
        return body_str

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_cell_location: bool = True,
        add_cell_text: bool = True,
        add_caption: bool = True,
    ):
        r"""Export table to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_cell_location: bool:  (Default value = True)
        :param add_cell_text: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_caption=add_caption,
                add_table_cell_location=add_cell_location,
                add_table_cell_text=add_cell_text,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    @validate_call
    def add_annotation(self, annotation: TableAnnotationType) -> None:
        """Add an annotation to the table."""
        self.annotations.append(annotation)

    def get_annotations(self) -> Sequence[BaseAnnotation]:
        """Get the annotations of this TableItem."""
        return self.annotations


class GraphCell(BaseModel):
    """GraphCell."""

    label: GraphCellLabel

    cell_id: int

    text: str  # sanitized text
    orig: str  # text as seen on document

    prov: Optional[ProvenanceItem] = None

    # in case you have a text, table or picture item
    item_ref: Optional[RefItem] = None


class GraphLink(BaseModel):
    """GraphLink."""

    label: GraphLinkLabel

    source_cell_id: int
    target_cell_id: int


class GraphData(BaseModel):
    """GraphData."""

    cells: List[GraphCell] = Field(default_factory=list)
    links: List[GraphLink] = Field(default_factory=list)

    @field_validator("links")
    @classmethod
    def validate_links(cls, links, info):
        """Ensure that each link is valid."""
        cells = info.data.get("cells", [])

        valid_cell_ids = {cell.cell_id for cell in cells}

        for link in links:
            if link.source_cell_id not in valid_cell_ids:
                raise ValueError(
                    f"Invalid source_cell_id {link.source_cell_id} in GraphLink"
                )
            if link.target_cell_id not in valid_cell_ids:
                raise ValueError(
                    f"Invalid target_cell_id {link.target_cell_id} in GraphLink"
                )

        return links


class KeyValueItem(FloatingItem):
    """KeyValueItem."""

    label: typing.Literal[DocItemLabel.KEY_VALUE_REGION] = DocItemLabel.KEY_VALUE_REGION

    graph: GraphData

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export key value item to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class FormItem(FloatingItem):
    """FormItem."""

    label: typing.Literal[DocItemLabel.FORM] = DocItemLabel.FORM

    graph: GraphData


ContentItem = Annotated[
    Union[
        TextItem,
        TitleItem,
        SectionHeaderItem,
        ListItem,
        CodeItem,
        FormulaItem,
        PictureItem,
        TableItem,
        KeyValueItem,
    ],
    Field(discriminator="label"),
]


class PageItem(BaseModel):
    """PageItem."""

    # A page carries separate root items for furniture and body,
    # only referencing items on the page
    size: Size
    image: Optional[ImageRef] = None
    page_no: int


class DoclingDocument(BaseModel):
    """DoclingDocument."""

    schema_name: typing.Literal["DoclingDocument"] = "DoclingDocument"
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        CURRENT_VERSION
    )
    name: str  # The working name of this document, without extensions
    # (could be taken from originating doc, or just "Untitled 1")
    origin: Optional[DocumentOrigin] = (
        None  # DoclingDocuments may specify an origin (converted to DoclingDocument).
        # This is optional, e.g. a DoclingDocument could also be entirely
        # generated from synthetic data.
    )

    furniture: Annotated[GroupItem, Field(deprecated=True)] = GroupItem(
        name="_root_",
        self_ref="#/furniture",
        content_layer=ContentLayer.FURNITURE,
    )  # List[RefItem] = []
    body: GroupItem = GroupItem(name="_root_", self_ref="#/body")  # List[RefItem] = []

    groups: List[Union[ListGroup, InlineGroup, GroupItem]] = []
    texts: List[
        Union[TitleItem, SectionHeaderItem, ListItem, CodeItem, FormulaItem, TextItem]
    ] = []
    pictures: List[PictureItem] = []
    tables: List[TableItem] = []
    key_value_items: List[KeyValueItem] = []
    form_items: List[FormItem] = []

    pages: Dict[int, PageItem] = {}  # empty as default

    @model_validator(mode="before")
    @classmethod
    def transform_to_content_layer(cls, data: dict) -> dict:
        """transform_to_content_layer."""
        # Since version 1.1.0, all NodeItems carry content_layer property.
        # We must assign previous page_header and page_footer instances to furniture.
        # Note: model_validators which check on the version must use "before".
        if "version" in data and data["version"] == "1.0.0":
            for item in data.get("texts", []):
                if "label" in item and item["label"] in [
                    DocItemLabel.PAGE_HEADER.value,
                    DocItemLabel.PAGE_FOOTER.value,
                ]:
                    item["content_layer"] = "furniture"
        return data

    # ---------------------------
    # Public Manipulation methods
    # ---------------------------

    def append_child_item(
        self, *, child: NodeItem, parent: Optional[NodeItem] = None
    ) -> None:
        """Adds an item."""
        if len(child.children) > 0:
            raise ValueError("Can not append a child with children")

        parent = parent if parent is not None else self.body

        success, stack = self._get_stack_of_item(item=parent)

        if not success:
            raise ValueError(
                f"Could not resolve the parent node in the document tree: {parent}"
            )

        # Append the item to the attributes of the doc
        self._append_item(item=child, parent_ref=parent.get_ref())

        # Update the tree of the doc
        success = self.body._add_child(doc=self, new_ref=child.get_ref(), stack=stack)

        # Clean the attribute (orphan) if not successful
        if not success:
            self._pop_item(item=child)
            raise ValueError(f"Could not append child: {child} to parent: {parent}")

    def insert_item_after_sibling(
        self, *, new_item: NodeItem, sibling: NodeItem
    ) -> None:
        """Inserts an item, given its node_item instance, after other as a sibling."""
        self._insert_item_at_refitem(item=new_item, ref=sibling.get_ref(), after=True)

    def insert_item_before_sibling(
        self, *, new_item: NodeItem, sibling: NodeItem
    ) -> None:
        """Inserts an item, given its node_item instance, before other as a sibling."""
        self._insert_item_at_refitem(item=new_item, ref=sibling.get_ref(), after=False)

    def delete_items(self, *, node_items: List[NodeItem]) -> None:
        """Deletes an item, given its instance or ref, and any children it has."""
        refs = []
        for _ in node_items:
            refs.append(_.get_ref())

        self._delete_items(refs=refs)

    def replace_item(self, *, new_item: NodeItem, old_item: NodeItem) -> None:
        """Replace item with new item."""
        self.insert_item_after_sibling(new_item=new_item, sibling=old_item)
        self.delete_items(node_items=[old_item])

    # ----------------------------
    # Private Manipulation methods
    # ----------------------------

    def _get_stack_of_item(self, item: NodeItem) -> tuple[bool, list[int]]:
        """Find the stack indices of the item."""
        return self._get_stack_of_refitem(ref=item.get_ref())

    def _get_stack_of_refitem(self, ref: RefItem) -> tuple[bool, list[int]]:
        """Find the stack indices of the reference."""
        if ref == self.body.get_ref():
            return (True, [])

        node = ref.resolve(doc=self)
        parent_ref = node._get_parent_ref(doc=self, stack=[])

        if parent_ref is None:
            return (False, [])

        stack: list[int] = []
        while parent_ref is not None:
            parent = parent_ref.resolve(doc=self)

            index = parent.children.index(node.get_ref())
            stack.insert(0, index)  # prepend the index

            node = parent
            parent_ref = node._get_parent_ref(doc=self, stack=[])

        return (True, stack)

    def _insert_item_at_refitem(
        self, item: NodeItem, ref: RefItem, after: bool
    ) -> RefItem:
        """Insert node-item using the self-reference."""
        success, stack = self._get_stack_of_refitem(ref=ref)

        if not success:
            raise ValueError(
                f"Could not insert at {ref.cref}: could not find the stack"
            )

        return self._insert_item_at_stack(item=item, stack=stack, after=after)

    def _append_item(self, *, item: NodeItem, parent_ref: RefItem) -> RefItem:
        """Append item of its type."""
        cref: str = ""  # to be updated

        if isinstance(item, TextItem):
            item_label = "texts"
            item_index = len(self.texts)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.texts.append(item)

        elif isinstance(item, TableItem):
            item_label = "tables"
            item_index = len(self.tables)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.tables.append(item)

        elif isinstance(item, PictureItem):
            item_label = "pictures"
            item_index = len(self.pictures)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.pictures.append(item)

        elif isinstance(item, KeyValueItem):
            item_label = "key_value_items"
            item_index = len(self.key_value_items)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.key_value_items.append(item)

        elif isinstance(item, FormItem):
            item_label = "form_items"
            item_index = len(self.form_items)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.form_items.append(item)

        elif isinstance(item, (ListGroup, InlineGroup)):
            item_label = "groups"
            item_index = len(self.groups)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.groups.append(item)
        elif isinstance(item, GroupItem):
            item_label = "groups"
            item_index = len(self.groups)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.groups.append(item)

        else:
            raise ValueError(f"Item {item} is not supported for insertion")

        return RefItem(cref=cref)

    def _pop_item(self, *, item: NodeItem):
        """Pop the last item of its type."""
        path = item.self_ref.split("/")

        if len(path) != 3:
            raise ValueError(f"Can not pop item with path: {path}")

        item_label = path[1]
        item_index = int(path[2])

        if (
            len(self.__getattribute__(item_label)) == item_index + 1
        ):  # we can only pop the last item
            del self.__getattribute__(item_label)[item_index]
        else:
            msg = f"index:{item_index}, len:{len(self.__getattribute__(item_label))}"
            raise ValueError(f"Failed to pop: item is not last ({msg})")

    def _insert_item_at_stack(
        self, item: NodeItem, stack: list[int], after: bool
    ) -> RefItem:
        """Insert node-item using the self-reference."""
        parent_ref = self.body._get_parent_ref(doc=self, stack=stack)

        if parent_ref is None:
            raise ValueError(f"Could not find a parent at stack: {stack}")

        new_ref = self._append_item(item=item, parent_ref=parent_ref)

        success = self.body._add_sibling(
            doc=self, stack=stack, new_ref=new_ref, after=after
        )

        if not success:
            self._pop_item(item=item)

            raise ValueError(
                f"Could not insert item: {item} under parent: {parent_ref.resolve(doc=self)}"
            )

        return item.get_ref()

    def _delete_items(self, refs: list[RefItem]):
        """Delete document item using the self-reference."""
        to_be_deleted_items: dict[tuple[int, ...], str] = {}  # stack to cref

        if not refs:
            return

        # Identify the to_be_deleted_items
        for item, stack in self._iterate_items_with_stack(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers={c for c in ContentLayer},
        ):
            ref = item.get_ref()

            if ref in refs:
                to_be_deleted_items[tuple(stack)] = ref.cref

            substacks = [stack[0 : i + 1] for i in range(len(stack) - 1)]
            for substack in substacks:
                if tuple(substack) in to_be_deleted_items:
                    to_be_deleted_items[tuple(stack)] = ref.cref

        if len(to_be_deleted_items) < len(refs):
            raise ValueError(
                f"Cannot find all provided RefItems in doc: {[r.cref for r in refs]}"
            )

        # Clean the tree, reverse the order to not have to update
        for stack_, ref_ in reversed(sorted(to_be_deleted_items.items())):
            success = self.body._delete_child(doc=self, stack=list(stack_))

            if not success:
                del to_be_deleted_items[stack_]
            else:
                _logger.info(f"deleted item in tree at stack: {stack_} => {ref_}")

        # Create a new lookup of the orphans:
        # dict of item_label (`texts`, `tables`, ...) to a
        # dict of item_label with delta (default = -1).
        lookup: dict[str, dict[int, int]] = {}

        for stack_, ref_ in to_be_deleted_items.items():
            path = ref_.split("/")
            if len(path) == 3:

                item_label = path[1]
                item_index = int(path[2])

                if item_label not in lookup:
                    lookup[item_label] = {}

                lookup[item_label][item_index] = -1

        # Remove the orphans in reverse order
        for item_label, item_inds in lookup.items():
            for item_index, val in reversed(
                sorted(item_inds.items())
            ):  # make sure you delete the last in the list first!
                _logger.debug(f"deleting item in doc for {item_label} for {item_index}")
                del self.__getattribute__(item_label)[item_index]

        self._update_breadth_first_with_lookup(
            node=self.body, refs_to_be_deleted=refs, lookup=lookup
        )

    # Update the references
    def _update_ref_with_lookup(
        self, item_label: str, item_index: int, lookup: dict[str, dict[int, int]]
    ) -> RefItem:
        """Update ref with lookup."""
        if item_label not in lookup:  # Nothing to be done
            return RefItem(cref=f"#/{item_label}/{item_index}")

        # Count how many items have been deleted in front of you
        delta = sum(
            val if item_index >= key else 0 for key, val in lookup[item_label].items()
        )
        new_index = item_index + delta

        return RefItem(cref=f"#/{item_label}/{new_index}")

    def _update_refitems_with_lookup(
        self,
        ref_items: list[RefItem],
        refs_to_be_deleted: list[RefItem],
        lookup: dict[str, dict[int, int]],
    ) -> list[RefItem]:
        """Update refitems with lookup."""
        new_refitems = []
        for ref_item in ref_items:

            if (
                ref_item not in refs_to_be_deleted
            ):  # if ref_item is in ref, then delete/skip them
                path = ref_item._split_ref_to_path()
                if len(path) == 3:
                    new_refitems.append(
                        self._update_ref_with_lookup(
                            item_label=path[1],
                            item_index=int(path[2]),
                            lookup=lookup,
                        )
                    )
                else:
                    new_refitems.append(ref_item)

        return new_refitems

    def _update_breadth_first_with_lookup(
        self,
        node: NodeItem,
        refs_to_be_deleted: list[RefItem],
        lookup: dict[str, dict[int, int]],
    ):
        """Update breadth first with lookup."""
        # Update the captions, references and footnote references
        if isinstance(node, FloatingItem):
            node.captions = self._update_refitems_with_lookup(
                ref_items=node.captions,
                refs_to_be_deleted=refs_to_be_deleted,
                lookup=lookup,
            )
            node.references = self._update_refitems_with_lookup(
                ref_items=node.references,
                refs_to_be_deleted=refs_to_be_deleted,
                lookup=lookup,
            )
            node.footnotes = self._update_refitems_with_lookup(
                ref_items=node.footnotes,
                refs_to_be_deleted=refs_to_be_deleted,
                lookup=lookup,
            )
            if isinstance(node, TableItem):
                for cell in node.data.table_cells:
                    if isinstance(cell, RichTableCell):
                        path = cell.ref._split_ref_to_path()
                        cell.ref = self._update_ref_with_lookup(
                            item_label=path[1],
                            item_index=int(path[2]),
                            lookup=lookup,
                        )

        # Update the self_ref reference
        if node.parent is not None:
            path = node.parent._split_ref_to_path()
            if len(path) == 3:
                node.parent = self._update_ref_with_lookup(
                    item_label=path[1], item_index=int(path[2]), lookup=lookup
                )

        # Update the parent reference
        if node.self_ref is not None:
            path = node.self_ref.split("/")
            if len(path) == 3:
                _ref = self._update_ref_with_lookup(
                    item_label=path[1], item_index=int(path[2]), lookup=lookup
                )
                node.self_ref = _ref.cref

        # Update the child references
        node.children = self._update_refitems_with_lookup(
            ref_items=node.children,
            refs_to_be_deleted=refs_to_be_deleted,
            lookup=lookup,
        )

        for i, child_ref in enumerate(node.children):
            node = child_ref.resolve(self)
            self._update_breadth_first_with_lookup(
                node=node, refs_to_be_deleted=refs_to_be_deleted, lookup=lookup
            )

    ###################################
    # TODO: refactor add* methods below
    ###################################

    def add_list_group(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> ListGroup:
        """add_list_group."""
        _parent = parent or self.body
        cref = f"#/groups/{len(self.groups)}"
        group = ListGroup(self_ref=cref, parent=_parent.get_ref())
        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self.groups.append(group)
        _parent.children.append(RefItem(cref=cref))
        return group

    @deprecated("Use add_list_group() instead.")
    def add_ordered_list(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> GroupItem:
        """add_ordered_list."""
        return self.add_list_group(
            name=name,
            parent=parent,
            content_layer=content_layer,
        )

    @deprecated("Use add_list_group() instead.")
    def add_unordered_list(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> GroupItem:
        """add_unordered_list."""
        return self.add_list_group(
            name=name,
            parent=parent,
            content_layer=content_layer,
        )

    def add_inline_group(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> InlineGroup:
        """add_inline_group."""
        _parent = parent or self.body
        cref = f"#/groups/{len(self.groups)}"
        group = InlineGroup(self_ref=cref, parent=_parent.get_ref())
        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self.groups.append(group)
        _parent.children.append(RefItem(cref=cref))
        return group

    def add_group(
        self,
        label: Optional[GroupLabel] = None,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> GroupItem:
        """add_group.

        :param label: Optional[GroupLabel]:  (Default value = None)
        :param name: Optional[str]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if label in [GroupLabel.LIST, GroupLabel.ORDERED_LIST]:
            return self.add_list_group(
                name=name,
                parent=parent,
                content_layer=content_layer,
            )
        elif label == GroupLabel.INLINE:
            return self.add_inline_group(
                name=name,
                parent=parent,
                content_layer=content_layer,
            )

        if not parent:
            parent = self.body

        group_index = len(self.groups)
        cref = f"#/groups/{group_index}"

        group = GroupItem(self_ref=cref, parent=parent.get_ref())
        if name is not None:
            group.name = name
        if label is not None:
            group.label = label
        if content_layer:
            group.content_layer = content_layer

        self.groups.append(group)
        parent.children.append(RefItem(cref=cref))

        return group

    def add_list_item(
        self,
        text: str,
        enumerated: bool = False,
        marker: Optional[str] = None,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_list_item.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if not isinstance(parent, ListGroup):
            warnings.warn(
                "ListItem parent must be a list group, creating one on the fly.",
                DeprecationWarning,
            )
            parent = self.add_list_group(parent=parent)

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            enumerated=enumerated,
            marker=marker or "",
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            list_item.prov.append(prov)
        if content_layer:
            list_item.content_layer = content_layer

        self.texts.append(list_item)
        parent.children.append(RefItem(cref=cref))

        return list_item

    def add_text(
        self,
        label: DocItemLabel,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_text.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        # Catch a few cases that are in principle allowed
        # but that will create confusion down the road
        if label in [DocItemLabel.TITLE]:
            return self.add_title(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif label in [DocItemLabel.LIST_ITEM]:
            return self.add_list_item(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif label in [DocItemLabel.SECTION_HEADER]:
            return self.add_heading(
                text=text,
                orig=orig,
                # NOTE: we do not / cannot pass the level here, lossy path..
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif label in [DocItemLabel.CODE]:
            return self.add_code(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )
        elif label in [DocItemLabel.FORMULA]:
            return self.add_formula(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        else:

            if not parent:
                parent = self.body

            if not orig:
                orig = text

            text_index = len(self.texts)
            cref = f"#/texts/{text_index}"
            text_item = TextItem(
                label=label,
                text=text,
                orig=orig,
                self_ref=cref,
                parent=parent.get_ref(),
                formatting=formatting,
                hyperlink=hyperlink,
            )
            if prov:
                text_item.prov.append(prov)

            if content_layer:
                text_item.content_layer = content_layer

            self.texts.append(text_item)
            parent.children.append(RefItem(cref=cref))

            return text_item

    def add_table(
        self,
        data: TableData,
        caption: Optional[Union[TextItem, RefItem]] = None,  # This is not cool yet.
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        label: DocItemLabel = DocItemLabel.TABLE,
        content_layer: Optional[ContentLayer] = None,
        annotations: Optional[list[TableAnnotationType]] = None,
    ):
        """add_table.

        :param data: TableData:
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param label: DocItemLabel:  (Default value = DocItemLabel.TABLE)

        """
        if not parent:
            parent = self.body

        table_index = len(self.tables)
        cref = f"#/tables/{table_index}"

        tbl_item = TableItem(
            label=label,
            data=data,
            self_ref=cref,
            parent=parent.get_ref(),
            annotations=annotations or [],
        )
        if prov:
            tbl_item.prov.append(prov)
        if content_layer:
            tbl_item.content_layer = content_layer

        if caption:
            tbl_item.captions.append(caption.get_ref())

        self.tables.append(tbl_item)
        parent.children.append(RefItem(cref=cref))

        return tbl_item

    def add_picture(
        self,
        annotations: Optional[List[PictureDataType]] = None,
        image: Optional[ImageRef] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ):
        """add_picture.

        :param data: Optional[List[PictureData]]: (Default value = None)
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        picture_index = len(self.pictures)
        cref = f"#/pictures/{picture_index}"

        fig_item = PictureItem(
            label=DocItemLabel.PICTURE,
            annotations=annotations or [],
            image=image,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            fig_item.prov.append(prov)
        if content_layer:
            fig_item.content_layer = content_layer
        if caption:
            fig_item.captions.append(caption.get_ref())

        self.pictures.append(fig_item)
        parent.children.append(RefItem(cref=cref))

        return fig_item

    def add_title(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_title.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        item = TitleItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            item.prov.append(prov)
        if content_layer:
            item.content_layer = content_layer

        self.texts.append(item)
        parent.children.append(RefItem(cref=cref))

        return item

    def add_code(
        self,
        text: str,
        code_language: Optional[CodeLanguageLabel] = None,
        orig: Optional[str] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_code.

        :param text: str:
        :param code_language: Optional[str]: (Default value = None)
        :param orig: Optional[str]:  (Default value = None)
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        code_item = CodeItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if code_language:
            code_item.code_language = code_language
        if content_layer:
            code_item.content_layer = content_layer
        if prov:
            code_item.prov.append(prov)
        if caption:
            code_item.captions.append(caption.get_ref())

        self.texts.append(code_item)
        parent.children.append(RefItem(cref=cref))

        return code_item

    def add_formula(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_formula.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        section_header_item = FormulaItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            section_header_item.prov.append(prov)
        if content_layer:
            section_header_item.content_layer = content_layer

        self.texts.append(section_header_item)
        parent.children.append(RefItem(cref=cref))

        return section_header_item

    def add_heading(
        self,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_heading.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        section_header_item = SectionHeaderItem(
            level=level,
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            section_header_item.prov.append(prov)
        if content_layer:
            section_header_item.content_layer = content_layer

        self.texts.append(section_header_item)
        parent.children.append(RefItem(cref=cref))

        return section_header_item

    def add_key_values(
        self,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_key_values.

        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        key_value_index = len(self.key_value_items)
        cref = f"#/key_value_items/{key_value_index}"

        kv_item = KeyValueItem(
            graph=graph,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            kv_item.prov.append(prov)

        self.key_value_items.append(kv_item)
        parent.children.append(RefItem(cref=cref))

        return kv_item

    def add_form(
        self,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_form.

        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        form_index = len(self.form_items)
        cref = f"#/form_items/{form_index}"

        form_item = FormItem(
            graph=graph,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            form_item.prov.append(prov)

        self.form_items.append(form_item)
        parent.children.append(RefItem(cref=cref))

        return form_item

    # ---------------------------
    # Node Item Insertion Methods
    # ---------------------------

    def _get_insertion_stack_and_parent(
        self, sibling: NodeItem
    ) -> tuple[list[int], RefItem]:
        """Get the stack and parent reference for inserting a new item at a sibling."""
        # Get the stack of the sibling
        sibling_ref = sibling.get_ref()

        success, stack = self._get_stack_of_refitem(ref=sibling_ref)

        if not success:
            raise ValueError(
                f"Could not insert at {sibling_ref.cref}: could not find the stack"
            )

        # Get the parent RefItem
        parent_ref = self.body._get_parent_ref(doc=self, stack=stack)

        if parent_ref is None:
            raise ValueError(f"Could not find a parent at stack: {stack}")

        return stack, parent_ref

    def _insert_in_structure(
        self,
        item: NodeItem,
        stack: list[int],
        after: bool,
        created_parent: Optional[bool] = False,
    ) -> None:
        """Insert item into the document structure at the specified stack and handle errors."""
        # Ensure the item has a parent reference
        if item.parent is None:
            item.parent = self.body.get_ref()

        self._append_item(item=item, parent_ref=item.parent)

        new_ref = item.get_ref()

        success = self.body._add_sibling(
            doc=self, stack=stack, new_ref=new_ref, after=after
        )

        # Error handling can be determined here
        if not success:
            self._pop_item(item=item)

            if created_parent:
                self.delete_items(node_items=[item.parent.resolve(self)])

            raise ValueError(
                f"Could not insert item: {item} under parent: {item.parent.resolve(doc=self)}"
            )

    def insert_list_group(
        self,
        sibling: NodeItem,
        name: Optional[str] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> ListGroup:
        """Creates a new ListGroup item and inserts it into the document.

        :param sibling: NodeItem:
        :param name: Optional[str]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: ListGroup: The newly created ListGroup item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        group = ListGroup(self_ref="#", parent=parent_ref)

        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self._insert_in_structure(item=group, stack=stack, after=after)

        return group

    def insert_inline_group(
        self,
        sibling: NodeItem,
        name: Optional[str] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> InlineGroup:
        """Creates a new InlineGroup item and inserts it into the document.

        :param sibling: NodeItem:
        :param name: Optional[str]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: InlineGroup: The newly created InlineGroup item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new InlineGroup NodeItem
        group = InlineGroup(self_ref="#", parent=parent_ref)

        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self._insert_in_structure(item=group, stack=stack, after=after)

        return group

    def insert_group(
        self,
        sibling: NodeItem,
        label: Optional[GroupLabel] = None,
        name: Optional[str] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> GroupItem:
        """Creates a new GroupItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param label: Optional[GroupLabel]:  (Default value = None)
        :param name: Optional[str]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: GroupItem: The newly created GroupItem.
        """
        if label in [GroupLabel.LIST, GroupLabel.ORDERED_LIST]:
            return self.insert_list_group(
                sibling=sibling,
                name=name,
                content_layer=content_layer,
                after=after,
            )
        elif label == GroupLabel.INLINE:
            return self.insert_inline_group(
                sibling=sibling,
                name=name,
                content_layer=content_layer,
                after=after,
            )

        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new GroupItem NodeItem
        group = GroupItem(self_ref="#", parent=parent_ref)

        if name is not None:
            group.name = name
        if label is not None:
            group.label = label
        if content_layer:
            group.content_layer = content_layer

        self._insert_in_structure(item=group, stack=stack, after=after)

        return group

    def insert_list_item(
        self,
        sibling: NodeItem,
        text: str,
        enumerated: bool = False,
        marker: Optional[str] = None,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> ListItem:
        """Creates a new ListItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param enumerated: bool:  (Default value = False)
        :param marker: Optional[str]:  (Default value = None)
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: ListItem: The newly created ListItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Ensure the parent is a ListGroup

        parent = parent_ref.resolve(self)
        set_parent = False

        if not isinstance(parent, ListGroup):
            warnings.warn(
                "ListItem parent must be a ListGroup, creating one on the fly.",
                DeprecationWarning,
            )
            parent = self.insert_list_group(sibling=sibling, after=after)
            parent_ref = parent.get_ref()
            if after:
                stack[-1] += 1
            stack.append(0)
            after = False
            set_parent = True

        # Create a new ListItem NodeItem
        if not orig:
            orig = text

        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            enumerated=enumerated,
            marker=marker or "",
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            list_item.prov.append(prov)
        if content_layer:
            list_item.content_layer = content_layer

        self._insert_in_structure(
            item=list_item, stack=stack, after=after, created_parent=set_parent
        )

        return list_item

    def insert_text(
        self,
        sibling: NodeItem,
        label: DocItemLabel,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> TextItem:
        """Creates a new TextItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: TextItem: The newly created TextItem item.
        """
        if label in [DocItemLabel.TITLE]:
            return self.insert_title(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.LIST_ITEM]:
            return self.insert_list_item(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.SECTION_HEADER]:
            return self.insert_heading(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.CODE]:
            return self.insert_code(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.FORMULA]:
            return self.insert_formula(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        else:
            # Get stack and parent reference of the sibling
            stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

            # Create a new TextItem NodeItem
            if not orig:
                orig = text

            text_item = TextItem(
                label=label,
                text=text,
                orig=orig,
                self_ref="#",
                parent=parent_ref,
                formatting=formatting,
                hyperlink=hyperlink,
            )

            if prov:
                text_item.prov.append(prov)
            if content_layer:
                text_item.content_layer = content_layer

            self._insert_in_structure(item=text_item, stack=stack, after=after)

            return text_item

    def insert_table(
        self,
        sibling: NodeItem,
        data: TableData,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        label: DocItemLabel = DocItemLabel.TABLE,
        content_layer: Optional[ContentLayer] = None,
        annotations: Optional[list[TableAnnotationType]] = None,
        after: bool = True,
    ) -> TableItem:
        """Creates a new TableItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param data: TableData:
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param label: DocItemLabel:  (Default value = DocItemLabel.TABLE)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param annotations: Optional[List[TableAnnotationType]]: (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: TableItem: The newly created TableItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new ListItem NodeItem
        table_item = TableItem(
            label=label,
            data=data,
            self_ref="#",
            parent=parent_ref,
            annotations=annotations or [],
        )

        if prov:
            table_item.prov.append(prov)
        if content_layer:
            table_item.content_layer = content_layer
        if caption:
            table_item.captions.append(caption.get_ref())

        self._insert_in_structure(item=table_item, stack=stack, after=after)

        return table_item

    def insert_picture(
        self,
        sibling: NodeItem,
        annotations: Optional[List[PictureDataType]] = None,
        image: Optional[ImageRef] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> PictureItem:
        """Creates a new PictureItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param annotations: Optional[List[PictureDataType]]: (Default value = None)
        :param image: Optional[ImageRef]:  (Default value = None)
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: PictureItem: The newly created PictureItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new PictureItem NodeItem
        picture_item = PictureItem(
            label=DocItemLabel.PICTURE,
            annotations=annotations or [],
            image=image,
            self_ref="#",
            parent=parent_ref,
        )

        if prov:
            picture_item.prov.append(prov)
        if content_layer:
            picture_item.content_layer = content_layer
        if caption:
            picture_item.captions.append(caption.get_ref())

        self._insert_in_structure(item=picture_item, stack=stack, after=after)

        return picture_item

    def insert_title(
        self,
        sibling: NodeItem,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> TitleItem:
        """Creates a new TitleItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: TitleItem: The newly created TitleItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new TitleItem NodeItem
        if not orig:
            orig = text

        title_item = TitleItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            title_item.prov.append(prov)
        if content_layer:
            title_item.content_layer = content_layer

        self._insert_in_structure(item=title_item, stack=stack, after=after)

        return title_item

    def insert_code(
        self,
        sibling: NodeItem,
        text: str,
        code_language: Optional[CodeLanguageLabel] = None,
        orig: Optional[str] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> CodeItem:
        """Creates a new CodeItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param code_language: Optional[str]: (Default value = None)
        :param orig: Optional[str]:  (Default value = None)
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: CodeItem: The newly created CodeItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new CodeItem NodeItem
        if not orig:
            orig = text

        code_item = CodeItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if code_language:
            code_item.code_language = code_language
        if content_layer:
            code_item.content_layer = content_layer
        if prov:
            code_item.prov.append(prov)
        if caption:
            code_item.captions.append(caption.get_ref())

        self._insert_in_structure(item=code_item, stack=stack, after=after)

        return code_item

    def insert_formula(
        self,
        sibling: NodeItem,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> FormulaItem:
        """Creates a new FormulaItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: FormulaItem: The newly created FormulaItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new FormulaItem NodeItem
        if not orig:
            orig = text

        formula_item = FormulaItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            formula_item.prov.append(prov)
        if content_layer:
            formula_item.content_layer = content_layer

        self._insert_in_structure(item=formula_item, stack=stack, after=after)

        return formula_item

    def insert_heading(
        self,
        sibling: NodeItem,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> SectionHeaderItem:
        """Creates a new SectionHeaderItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: SectionHeaderItem: The newly created SectionHeaderItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new SectionHeaderItem NodeItem
        if not orig:
            orig = text

        section_header_item = SectionHeaderItem(
            level=level,
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            section_header_item.prov.append(prov)
        if content_layer:
            section_header_item.content_layer = content_layer

        self._insert_in_structure(item=section_header_item, stack=stack, after=after)

        return section_header_item

    def insert_key_values(
        self,
        sibling: NodeItem,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        after: bool = True,
    ) -> KeyValueItem:
        """Creates a new KeyValueItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: KeyValueItem: The newly created KeyValueItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new KeyValueItem NodeItem
        key_value_item = KeyValueItem(graph=graph, self_ref="#", parent=parent_ref)

        if prov:
            key_value_item.prov.append(prov)

        self._insert_in_structure(item=key_value_item, stack=stack, after=after)

        return key_value_item

    def insert_form(
        self,
        sibling: NodeItem,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        after: bool = True,
    ) -> FormItem:
        """Creates a new FormItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: FormItem: The newly created FormItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new FormItem NodeItem
        form_item = FormItem(graph=graph, self_ref="#", parent=parent_ref)

        if prov:
            form_item.prov.append(prov)

        self._insert_in_structure(item=form_item, stack=stack, after=after)

        return form_item

    # ---------------------------
    # Range Manipulation Methods
    # ---------------------------

    def delete_items_range(
        self,
        *,
        start: NodeItem,
        end: NodeItem,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
    ) -> None:
        """Deletes all NodeItems and their children in the range from the start NodeItem to the end NodeItem.

        :param start: NodeItem:  The starting NodeItem of the range
        :param end: NodeItem:  The ending NodeItem of the range
        :param start_inclusive: bool:  (Default value = True):  If True, the start NodeItem will also be deleted
        :param end_inclusive: bool:  (Default value = True):  If True, the end NodeItem will also be deleted

        :returns: None
        """
        start_parent_ref = (
            start.parent if start.parent is not None else self.body.get_ref()
        )
        end_parent_ref = end.parent if end.parent is not None else self.body.get_ref()

        if start.parent != end.parent:
            raise ValueError(
                "Start and end NodeItems must have the same parent to delete a range."
            )

        start_ref = start.get_ref()
        end_ref = end.get_ref()

        start_parent = start_parent_ref.resolve(doc=self)
        end_parent = end_parent_ref.resolve(doc=self)

        start_index = start_parent.children.index(start_ref)
        end_index = end_parent.children.index(end_ref)

        if start_index > end_index:
            raise ValueError(
                "Start NodeItem must come before or be the same as the end NodeItem in the document structure."
            )

        to_delete = start_parent.children[start_index : end_index + 1]

        if not start_inclusive:
            to_delete = to_delete[1:]
        if not end_inclusive:
            to_delete = to_delete[:-1]

        self._delete_items(refs=to_delete)

    def extract_items_range(
        self,
        *,
        start: NodeItem,
        end: NodeItem,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
        delete: bool = False,
    ) -> "DoclingDocument":
        """Extracts NodeItems and children in the range from the start NodeItem to the end as a new DoclingDocument.

        :param start: NodeItem:  The starting NodeItem of the range (must be a direct child of the document body)
        :param end: NodeItem:  The ending NodeItem of the range  (must be a direct child of the document body)
        :param start_inclusive: bool:  (Default value = True):  If True, the start NodeItem will also be extracted
        :param end_inclusive: bool:  (Default value = True):  If True, the end NodeItem will also be extracted
        :param delete: bool:  (Default value = False):  If True, extracted items are deleted in the original document

        :returns: DoclingDocument: A new document containing the extracted NodeItems and their children
        """
        if not start.parent == end.parent:
            raise ValueError(
                "Start and end NodeItems must have the same parent to extract a range."
            )

        start_ref = start.get_ref()
        end_ref = end.get_ref()

        start_parent_ref = (
            start.parent if start.parent is not None else self.body.get_ref()
        )
        end_parent_ref = end.parent if end.parent is not None else self.body.get_ref()

        start_parent = start_parent_ref.resolve(doc=self)
        end_parent = end_parent_ref.resolve(doc=self)

        start_index = start_parent.children.index(start_ref) + (
            0 if start_inclusive else 1
        )
        end_index = end_parent.children.index(end_ref) + (1 if end_inclusive else 0)

        if start_index > end_index:
            raise ValueError(
                "Start NodeItem must come before or be the same as the end NodeItem in the document structure."
            )

        new_doc = DoclingDocument(name=f"{self.name}- Extracted Range")

        ref_items = start_parent.children[start_index:end_index]
        node_items = [ref.resolve(self) for ref in ref_items]

        new_doc.add_node_items(node_items=node_items, doc=self)

        if delete:
            self.delete_items_range(
                start=start,
                end=end,
                start_inclusive=start_inclusive,
                end_inclusive=end_inclusive,
            )

        return new_doc

    def insert_document(
        self,
        doc: "DoclingDocument",
        sibling: NodeItem,
        after: bool = True,
    ) -> None:
        """Inserts the content from the body of a DoclingDocument into this document at a specific position.

        :param doc: DoclingDocument: The document whose content will be inserted
        :param sibling: NodeItem: The NodeItem after/before which the new items will be inserted
        :param after: bool: If True, insert after the sibling; if False, insert before (Default value = True)

        :returns: None
        """
        ref_items = doc.body.children
        node_items = [ref.resolve(doc) for ref in ref_items]
        self.insert_node_items(
            sibling=sibling, node_items=node_items, doc=doc, after=after
        )

    def add_document(
        self,
        doc: "DoclingDocument",
        parent: Optional[NodeItem] = None,
    ) -> None:
        """Adds the content from the body of a DoclingDocument to this document under a specific parent.

        :param doc: DoclingDocument: The document whose content will be added
        :param parent: Optional[NodeItem]: The parent NodeItem under which new items are added (Default value = None)

        :returns: None
        """
        ref_items = doc.body.children
        node_items = [ref.resolve(doc) for ref in ref_items]
        self.add_node_items(node_items=node_items, doc=doc, parent=parent)

    def add_node_items(
        self,
        node_items: List[NodeItem],
        doc: "DoclingDocument",
        parent: Optional[NodeItem] = None,
    ) -> None:
        """Adds multiple NodeItems and their children under a parent in this document.

        :param node_items: list[NodeItem]: The NodeItems to be added
        :param doc: DoclingDocument: The document to which the NodeItems and their children belong
        :param parent: Optional[NodeItem]: The parent NodeItem under which new items are added (Default value = None)

        :returns: None
        """
        parent = self.body if parent is None else parent

        # Check for ListItem parent violations
        if not isinstance(parent, ListGroup):
            for item in node_items:
                if isinstance(item, ListItem):
                    raise ValueError("Cannot add ListItem into a non-ListGroup parent.")

        # Append the NodeItems to the document content

        parent_ref = parent.get_ref()

        new_refs = self._append_item_copies(
            node_items=node_items, parent_ref=parent_ref, doc=doc
        )

        # Add the new item refs in the document structure

        for ref in new_refs:
            parent.children.append(ref)

    def insert_node_items(
        self,
        sibling: NodeItem,
        node_items: List[NodeItem],
        doc: "DoclingDocument",
        after: bool = True,
    ) -> None:
        """Insert multiple NodeItems and their children at a specific position in the document.

        :param sibling: NodeItem: The NodeItem after/before which the new items will be inserted
        :param node_items: list[NodeItem]: The NodeItems to be inserted
        :param doc: DoclingDocument: The document to which the NodeItems and their children belong
        :param after: bool: If True, insert after the sibling; if False, insert before (Default value = True)

        :returns: None
        """
        # Check for ListItem parent violations
        parent = sibling.parent.resolve(self) if sibling.parent else self.body

        if not isinstance(parent, ListGroup):
            for item in node_items:
                if isinstance(item, ListItem):
                    raise ValueError(
                        "Cannot insert ListItem into a non-ListGroup parent."
                    )

        # Append the NodeItems to the document content

        parent_ref = parent.get_ref()

        new_refs = self._append_item_copies(
            node_items=node_items, parent_ref=parent_ref, doc=doc
        )

        # Get the stack of the sibling

        sibling_ref = sibling.get_ref()

        success, stack = self._get_stack_of_refitem(ref=sibling_ref)

        if not success:
            raise ValueError(
                f"Could not insert at {sibling_ref.cref}: could not find the stack"
            )

        # Insert the new item refs in the document structure

        reversed_new_refs = new_refs[::-1]

        for ref in reversed_new_refs:
            success = self.body._add_sibling(
                doc=self, stack=stack, new_ref=ref, after=after
            )

            if not success:
                raise ValueError(
                    f"Could not insert item {ref.cref} at {sibling.get_ref().cref}"
                )

    def _append_item_copies(
        self,
        node_items: List[NodeItem],
        parent_ref: RefItem,
        doc: "DoclingDocument",
    ) -> List[RefItem]:
        """Append node item copies (with their children) from a different document to the content of this document.

        :param node_items: List[NodeItem]: The NodeItems to be appended
        :param parent_ref: RefItem: The reference of the parent of the new items in this document
        :param doc: DoclingDocument: The document from which the NodeItems are taken

        :returns: List[RefItem]: A list of references to the newly added items in this document
        """
        new_refs: List[RefItem] = []

        for item in node_items:
            item_copy = item.model_copy(deep=True)

            self._append_item(item=item_copy, parent_ref=parent_ref)

            if item_copy.children:
                children_node_items = [ref.resolve(doc) for ref in item_copy.children]

                item_copy.children = self._append_item_copies(
                    node_items=children_node_items,
                    parent_ref=item_copy.get_ref(),
                    doc=doc,
                )

            new_ref = item_copy.get_ref()
            new_refs.append(new_ref)

        return new_refs

    def num_pages(self):
        """num_pages."""
        return len(self.pages.values())

    def validate_tree(self, root: NodeItem) -> bool:
        """validate_tree."""
        for child_ref in root.children:
            child = child_ref.resolve(self)
            if child.parent.resolve(self) != root or not self.validate_tree(child):
                return False

        if isinstance(root, TableItem):
            for cell in root.data.table_cells:
                if isinstance(cell, RichTableCell) and (
                    (par_ref := cell.ref.resolve(self).parent) is None
                    or par_ref.resolve(self) != root
                ):
                    return False

        return True

    def iterate_items(
        self,
        root: Optional[NodeItem] = None,
        with_groups: bool = False,
        traverse_pictures: bool = False,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        _level: int = 0,  # deprecated
    ) -> typing.Iterable[Tuple[NodeItem, int]]:  # tuple of node and level
        """Iterate elements with level."""
        for item, stack in self._iterate_items_with_stack(
            root=root,
            with_groups=with_groups,
            traverse_pictures=traverse_pictures,
            page_nrs={page_no} if page_no is not None else None,
            included_content_layers=included_content_layers,
        ):
            yield item, len(stack)

    def _iterate_items_with_stack(
        self,
        root: Optional[NodeItem] = None,
        with_groups: bool = False,
        traverse_pictures: bool = False,
        page_nrs: Optional[set[int]] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        _stack: Optional[list[int]] = None,
    ) -> typing.Iterable[Tuple[NodeItem, list[int]]]:  # tuple of node and level
        """Iterate elements with stack."""
        my_layers = (
            included_content_layers
            if included_content_layers is not None
            else DEFAULT_CONTENT_LAYERS
        )
        my_stack: list[int] = _stack if _stack is not None else []

        if not root:
            root = self.body

        # Yield non-group items or group items when with_groups=True

        # Combine conditions to have a single yield point
        should_yield = (
            (not isinstance(root, GroupItem) or with_groups)
            and (
                not isinstance(root, DocItem)
                or (
                    page_nrs is None
                    or any(prov.page_no in page_nrs for prov in root.prov)
                )
            )
            and root.content_layer in my_layers
        )

        if should_yield:
            yield root, my_stack

        my_stack.append(-1)

        allowed_pic_refs: set[str] = (
            {r.cref for r in root.captions}
            if (root_is_picture := isinstance(root, PictureItem))
            else set()
        )

        # Traverse children
        for child_ind, child_ref in enumerate(root.children):
            child = child_ref.resolve(self)
            if (
                root_is_picture
                and not traverse_pictures
                and isinstance(child, NodeItem)
                and child.self_ref not in allowed_pic_refs
            ):
                continue
            my_stack[-1] = child_ind

            if isinstance(child, NodeItem):
                yield from self._iterate_items_with_stack(
                    child,
                    with_groups=with_groups,
                    traverse_pictures=traverse_pictures,
                    page_nrs=page_nrs,
                    _stack=my_stack,
                    included_content_layers=my_layers,
                )

        my_stack.pop()

    def _clear_picture_pil_cache(self):
        """Clear cache storage of all images."""
        for item, level in self.iterate_items(with_groups=False):
            if isinstance(item, PictureItem):
                if item.image is not None and item.image._pil is not None:
                    item.image._pil.close()

    def _list_images_on_disk(self) -> List[Path]:
        """List all images on disk."""
        result: List[Path] = []

        for item, level in self.iterate_items(with_groups=False):
            if isinstance(item, PictureItem):
                if item.image is not None:
                    if (
                        isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "file"
                        and item.image.uri.path is not None
                    ):
                        local_path = Path(unquote(item.image.uri.path))
                        result.append(local_path)
                    elif isinstance(item.image.uri, Path):
                        result.append(item.image.uri)

        return result

    def _with_embedded_pictures(self) -> "DoclingDocument":
        """Document with embedded images.

        Creates a copy of this document where all pictures referenced
        through a file URI are turned into base64 embedded form.
        """
        result: DoclingDocument = copy.deepcopy(self)

        for ix, (item, level) in enumerate(result.iterate_items(with_groups=True)):
            if isinstance(item, PictureItem):

                if item.image is not None:
                    if (
                        isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "file"
                    ):
                        assert isinstance(item.image.uri.path, str)
                        tmp_image = PILImage.open(str(unquote(item.image.uri.path)))
                        item.image = ImageRef.from_pil(tmp_image, dpi=item.image.dpi)

                    elif isinstance(item.image.uri, Path):
                        tmp_image = PILImage.open(str(item.image.uri))
                        item.image = ImageRef.from_pil(tmp_image, dpi=item.image.dpi)

        return result

    def _with_pictures_refs(
        self,
        image_dir: Path,
        page_no: Optional[int],
        reference_path: Optional[Path] = None,
    ) -> "DoclingDocument":
        """Document with images as refs.

        Creates a copy of this document where all picture data is
        saved to image_dir and referenced through file URIs.
        """
        result: DoclingDocument = copy.deepcopy(self)

        img_count = 0
        image_dir.mkdir(parents=True, exist_ok=True)

        if image_dir.is_dir():
            for item, level in result.iterate_items(page_no=page_no, with_groups=False):
                if isinstance(item, PictureItem):
                    img = item.get_image(doc=self)
                    if img is not None:

                        hexhash = PictureItem._image_to_hexhash(img)

                        # loc_path = image_dir / f"image_{img_count:06}.png"
                        if hexhash is not None:
                            loc_path = image_dir / f"image_{img_count:06}_{hexhash}.png"

                            img.save(loc_path)
                            if reference_path is not None:
                                obj_path = relative_path(
                                    reference_path.resolve(),
                                    loc_path.resolve(),
                                )
                            else:
                                obj_path = loc_path

                            if item.image is None:
                                scale = img.size[0] / item.prov[0].bbox.width
                                item.image = ImageRef.from_pil(
                                    image=img, dpi=round(72 * scale)
                                )
                            item.image.uri = Path(obj_path)

                        # if item.image._pil is not None:
                        #    item.image._pil.close()

                    img_count += 1

        return result

    def print_element_tree(self):
        """Print_element_tree."""
        for ix, (item, level) in enumerate(
            self.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers={cl for cl in ContentLayer},
            )
        ):
            if isinstance(item, GroupItem):
                print(
                    " " * level,
                    f"{ix}: {item.label.value} with name={item.name}",
                )
            elif isinstance(item, TextItem):
                print(
                    " " * level,
                    f"{ix}: {item.label.value}: {item.text[:min(len(item.text), 100)]}",
                )

            elif isinstance(item, DocItem):
                print(" " * level, f"{ix}: {item.label.value}")

    def export_to_element_tree(self) -> str:
        """Export_to_element_tree."""
        texts = []
        for ix, (item, level) in enumerate(
            self.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers={cl for cl in ContentLayer},
            )
        ):
            if isinstance(item, GroupItem):
                texts.append(
                    " " * level + f"{ix}: {item.label.value} with name={item.name}"
                )
            elif isinstance(item, TextItem):
                texts.append(
                    " " * level
                    + f"{ix}: {item.label.value}: {item.text[:min(len(item.text), 100)]}"
                )
            elif isinstance(item, DocItem):
                texts.append(" " * level + f"{ix}: {item.label.value}")

        return "\n".join(texts)

    def save_as_json(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        indent: int = 2,
        coord_precision: Optional[int] = None,
        confid_precision: Optional[int] = None,
    ):
        """Save as json."""
        if isinstance(filename, str):
            filename = Path(filename)
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, page_no=None, reference_path=reference_path
        )

        out = new_doc.export_to_dict(
            coord_precision=coord_precision, confid_precision=confid_precision
        )
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Union[str, Path]) -> "DoclingDocument":
        """load_from_json.

        :param filename: The filename to load a saved DoclingDocument from a .json.
        :type filename: Path

        :returns: The loaded DoclingDocument.
        :rtype: DoclingDocument

        """
        if isinstance(filename, str):
            filename = Path(filename)
        with open(filename, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def save_as_yaml(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        default_flow_style: bool = False,
        coord_precision: Optional[int] = None,
        confid_precision: Optional[int] = None,
    ):
        """Save as yaml."""
        if isinstance(filename, str):
            filename = Path(filename)
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, page_no=None, reference_path=reference_path
        )

        out = new_doc.export_to_dict(
            coord_precision=coord_precision, confid_precision=confid_precision
        )
        with open(filename, "w", encoding="utf-8") as fw:
            yaml.dump(out, fw, default_flow_style=default_flow_style)

    @classmethod
    def load_from_yaml(cls, filename: Union[str, Path]) -> "DoclingDocument":
        """load_from_yaml.

        Args:
            filename: The filename to load a YAML-serialized DoclingDocument from.

        Returns:
            DoclingDocument: the loaded DoclingDocument
        """
        if isinstance(filename, str):
            filename = Path(filename)
        with open(filename, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return DoclingDocument.model_validate(data)

    def export_to_dict(
        self,
        mode: str = "json",
        by_alias: bool = True,
        exclude_none: bool = True,
        coord_precision: Optional[int] = None,
        confid_precision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Export to dict."""
        context = {}
        if coord_precision is not None:
            context[PydanticSerCtxKey.COORD_PREC.value] = coord_precision
        if confid_precision is not None:
            context[PydanticSerCtxKey.CONFID_PREC.value] = confid_precision
        out = self.model_dump(
            mode=mode, by_alias=by_alias, exclude_none=exclude_none, context=context
        )

        return out

    def save_as_markdown(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        strict_text: bool = False,
        escaping_underscores: bool = True,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        page_break_placeholder: Optional[str] = None,
        include_annotations: bool = True,
    ):
        """Save to markdown."""
        if isinstance(filename, str):
            filename = Path(filename)
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, page_no, reference_path=reference_path
        )

        md_out = new_doc.export_to_markdown(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            strict_text=strict_text,
            escape_underscores=escaping_underscores,
            image_placeholder=image_placeholder,
            image_mode=image_mode,
            indent=indent,
            text_width=text_width,
            page_no=page_no,
            included_content_layers=included_content_layers,
            page_break_placeholder=page_break_placeholder,
            include_annotations=include_annotations,
        )

        with open(filename, "w", encoding="utf-8") as fw:
            fw.write(md_out)

    def export_to_markdown(  # noqa: C901
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        strict_text: bool = False,
        escape_underscores: bool = True,
        image_placeholder: str = "<!-- image -->",
        enable_chart_tables: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        page_break_placeholder: Optional[str] = None,  # e.g. "<!-- page break -->",
        include_annotations: bool = True,
        mark_annotations: bool = False,
    ) -> str:
        r"""Serialize to Markdown.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole document.

        :param delim: Deprecated.
        :type delim: str = "\n\n"
        :param from_element: Body slicing start index (inclusive).
                (Default value = 0).
        :type from_element: int = 0
        :param to_element: Body slicing stop index
                (exclusive). (Default value = maxint).
        :type to_element: int = sys.maxsize
        :param labels: The set of document labels to include in the export. None falls
            back to the system-defined default.
        :type labels: Optional[set[DocItemLabel]] = None
        :param strict_text: Deprecated.
        :type strict_text: bool = False
        :param escape_underscores: bool: Whether to escape underscores in the
            text content of the document. (Default value = True).
        :type escape_underscores: bool = True
        :param image_placeholder: The placeholder to include to position
            images in the markdown. (Default value = "\<!-- image --\>").
        :type image_placeholder: str = "<!-- image -->"
        :param image_mode: The mode to use for including images in the
            markdown. (Default value = ImageRefMode.PLACEHOLDER).
        :type image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
        :param indent: The indent in spaces of the nested lists.
            (Default value = 4).
        :type indent: int = 4
        :param included_content_layers: The set of layels to include in the export. None
            falls back to the system-defined default.
        :type included_content_layers: Optional[set[ContentLayer]] = None
        :param page_break_placeholder: The placeholder to include for marking page
            breaks. None means no page break placeholder will be used.
        :type page_break_placeholder: Optional[str] = None
        :param include_annotations: bool: Whether to include annotations in the export.
            (Default value = True).
        :type include_annotations: bool = True
        :param mark_annotations: bool: Whether to mark annotations in the export; only
            relevant if include_annotations is True. (Default value = False).
        :type mark_annotations: bool = False
        :returns: The exported Markdown representation.
        :rtype: str
        """
        from docling_core.transforms.serializer.markdown import (
            MarkdownDocSerializer,
            MarkdownParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        my_layers = (
            included_content_layers
            if included_content_layers is not None
            else DEFAULT_CONTENT_LAYERS
        )
        serializer = MarkdownDocSerializer(
            doc=self,
            params=MarkdownParams(
                labels=my_labels,
                layers=my_layers,
                pages={page_no} if page_no is not None else None,
                start_idx=from_element,
                stop_idx=to_element,
                escape_underscores=escape_underscores,
                image_placeholder=image_placeholder,
                enable_chart_tables=enable_chart_tables,
                image_mode=image_mode,
                indent=indent,
                wrap_width=text_width if text_width > 0 else None,
                page_break_placeholder=page_break_placeholder,
                include_annotations=include_annotations,
                mark_annotations=mark_annotations,
            ),
        )
        ser_res = serializer.serialize()

        if delim != "\n\n":
            _logger.warning(
                "Parameter `delim` has been deprecated and will be ignored.",
            )
        if strict_text:
            _logger.warning(
                "Parameter `strict_text` has been deprecated and will be ignored.",
            )

        return ser_res.text

    def export_to_text(  # noqa: C901
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = 1000000,
        labels: Optional[set[DocItemLabel]] = None,
    ) -> str:
        """export_to_text."""
        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS

        return self.export_to_markdown(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=my_labels,
            strict_text=True,
            escape_underscores=False,
            image_placeholder="",
        )

    def save_as_html(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        formula_to_mathml: bool = True,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = "null",  # should be deprecated
        included_content_layers: Optional[set[ContentLayer]] = None,
        split_page_view: bool = False,
        include_annotations: bool = True,
    ):
        """Save to HTML."""
        if isinstance(filename, str):
            filename = Path(filename)

        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, page_no, reference_path=reference_path
        )

        html_out = new_doc.export_to_html(
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            image_mode=image_mode,
            formula_to_mathml=formula_to_mathml,
            page_no=page_no,
            html_lang=html_lang,
            html_head=html_head,
            included_content_layers=included_content_layers,
            split_page_view=split_page_view,
            include_annotations=include_annotations,
        )

        with open(filename, "w", encoding="utf-8") as fw:
            fw.write(html_out)

    def _get_output_paths(
        self, filename: Union[str, Path], artifacts_dir: Optional[Path] = None
    ) -> Tuple[Path, Optional[Path]]:
        if isinstance(filename, str):
            filename = Path(filename)
        if artifacts_dir is None:
            # Remove the extension and add '_pictures'
            artifacts_dir = filename.with_suffix("")
            artifacts_dir = artifacts_dir.with_name(artifacts_dir.name + "_artifacts")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = filename.parent
            artifacts_dir = reference_path / artifacts_dir

        return artifacts_dir, reference_path

    def _make_copy_with_refmode(
        self,
        artifacts_dir: Path,
        image_mode: ImageRefMode,
        page_no: Optional[int],
        reference_path: Optional[Path] = None,
    ):
        new_doc = None
        if image_mode == ImageRefMode.PLACEHOLDER:
            new_doc = self
        elif image_mode == ImageRefMode.REFERENCED:
            new_doc = self._with_pictures_refs(
                image_dir=artifacts_dir, page_no=page_no, reference_path=reference_path
            )
        elif image_mode == ImageRefMode.EMBEDDED:
            new_doc = self._with_embedded_pictures()
        else:
            raise ValueError("Unsupported ImageRefMode")
        return new_doc

    def export_to_html(  # noqa: C901
        self,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        enable_chart_tables: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        formula_to_mathml: bool = True,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = "null",  # should be deprecated ...
        included_content_layers: Optional[set[ContentLayer]] = None,
        split_page_view: bool = False,
        include_annotations: bool = True,
    ) -> str:
        r"""Serialize to HTML."""
        from docling_core.transforms.serializer.html import (
            HTMLDocSerializer,
            HTMLOutputStyle,
            HTMLParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        my_layers = (
            included_content_layers
            if included_content_layers is not None
            else DEFAULT_CONTENT_LAYERS
        )

        output_style = HTMLOutputStyle.SINGLE_COLUMN
        if split_page_view:
            output_style = HTMLOutputStyle.SPLIT_PAGE

        params = HTMLParams(
            labels=my_labels,
            layers=my_layers,
            pages={page_no} if page_no is not None else None,
            start_idx=from_element,
            stop_idx=to_element,
            image_mode=image_mode,
            enable_chart_tables=enable_chart_tables,
            formula_to_mathml=formula_to_mathml,
            html_head=html_head,
            html_lang=html_lang,
            output_style=output_style,
            include_annotations=include_annotations,
        )

        if html_head == "null":
            params.html_head = None

        serializer = HTMLDocSerializer(
            doc=self,
            params=params,
        )
        ser_res = serializer.serialize()

        return ser_res.text

    @staticmethod
    def load_from_doctags(  # noqa: C901
        doctag_document: DocTagsDocument, document_name: str = "Document"
    ) -> "DoclingDocument":
        r"""Load Docling document from lists of DocTags and Images."""
        # Maps the recognized tag to a Docling label.
        # Code items will be given DocItemLabel.CODE
        tag_to_doclabel = {
            "title": DocItemLabel.TITLE,
            "document_index": DocItemLabel.DOCUMENT_INDEX,
            "otsl": DocItemLabel.TABLE,
            "section_header_level_1": DocItemLabel.SECTION_HEADER,
            "section_header_level_2": DocItemLabel.SECTION_HEADER,
            "section_header_level_3": DocItemLabel.SECTION_HEADER,
            "section_header_level_4": DocItemLabel.SECTION_HEADER,
            "section_header_level_5": DocItemLabel.SECTION_HEADER,
            "section_header_level_6": DocItemLabel.SECTION_HEADER,
            "checkbox_selected": DocItemLabel.CHECKBOX_SELECTED,
            "checkbox_unselected": DocItemLabel.CHECKBOX_UNSELECTED,
            "text": DocItemLabel.TEXT,
            "page_header": DocItemLabel.PAGE_HEADER,
            "page_footer": DocItemLabel.PAGE_FOOTER,
            "formula": DocItemLabel.FORMULA,
            "caption": DocItemLabel.CAPTION,
            "picture": DocItemLabel.PICTURE,
            "list_item": DocItemLabel.LIST_ITEM,
            "footnote": DocItemLabel.FOOTNOTE,
            "code": DocItemLabel.CODE,
            "key_value_region": DocItemLabel.KEY_VALUE_REGION,
        }

        doc = DoclingDocument(name=document_name)

        def extract_bounding_box(text_chunk: str) -> Optional[BoundingBox]:
            """Extract <loc_...> coords from the chunk, normalized by / 500."""
            coords = re.findall(r"<loc_(\d+)>", text_chunk)
            if len(coords) > 4:
                coords = coords[:4]
            if len(coords) == 4:
                l, t, r, b = map(float, coords)
                return BoundingBox(l=l / 500, t=t / 500, r=r / 500, b=b / 500)
            return None

        def extract_inner_text(text_chunk: str) -> str:
            """Strip all <...> tags inside the chunk to get the raw text content."""
            return re.sub(r"<.*?>", "", text_chunk, flags=re.DOTALL).strip()

        def extract_caption(
            text_chunk: str,
        ) -> tuple[Optional[TextItem], Optional[BoundingBox]]:
            """Extract caption text from the chunk."""
            caption = re.search(r"<caption>(.*?)</caption>", text_chunk)
            if caption is not None:
                caption_content = caption.group(1)
                bbox = extract_bounding_box(caption_content)
                caption_text = extract_inner_text(caption_content)
                caption_item = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=caption_text,
                    parent=None,
                )
            else:
                caption_item = None
                bbox = None
            return caption_item, bbox

        def extract_chart_type(text_chunk: str):
            label = None
            chart_labels = [
                PictureClassificationLabel.PIE_CHART,
                PictureClassificationLabel.BAR_CHART,
                PictureClassificationLabel.STACKED_BAR_CHART,
                PictureClassificationLabel.LINE_CHART,
                PictureClassificationLabel.FLOW_CHART,
                PictureClassificationLabel.SCATTER_CHART,
                PictureClassificationLabel.HEATMAP,
                "line",
                "dot_line",
                "vbar_categorical",
                "hbar_categorical",
            ]

            # Current SmolDocling can predict different labels:
            chart_labels_mapping = {
                "line": PictureClassificationLabel.LINE_CHART,
                "dot_line": PictureClassificationLabel.LINE_CHART,
                "vbar_categorical": PictureClassificationLabel.BAR_CHART,
                "hbar_categorical": PictureClassificationLabel.BAR_CHART,
            }

            for clabel in chart_labels:
                tag = f"<{clabel}>"
                if tag in text_chunk:
                    if clabel in chart_labels_mapping:
                        label = PictureClassificationLabel(chart_labels_mapping[clabel])
                    else:
                        label = PictureClassificationLabel(clabel)
                    break
            return label

        def parse_key_value_item(
            tokens: str, image: Optional[PILImage.Image] = None
        ) -> Tuple[GraphData, Optional[ProvenanceItem]]:
            if image is not None:
                pg_width = image.width
                pg_height = image.height
            else:
                pg_width = 1
                pg_height = 1

            start_locs_match = re.search(r"<key_value_region>(.*?)<key", tokens)
            if start_locs_match:
                overall_locs = start_locs_match.group(1)
                overall_bbox = extract_bounding_box(overall_locs) if image else None
                overall_prov = (
                    ProvenanceItem(
                        bbox=overall_bbox.resize_by_scale(pg_width, pg_height),
                        charspan=(0, 0),
                        page_no=1,
                    )
                    if overall_bbox
                    else None
                )
            else:
                overall_prov = None

            # here we assumed the labels as only key or value, later on we can update
            # it to have unspecified, checkbox etc.
            cell_pattern = re.compile(
                r"<(?P<label>key|value)_(?P<id>\d+)>"
                r"(?P<content>.*?)"
                r"</(?P=label)_(?P=id)>",
                re.DOTALL,
            )

            cells: List["GraphCell"] = []
            links: List["GraphLink"] = []
            raw_link_predictions = []

            for cell_match in cell_pattern.finditer(tokens):
                cell_label_str = cell_match.group("label")  # "key" or "value"
                cell_id = int(cell_match.group("id"))
                raw_content = cell_match.group("content")

                # link tokens
                link_matches = re.findall(r"<link_(\d+)>", raw_content)

                cell_bbox = extract_bounding_box(raw_content) if image else None
                cell_prov = None
                if cell_bbox is not None:
                    cell_prov = ProvenanceItem(
                        bbox=cell_bbox.resize_by_scale(pg_width, pg_height),
                        charspan=(0, 0),
                        page_no=1,
                    )

                cleaned_text = re.sub(r"<loc_\d+>", "", raw_content)
                cleaned_text = re.sub(r"<link_\d+>", "", cleaned_text).strip()

                cell_obj = GraphCell(
                    label=GraphCellLabel(cell_label_str),
                    cell_id=cell_id,
                    text=cleaned_text,
                    orig=cleaned_text,
                    prov=cell_prov,
                    item_ref=None,
                )
                cells.append(cell_obj)

                cell_ids = {cell.cell_id for cell in cells}

                for target_str in link_matches:
                    raw_link_predictions.append((cell_id, int(target_str)))

            cell_ids = {cell.cell_id for cell in cells}

            for source_id, target_id in raw_link_predictions:
                # basic check to validate the prediction
                if target_id not in cell_ids:
                    continue
                link_obj = GraphLink(
                    label=GraphLinkLabel.TO_VALUE,
                    source_cell_id=source_id,
                    target_cell_id=target_id,
                )
                links.append(link_obj)

            return (GraphData(cells=cells, links=links), overall_prov)

        def _add_text(
            full_chunk: str,
            bbox: Optional[BoundingBox],
            pg_width: int,
            pg_height: int,
            page_no: int,
            tag_name: str,
            doc_label: DocItemLabel,
            doc: DoclingDocument,
            parent: Optional[NodeItem],
        ):
            # For everything else, treat as text
            text_content = extract_inner_text(full_chunk)
            element_prov = (
                ProvenanceItem(
                    bbox=bbox.resize_by_scale(pg_width, pg_height),
                    charspan=(0, len(text_content)),
                    page_no=page_no,
                )
                if bbox
                else None
            )

            content_layer = ContentLayer.BODY
            if tag_name in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                content_layer = ContentLayer.FURNITURE

            if doc_label == DocItemLabel.SECTION_HEADER:
                # Extract level from tag_name (e.g. "section_level_header_1" -> 1)
                level = int(tag_name.split("_")[-1])
                doc.add_heading(
                    text=text_content,
                    level=level,
                    prov=element_prov,
                    parent=parent,
                    content_layer=content_layer,
                )
            else:
                doc.add_text(
                    label=doc_label,
                    text=text_content,
                    prov=element_prov,
                    parent=parent,
                    content_layer=content_layer,
                )

        # doc = DoclingDocument(name="Document")
        for pg_idx, doctag_page in enumerate(doctag_document.pages):
            page_doctags = doctag_page.tokens
            image = doctag_page.image

            page_no = pg_idx + 1
            # bounding_boxes = []

            if image is not None:
                pg_width = image.width
                pg_height = image.height
            else:
                pg_width = 1
                pg_height = 1

            doc.add_page(
                page_no=page_no,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=image, dpi=72) if image else None,
            )

            """
            1. Finds all <tag>...</tag>
               blocks in the entire string (multi-line friendly)
               in the order they appear.
            2. For each chunk, extracts bounding box (if any) and inner text.
            3. Adds the item to a DoclingDocument structure with the right label.
            4. Tracks bounding boxes+color in a separate list for later visualization.
            """

            # Regex for root level recognized tags
            tag_pattern = (
                rf"<(?P<tag>{DocItemLabel.TITLE}|{DocItemLabel.DOCUMENT_INDEX}|"
                rf"{DocItemLabel.CHECKBOX_UNSELECTED}|{DocItemLabel.CHECKBOX_SELECTED}|"
                rf"{DocItemLabel.TEXT}|{DocItemLabel.PAGE_HEADER}|{GroupLabel.INLINE}|"
                rf"{DocItemLabel.PAGE_FOOTER}|{DocItemLabel.FORMULA}|"
                rf"{DocItemLabel.CAPTION}|{DocItemLabel.PICTURE}|"
                rf"{DocItemLabel.FOOTNOTE}|{DocItemLabel.CODE}|"
                rf"{DocItemLabel.SECTION_HEADER}_level_[1-6]|"
                rf"{DocumentToken.ORDERED_LIST.value}|"
                rf"{DocumentToken.UNORDERED_LIST.value}|"
                rf"{DocItemLabel.KEY_VALUE_REGION}|"
                rf"{DocumentToken.CHART.value}|"
                rf"{DocumentToken.OTSL.value})>"
                rf"(?P<content>.*?)"
                rf"(?:(?P<closed></(?P=tag)>)|(?P<eof>$))"
            )
            pattern = re.compile(tag_pattern, re.DOTALL)

            # Go through each match in order
            for match in pattern.finditer(page_doctags):
                full_chunk = match.group(0)
                tag_name = match.group("tag")

                bbox = extract_bounding_box(full_chunk)  # Extracts first bbox
                if not match.group("closed"):
                    # no closing tag; only the existence of the item is recovered
                    full_chunk = f"<{tag_name}></{tag_name}>"

                doc_label = tag_to_doclabel.get(tag_name, DocItemLabel.TEXT)

                if tag_name == DocumentToken.OTSL.value:
                    table_data = parse_otsl_table_content(full_chunk)
                    caption, caption_bbox = extract_caption(full_chunk)
                    if caption is not None and caption_bbox is not None:
                        caption.prov.append(
                            ProvenanceItem(
                                bbox=caption_bbox.resize_by_scale(pg_width, pg_height),
                                charspan=(0, len(caption.text)),
                                page_no=page_no,
                            )
                        )
                    if bbox:
                        prov = ProvenanceItem(
                            bbox=bbox.resize_by_scale(pg_width, pg_height),
                            charspan=(0, 0),
                            page_no=page_no,
                        )
                        doc.add_table(data=table_data, prov=prov, caption=caption)
                    else:
                        doc.add_table(data=table_data, caption=caption)

                elif tag_name == GroupLabel.INLINE:
                    inline_group = doc.add_inline_group()
                    content = match.group("content")
                    common_bbox = extract_bounding_box(content)
                    for item_match in pattern.finditer(content):
                        item_tag = item_match.group("tag")
                        _add_text(
                            full_chunk=item_match.group(0),
                            bbox=common_bbox,
                            pg_width=pg_width,
                            pg_height=pg_height,
                            page_no=page_no,
                            tag_name=item_tag,
                            doc_label=tag_to_doclabel.get(item_tag, DocItemLabel.TEXT),
                            doc=doc,
                            parent=inline_group,
                        )

                elif tag_name in [DocItemLabel.PICTURE, DocItemLabel.CHART]:
                    caption, caption_bbox = extract_caption(full_chunk)
                    table_data = None
                    chart_type = None
                    if tag_name == DocumentToken.CHART.value:
                        table_data = parse_otsl_table_content(full_chunk)
                        chart_type = extract_chart_type(full_chunk)
                    if image:
                        if bbox:
                            im_width, im_height = image.size

                            crop_box = (
                                int(bbox.l * im_width),
                                int(bbox.t * im_height),
                                int(bbox.r * im_width),
                                int(bbox.b * im_height),
                            )
                            cropped_image = image.crop(crop_box)
                            pic = doc.add_picture(
                                parent=None,
                                image=ImageRef.from_pil(image=cropped_image, dpi=72),
                                prov=(
                                    ProvenanceItem(
                                        bbox=bbox.resize_by_scale(pg_width, pg_height),
                                        charspan=(0, 0),
                                        page_no=page_no,
                                    )
                                ),
                            )
                            # If there is a caption to an image, add it as well
                            if caption is not None and caption_bbox is not None:
                                caption.prov.append(
                                    ProvenanceItem(
                                        bbox=caption_bbox.resize_by_scale(
                                            pg_width, pg_height
                                        ),
                                        charspan=(0, len(caption.text)),
                                        page_no=page_no,
                                    )
                                )
                                pic.captions.append(caption.get_ref())
                            pic_title = "picture"
                            if chart_type is not None:
                                pic.annotations.append(
                                    PictureClassificationData(
                                        provenance="load_from_doctags",
                                        predicted_classes=[
                                            # chart_type
                                            PictureClassificationClass(
                                                class_name=chart_type, confidence=1.0
                                            )
                                        ],
                                    )
                                )
                                pic_title = chart_type
                            if table_data is not None:
                                # Add chart data as PictureTabularChartData
                                pd = PictureTabularChartData(
                                    chart_data=table_data, title=pic_title
                                )
                                pic.annotations.append(pd)
                    else:
                        if bbox:
                            # In case we don't have access to an binary of an image
                            pic = doc.add_picture(
                                parent=None,
                                prov=ProvenanceItem(
                                    bbox=bbox, charspan=(0, 0), page_no=page_no
                                ),
                            )
                            # If there is a caption to an image, add it as well
                            if caption is not None and caption_bbox is not None:
                                caption.prov.append(
                                    ProvenanceItem(
                                        bbox=caption_bbox.resize_by_scale(
                                            pg_width, pg_height
                                        ),
                                        charspan=(0, len(caption.text)),
                                        page_no=page_no,
                                    )
                                )
                                pic.captions.append(caption.get_ref())
                            if chart_type is not None:
                                pic.annotations.append(
                                    PictureClassificationData(
                                        provenance="load_from_doctags",
                                        predicted_classes=[
                                            # chart_type
                                            PictureClassificationClass(
                                                class_name=chart_type, confidence=1.0
                                            )
                                        ],
                                    )
                                )
                            if table_data is not None:
                                # Add chart data as PictureTabularChartData
                                pd = PictureTabularChartData(
                                    chart_data=table_data, title=pic_title
                                )
                                pic.annotations.append(pd)

                elif tag_name == DocItemLabel.KEY_VALUE_REGION:
                    key_value_data, kv_item_prov = parse_key_value_item(
                        full_chunk, image
                    )
                    doc.add_key_values(graph=key_value_data, prov=kv_item_prov)
                elif tag_name in [
                    DocumentToken.ORDERED_LIST.value,
                    DocumentToken.UNORDERED_LIST.value,
                ]:
                    GroupLabel.LIST
                    enum_marker = ""
                    enum_value = 0
                    if tag_name == DocumentToken.ORDERED_LIST.value:
                        GroupLabel.ORDERED_LIST

                    list_item_pattern = (
                        rf"<(?P<tag>{DocItemLabel.LIST_ITEM})>.*?</(?P=tag)>"
                    )
                    li_pattern = re.compile(list_item_pattern, re.DOTALL)
                    # Add list group:
                    new_list = doc.add_list_group(name="list")
                    # Pricess list items
                    for li_match in li_pattern.finditer(full_chunk):
                        enum_value += 1
                        if tag_name == DocumentToken.ORDERED_LIST.value:
                            enum_marker = str(enum_value) + "."

                        li_full_chunk = li_match.group(0)
                        li_bbox = extract_bounding_box(li_full_chunk) if image else None
                        text_content = extract_inner_text(li_full_chunk)
                        # Add list item
                        doc.add_list_item(
                            marker=enum_marker,
                            enumerated=(tag_name == DocumentToken.ORDERED_LIST.value),
                            parent=new_list,
                            text=text_content,
                            prov=(
                                ProvenanceItem(
                                    bbox=li_bbox.resize_by_scale(pg_width, pg_height),
                                    charspan=(0, len(text_content)),
                                    page_no=page_no,
                                )
                                if li_bbox
                                else None
                            ),
                        )
                else:
                    # For everything else, treat as text
                    _add_text(
                        full_chunk=full_chunk,
                        bbox=bbox,
                        pg_width=pg_width,
                        pg_height=pg_height,
                        page_no=page_no,
                        tag_name=tag_name,
                        doc_label=doc_label,
                        doc=doc,
                        parent=None,
                    )
        return doc

    @deprecated("Use save_as_doctags instead.")
    def save_as_document_tokens(self, *args, **kwargs):
        r"""Save the document content to a DocumentToken format."""
        return self.save_as_doctags(*args, **kwargs)

    def save_as_doctags(
        self,
        filename: Union[str, Path],
        delim: str = "",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_text: bool = True,
        minified: bool = False,
    ):
        r"""Save the document content to DocTags format."""
        if isinstance(filename, str):
            filename = Path(filename)
        out = self.export_to_doctags(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            xsize=xsize,
            ysize=ysize,
            add_location=add_location,
            add_content=add_content,
            add_page_index=add_page_index,
            # table specific flags
            add_table_cell_location=add_table_cell_location,
            add_table_cell_text=add_table_cell_text,
            minified=minified,
        )

        with open(filename, "w", encoding="utf-8") as fw:
            fw.write(out)

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(  # noqa: C901
        self,
        delim: str = "",  # deprecated
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_text: bool = True,
        minified: bool = False,
        pages: Optional[set[int]] = None,
    ) -> str:
        r"""Exports the document content to a DocumentToken format.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole main_text.

        :param delim: str:  (Default value = "")  Deprecated
        :param from_element: int:  (Default value = 0)
        :param to_element: Optional[int]:  (Default value = None)
        :param labels: set[DocItemLabel]
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param add_page_index: bool:  (Default value = True)
        :param # table specific flagsadd_table_cell_location: bool
        :param add_table_cell_text: bool:  (Default value = True)
        :param minified: bool:  (Default value = False)
        :param pages: set[int]: (Default value = None)
        :returns: The content of the document formatted as a DocTags string.
        :rtype: str
        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        serializer = DocTagsDocSerializer(
            doc=self,
            params=DocTagsParams(
                labels=my_labels,
                # layers=...,  # not exposed
                start_idx=from_element,
                stop_idx=to_element,
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                # add_caption=...,  # not exposed
                add_content=add_content,
                add_page_break=add_page_index,
                add_table_cell_location=add_table_cell_location,
                add_table_cell_text=add_table_cell_text,
                pages=pages,
                mode=(
                    DocTagsParams.Mode.MINIFIED
                    if minified
                    else DocTagsParams.Mode.HUMAN_FRIENDLY
                ),
            ),
        )
        ser_res = serializer.serialize()
        return ser_res.text

    def _export_to_indented_text(
        self,
        indent="  ",
        max_text_len: int = -1,
        explicit_tables: bool = False,
    ):
        """Export the document to indented text to expose hierarchy."""
        result = []

        def get_text(text: str, max_text_len: int):

            middle = " ... "

            if max_text_len == -1:
                return text
            elif len(text) < max_text_len + len(middle):
                return text
            else:
                tbeg = int((max_text_len - len(middle)) / 2)
                tend = int(max_text_len - tbeg)

                return text[0:tbeg] + middle + text[-tend:]

        for i, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                result.append(
                    indent * level
                    + f"item-{i} at level {level}: {item.label}: group {item.name}"
                )

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, SectionHeaderItem):
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, ListItem) and item.label in [DocItemLabel.LIST_ITEM]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, TextItem):
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}: {text}"
                )

            elif isinstance(item, TableItem):

                result.append(
                    indent * level
                    + f"item-{i} at level {level}: {item.label} with "
                    + f"[{item.data.num_rows}x{item.data.num_cols}]"
                )

                for _ in item.captions:
                    caption = _.resolve(self)
                    result.append(
                        indent * (level + 1)
                        + f"item-{i} at level {level + 1}: {caption.label}: "
                        + f"{caption.text}"
                    )

                if explicit_tables:
                    grid: list[list[str]] = []
                    for i, row in enumerate(item.data.grid):
                        grid.append([])
                        for j, cell in enumerate(row):
                            if j < 10:
                                text = get_text(
                                    cell._get_text(doc=self), max_text_len=16
                                )
                                grid[-1].append(text)

                    result.append("\n" + tabulate(grid) + "\n")

            elif isinstance(item, PictureItem):

                result.append(
                    indent * level + f"item-{i} at level {level}: {item.label}"
                )

                for _ in item.captions:
                    caption = _.resolve(self)
                    result.append(
                        indent * (level + 1)
                        + f"item-{i} at level {level + 1}: {caption.label}: "
                        + f"{caption.text}"
                    )

            elif isinstance(item, DocItem):
                result.append(
                    indent * (level + 1)
                    + f"item-{i} at level {level}: {item.label}: ignored"
                )

        return "\n".join(result)

    def add_page(
        self, page_no: int, size: Size, image: Optional[ImageRef] = None
    ) -> PageItem:
        """add_page.

        :param page_no: int:
        :param size: Size:

        """
        pitem = PageItem(page_no=page_no, size=size, image=image)

        self.pages[page_no] = pitem
        return pitem

    def get_visualization(
        self,
        show_label: bool = True,
        show_branch_numbering: bool = False,
        viz_mode: Literal["reading_order", "key_value"] = "reading_order",
        show_cell_id: bool = False,
    ) -> dict[Optional[int], PILImage.Image]:
        """Get visualization of the document as images by page.

        :param show_label: Show labels on elements (applies to all visualizers).
        :type show_label: bool
        :param show_branch_numbering: Show branch numbering (reading order visualizer only).
        :type show_branch_numbering: bool
        :param visualizer: Which visualizer to use. One of 'reading_order' (default), 'key_value'.
        :type visualizer: str
        :param show_cell_id: Show cell IDs (key value visualizer only).
        :type show_cell_id: bool

        :returns: Dictionary mapping page numbers to PIL images.
        :rtype: dict[Optional[int], PILImage.Image]
        """
        from docling_core.transforms.visualizer.base import BaseVisualizer
        from docling_core.transforms.visualizer.key_value_visualizer import (
            KeyValueVisualizer,
        )
        from docling_core.transforms.visualizer.layout_visualizer import (
            LayoutVisualizer,
        )
        from docling_core.transforms.visualizer.reading_order_visualizer import (
            ReadingOrderVisualizer,
        )

        visualizer_obj: BaseVisualizer
        if viz_mode == "reading_order":
            visualizer_obj = ReadingOrderVisualizer(
                base_visualizer=LayoutVisualizer(
                    params=LayoutVisualizer.Params(
                        show_label=show_label,
                    ),
                ),
                params=ReadingOrderVisualizer.Params(
                    show_branch_numbering=show_branch_numbering,
                ),
            )
        elif viz_mode == "key_value":
            visualizer_obj = KeyValueVisualizer(
                base_visualizer=LayoutVisualizer(
                    params=LayoutVisualizer.Params(
                        show_label=show_label,
                    ),
                ),
                params=KeyValueVisualizer.Params(
                    show_label=show_label,
                    show_cell_id=show_cell_id,
                ),
            )
        else:
            raise ValueError(f"Unknown visualization mode: {viz_mode}")

        images = visualizer_obj.get_visualization(doc=self)
        return images

    @field_validator("version")
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this document version is compatible with SDK schema version."""
        sdk_match = re.match(VERSION_PATTERN, CURRENT_VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or sdk_match is None
            or doc_match["major"] != sdk_match["major"]
            or doc_match["minor"] > sdk_match["minor"]
        ):
            raise ValueError(
                f"Doc version {v} incompatible with SDK schema version {CURRENT_VERSION}"
            )
        else:
            return CURRENT_VERSION

    @model_validator(mode="after")  # type: ignore
    @classmethod
    def validate_document(cls, d: "DoclingDocument"):
        """validate_document."""
        with warnings.catch_warnings():
            # ignore warning from deprecated furniture
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if not d.validate_tree(d.body) or not d.validate_tree(d.furniture):
                raise ValueError("Document hierachy is inconsistent.")

        return d

    @model_validator(mode="after")
    def validate_misplaced_list_items(self):
        """validate_misplaced_list_items."""
        # find list items without list parent, putting succesive ones together
        misplaced_list_items: list[list[ListItem]] = []
        prev: Optional[NodeItem] = None
        for item, _ in self.iterate_items(
            traverse_pictures=True,
            included_content_layers={c for c in ContentLayer},
            with_groups=True,  # so that we can distinguish neighboring lists
        ):
            if isinstance(item, ListItem) and (
                item.parent is None
                or not isinstance(item.parent.resolve(doc=self), ListGroup)
            ):
                if isinstance(prev, ListItem) and (
                    prev.parent is None or prev.parent.resolve(self) == self.body
                ):  # case of continuing list
                    misplaced_list_items[-1].append(item)
                else:  # case of new list
                    misplaced_list_items.append([item])
            prev = item

        for curr_list_items in reversed(misplaced_list_items):

            # add group
            new_group = ListGroup(self_ref="#")
            self.insert_item_before_sibling(
                new_item=new_group,
                sibling=curr_list_items[0],
            )

            # delete list items from document (should not be affected by group addition)
            self.delete_items(node_items=curr_list_items)

            # add list items to new group
            for li in curr_list_items:
                self.add_list_item(
                    text=li.text,
                    enumerated=li.enumerated,
                    marker=li.marker,
                    orig=li.orig,
                    prov=li.prov[0] if li.prov else None,
                    parent=new_group,
                    content_layer=li.content_layer,
                    formatting=li.formatting,
                    hyperlink=li.hyperlink,
                )
        return self

    class _DocIndex(BaseModel):
        """A document merge buffer."""

        groups: list[GroupItem] = []
        texts: list[TextItem] = []
        pictures: list[PictureItem] = []
        tables: list[TableItem] = []
        key_value_items: list[KeyValueItem] = []
        form_items: list[FormItem] = []

        pages: dict[int, PageItem] = {}

        _body: Optional[GroupItem] = None
        _max_page: int = 0
        _names: list[str] = []

        def get_item_list(self, key: str) -> list[NodeItem]:
            return getattr(self, key)

        def index(
            self, doc: "DoclingDocument", page_nrs: Optional[set[int]] = None
        ) -> None:

            orig_ref_to_new_ref: dict[str, str] = {}
            page_delta = self._max_page - min(doc.pages.keys()) + 1 if doc.pages else 0

            if self._body is None:
                self._body = GroupItem(**doc.body.model_dump(exclude={"children"}))

            self._names.append(doc.name)

            # collect items in traversal order
            for item, _ in doc._iterate_items_with_stack(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers={c for c in ContentLayer},
                page_nrs=page_nrs,
            ):
                key = item.self_ref.split("/")[1]
                is_body = key == "body"
                new_cref = (
                    "#/body" if is_body else f"#/{key}/{len(self.get_item_list(key))}"
                )
                # register cref mapping:
                orig_ref_to_new_ref[item.self_ref] = new_cref

                if not is_body:
                    new_item = copy.deepcopy(item)
                    new_item.children = []

                    # put item in the right list
                    self.get_item_list(key).append(new_item)

                    # update item's self reference
                    new_item.self_ref = new_cref

                    if isinstance(new_item, DocItem):
                        # update page numbers
                        # NOTE other prov sources (e.g. GraphCell) currently not covered
                        for prov in new_item.prov:
                            prov.page_no += page_delta

                    if item.parent:
                        # set item's parent
                        new_parent_cref = orig_ref_to_new_ref[item.parent.cref]
                        new_item.parent = RefItem(cref=new_parent_cref)

                        # add item to parent's children
                        path_components = new_parent_cref.split("/")
                        num_components = len(path_components)
                        if num_components == 3:
                            _, parent_key, parent_index_str = path_components
                            parent_index = int(parent_index_str)
                            parent_item = self.get_item_list(parent_key)[parent_index]

                            # update captions field (not possible in iterate_items order):
                            if isinstance(parent_item, FloatingItem):
                                for cap_it, cap in enumerate(parent_item.captions):
                                    if cap.cref == item.self_ref:
                                        parent_item.captions[cap_it] = RefItem(
                                            cref=new_cref
                                        )
                                        break

                            # update rich table cells references:
                            if isinstance(parent_item, TableItem):
                                for cell in parent_item.data.table_cells:
                                    if (
                                        isinstance(cell, RichTableCell)
                                        and cell.ref.cref == item.self_ref
                                    ):
                                        cell.ref.cref = new_cref
                                        break

                        elif num_components == 2 and path_components[1] == "body":
                            parent_item = self._body
                        else:
                            raise RuntimeError(
                                f"Unsupported ref format: {new_parent_cref}"
                            )
                        parent_item.children.append(RefItem(cref=new_cref))

            # update pages
            new_max_page = None
            for page_nr in doc.pages:
                if page_nrs is None or page_nr in page_nrs:
                    new_page = copy.deepcopy(doc.pages[page_nr])
                    new_page_nr = page_nr + page_delta
                    new_page.page_no = new_page_nr
                    self.pages[new_page_nr] = new_page
                    if new_max_page is None or new_page_nr > new_max_page:
                        new_max_page = new_page_nr
            if new_max_page is not None:
                self._max_page = new_max_page

        def get_name(self) -> str:
            return " + ".join(self._names)

    def _update_from_index(self, doc_index: "_DocIndex") -> None:
        if doc_index._body is not None:
            self.body = doc_index._body
        self.groups = doc_index.groups
        self.texts = doc_index.texts
        self.pictures = doc_index.pictures
        self.tables = doc_index.tables
        self.key_value_items = doc_index.key_value_items
        self.form_items = doc_index.form_items
        self.pages = doc_index.pages
        self.name = doc_index.get_name()

    def _normalize_references(self) -> None:
        doc_index = DoclingDocument._DocIndex()
        doc_index.index(doc=self)
        self._update_from_index(doc_index)

    def filter(self, page_nrs: Optional[set[int]] = None) -> "DoclingDocument":
        """Create a new document based on the provided filter parameters."""
        doc_index = DoclingDocument._DocIndex()
        doc_index.index(doc=self, page_nrs=page_nrs)
        res_doc = DoclingDocument(name=self.name)
        res_doc._update_from_index(doc_index)
        return res_doc

    @classmethod
    def concatenate(cls, docs: Sequence["DoclingDocument"]) -> "DoclingDocument":
        """Concatenate multiple documents into a single document."""
        doc_index = DoclingDocument._DocIndex()
        for doc in docs:
            doc_index.index(doc=doc)

        res_doc = DoclingDocument(name=" + ".join([doc.name for doc in docs]))
        res_doc._update_from_index(doc_index)
        return res_doc

    def _validate_rules(self):
        def validate_list_group(doc: DoclingDocument, item: ListGroup):
            for ref in item.children:
                child = ref.resolve(doc)
                if not isinstance(child, ListItem):
                    raise ValueError(
                        f"ListGroup {item.self_ref} contains non-ListItem {child.self_ref} ({child.label=})"
                    )

        def validate_list_item(doc: DoclingDocument, item: ListItem):
            if item.parent is None:
                raise ValueError(f"ListItem {item.self_ref} has no parent")
            if not isinstance(item.parent.resolve(doc), ListGroup):
                raise ValueError(
                    f"ListItem {item.self_ref} has non-ListGroup parent: {item.parent.cref}"
                )

        def validate_group(doc: DoclingDocument, item: GroupItem):
            if (
                item.parent and not item.children
            ):  # tolerate empty body, but not other groups
                raise ValueError(f"Group {item.self_ref} has no children")

        for item, _ in self.iterate_items(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers={c for c in ContentLayer},
        ):
            if isinstance(item, ListGroup):
                validate_list_group(self, item)

            elif isinstance(item, GroupItem):
                validate_group(self, item)

            elif isinstance(item, ListItem):
                validate_list_item(self, item)

    def add_table_cell(self, table_item: TableItem, cell: TableCell) -> None:
        """Add a table cell to the table."""
        if isinstance(cell, RichTableCell):
            item = cell.ref.resolve(doc=self)
            if isinstance(item, NodeItem) and (
                (not item.parent) or item.parent.cref != table_item.self_ref
            ):
                raise ValueError(
                    f"Trying to add cell with another parent {item.parent} to {table_item.self_ref}"
                )
        table_item.data.table_cells.append(cell)


# deprecated aliases (kept for backwards compatibility):
BasePictureData = BaseAnnotation
PictureDescriptionData = DescriptionAnnotation
PictureMiscData = MiscAnnotation
UnorderedList = ListGroup
