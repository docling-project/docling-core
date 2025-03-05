"""Models for the Docling Document data type."""

import base64
import copy
import hashlib
import html
import json
import logging
import mimetypes
import os
import re
import sys
import textwrap
import typing
import warnings
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union
from urllib.parse import quote, unquote
from xml.etree.cElementTree import SubElement, tostring
from xml.sax.saxutils import unescape

import latex2mathml.converter
import latex2mathml.exceptions
import pandas as pd
import yaml
from PIL import Image as PILImage
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)
from tabulate import tabulate
from typing_extensions import Annotated, Self

from docling_core.search.package import VERSION_PATTERN
from docling_core.types.base import _JSON_POINTER_REGEX
from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    GraphCellLabel,
    GraphLinkLabel,
    GroupLabel,
)
from docling_core.types.doc.tokens import DocumentToken, TableToken
from docling_core.types.doc.utils import (
    get_html_tag_with_text_direction,
    get_text_direction,
    relative_path,
)

_logger = logging.getLogger(__name__)

Uint64 = typing.Annotated[int, Field(ge=0, le=(2**64 - 1))]
LevelNumber = typing.Annotated[int, Field(ge=1, le=100)]
CURRENT_VERSION: Final = "1.2.0"

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
}

DOCUMENT_TOKENS_EXPORT_LABELS = DEFAULT_EXPORT_LABELS.copy()
DOCUMENT_TOKENS_EXPORT_LABELS.update(
    [
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
    ]
)


class BasePictureData(BaseModel):
    """BasePictureData."""

    kind: str


class PictureClassificationClass(BaseModel):
    """PictureClassificationData."""

    class_name: str
    confidence: float


class PictureClassificationData(BasePictureData):
    """PictureClassificationData."""

    kind: Literal["classification"] = "classification"
    provenance: str
    predicted_classes: List[PictureClassificationClass]


class PictureDescriptionData(BasePictureData):
    """PictureDescriptionData."""

    kind: Literal["description"] = "description"
    text: str
    provenance: str


class PictureMoleculeData(BaseModel):
    """PictureMoleculeData."""

    kind: Literal["molecule_data"] = "molecule_data"

    smi: str
    confidence: float
    class_name: str
    segmentation: List[Tuple[float, float]]
    provenance: str


class PictureMiscData(BaseModel):
    """PictureMiscData."""

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


class PictureChartData(BaseModel):
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


PictureDataType = Annotated[
    Union[
        PictureClassificationData,
        PictureDescriptionData,
        PictureMoleculeData,
        PictureMiscData,
        PictureLineChartData,
        PictureBarChartData,
        PictureStackedBarChartData,
        PicturePieChartData,
        PictureScatterChartData,
    ],
    Field(discriminator="kind"),
]


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
        if isinstance(data, Dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                # "bbox" not in data
                # or data["bbox"] is None
                # or isinstance(data["bbox"], BoundingBox)
                "text"
                in data
            ):
                return data
            text = data["bbox"].get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "

                text = text.strip()
            data["text"] = text

        return data


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: List[TableCell] = []
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

    def resolve(self, doc: "DoclingDocument"):
        """resolve."""
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


class ProvenanceItem(BaseModel):
    """ProvenanceItem."""

    page_no: int
    bbox: BoundingBox
    charspan: Tuple[int, int]


class ContentLayer(str, Enum):
    """ContentLayer."""

    BODY = "body"
    FURNITURE = "furniture"


DEFAULT_CONTENT_LAYERS = {ContentLayer.BODY}


class NodeItem(BaseModel):
    """NodeItem."""

    self_ref: str = Field(pattern=_JSON_POINTER_REGEX)
    parent: Optional[RefItem] = None
    children: List[RefItem] = []

    content_layer: ContentLayer = ContentLayer.BODY

    model_config = ConfigDict(extra="forbid")

    def get_ref(self):
        """get_ref."""
        return RefItem(cref=self.self_ref)


class GroupItem(NodeItem):  # Container type, can't be a leaf node
    """GroupItem."""

    name: str = (
        "group"  # Name of the group, e.g. "Introduction Chapter",
        # "Slide 5", "Navigation menu list", ...
    )
    label: GroupLabel = GroupLabel.UNSPECIFIED


class DocItem(
    NodeItem
):  # Base type for any element that carries content, can be a leaf node
    """DocItem."""

    label: DocItemLabel
    prov: List[ProvenanceItem] = []

    def get_location_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str,
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
            location += f"{loc_str}{new_line}"

        return location

    def get_image(self, doc: "DoclingDocument") -> Optional[PILImage.Image]:
        """Returns the image of this DocItem.

        The function returns None if this DocItem has no valid provenance or
        if a valid image of the page containing this DocItem is not available
        in doc.
        """
        if not len(self.prov):
            return None

        page = doc.pages.get(self.prov[0].page_no)
        if page is None or page.size is None or page.image is None:
            return None

        page_image = page.image.pil_image
        if not page_image:
            return None
        crop_bbox = (
            self.prov[0]
            .bbox.to_top_left_origin(page_height=page.size.height)
            .scale_to_size(old_size=page.size, new_size=page.image.size)
            # .scaled(scale=page_image.height / page.size.height)
        )
        return page_image.crop(crop_bbox.as_tuple())


class TextItem(DocItem):
    """TextItem."""

    label: typing.Literal[
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.FORMULA,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.REFERENCE,
        DocItemLabel.TEXT,
        DocItemLabel.TITLE,
    ]

    orig: str  # untreated representation
    text: str  # sanitized representation

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        body = f"<{self.label.value}>{new_line}"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
            )

        if add_content and self.text is not None:
            body += f"{self.text.strip()}{new_line}"

        body += f"</{self.label.value}>\n"

        return body


class SectionHeaderItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.SECTION_HEADER] = (
        DocItemLabel.SECTION_HEADER  # type: ignore[assignment]
    )
    level: LevelNumber = 1

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        body = f"<{self.label.value}_level_{self.level}>{new_line}"

        # TODO: This must be done through an explicit mapping.
        # assert DocumentToken.is_known_token(
        #    body
        # ), f"failed DocumentToken.is_known_token({body})"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
            )

        if add_content and self.text is not None:
            body += f"{self.text.strip()}{new_line}"

        body += f"</{self.label.value}_level_{self.level}>\n"

        return body


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

    def get_image(self, doc: "DoclingDocument") -> Optional[PILImage.Image]:
        """Returns the image corresponding to this FloatingItem.

        This function returns the PIL image from self.image if one is available.
        Otherwise, it uses DocItem.get_image to get an image of this FloatingItem.

        In particular, when self.image is None, the function returns None if this
        FloatingItem has no valid provenance or the doc does not contain a valid image
        for the required page.
        """
        if self.image is not None:
            return self.image.pil_image
        return super().get_image(doc=doc)


class CodeItem(FloatingItem, TextItem):
    """CodeItem."""

    label: typing.Literal[DocItemLabel.CODE] = (
        DocItemLabel.CODE  # type: ignore[assignment]
    )
    code_language: CodeLanguageLabel = CodeLanguageLabel.UNKNOWN

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        body = f"<{self.label.value}>{new_line}"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
            )

        if add_content and self.text is not None:
            body += f"<_{self.code_language.value}_>{self.text}{new_line}"

        body += f"</{self.label.value}>\n"

        return body


class PictureItem(FloatingItem):
    """PictureItem."""

    label: typing.Literal[DocItemLabel.PICTURE] = DocItemLabel.PICTURE

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

    def _image_to_hexhash(self) -> Optional[str]:
        """Hexash from the image."""
        if self.image is not None and self.image._pil is not None:
            # Convert the image to raw bytes
            image_bytes = self.image._pil.tobytes()

            # Create a hash object (e.g., SHA-256)
            hasher = hashlib.sha256()

            # Feed the image bytes into the hash object
            hasher.update(image_bytes)

            # Get the hexadecimal representation of the hash
            return hasher.hexdigest()

        return None

    def export_to_markdown(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        image_placeholder: str = "<!-- image -->",
    ) -> str:
        """Export picture to Markdown format."""
        default_response = image_placeholder
        error_response = (
            "<!-- 🖼️❌ Image not available. "
            "Please use `PdfPipelineOptions(generate_picture_images=True)`"
            " -->"
        )

        if image_mode == ImageRefMode.PLACEHOLDER:
            return default_response

        elif image_mode == ImageRefMode.EMBEDDED:

            # short-cut: we already have the image in base64
            if (
                isinstance(self.image, ImageRef)
                and isinstance(self.image.uri, AnyUrl)
                and self.image.uri.scheme == "data"
            ):
                text = f"![Image]({self.image.uri})"
                return text

            # get the self.image._pil or crop it out of the page-image
            img = self.get_image(doc)

            if img is not None:
                imgb64 = self._image_to_base64(img)
                text = f"![Image](data:image/png;base64,{imgb64})"

                return text
            else:
                return error_response

        elif image_mode == ImageRefMode.REFERENCED:
            if not isinstance(self.image, ImageRef) or (
                isinstance(self.image.uri, AnyUrl) and self.image.uri.scheme == "data"
            ):
                return default_response

            text = f"![Image]({quote(str(self.image.uri))})"
            return text

        else:
            return default_response

    def export_to_html(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
    ) -> str:
        """Export picture to HTML format."""
        text = ""
        if add_caption and len(self.captions):
            text = self.caption_text(doc)

        caption_text = ""
        if len(text) > 0:
            caption_text = get_html_tag_with_text_direction(
                html_tag="figcaption", text=text
            )

        default_response = f"<figure>{caption_text}</figure>"

        if image_mode == ImageRefMode.PLACEHOLDER:
            return default_response

        elif image_mode == ImageRefMode.EMBEDDED:
            # short-cut: we already have the image in base64
            if (
                isinstance(self.image, ImageRef)
                and isinstance(self.image.uri, AnyUrl)
                and self.image.uri.scheme == "data"
            ):
                img_text = f'<img src="{self.image.uri}">'
                return f"<figure>{caption_text}{img_text}</figure>"

            # get the self.image._pil or crop it out of the page-image
            img = self.get_image(doc)

            if img is not None:
                imgb64 = self._image_to_base64(img)
                img_text = f'<img src="data:image/png;base64,{imgb64}">'

                return f"<figure>{caption_text}{img_text}</figure>"
            else:
                return default_response

        elif image_mode == ImageRefMode.REFERENCED:

            if not isinstance(self.image, ImageRef) or (
                isinstance(self.image.uri, AnyUrl) and self.image.uri.scheme == "data"
            ):
                return default_response

            img_text = f'<img src="{quote(str(self.image.uri))}">'
            return f"<figure>{caption_text}{img_text}</figure>"

        else:
            return default_response

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,  # not used at the moment
    ):
        r"""Export picture to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param # not used at the moment

        """
        body = f"<{self.label.value}>{new_line}"
        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
            )

        classifications = [
            ann
            for ann in self.annotations
            if isinstance(ann, PictureClassificationData)
        ]
        if len(classifications) > 0:
            # ! TODO: currently this code assumes class_name is of type 'str'
            # ! TODO: when it will change to an ENUM --> adapt code
            predicted_class = classifications[0].predicted_classes[0].class_name
            body += DocumentToken.get_picture_classification_token(predicted_class)

        if add_caption and len(self.captions):
            text = self.caption_text(doc)

            if len(text):
                body += f"<{DocItemLabel.CAPTION.value}>"
                for caption in self.captions:
                    body += caption.resolve(doc).get_location_tokens(
                        doc=doc,
                        new_line=new_line,
                        xsize=xsize,
                        ysize=ysize,
                    )
                body += f"{text.strip()}"
                body += f"</{DocItemLabel.CAPTION.value}>"
                body += f"{new_line}"

        body += f"</{self.label.value}>\n"

        return body


class TableItem(FloatingItem):
    """TableItem."""

    data: TableData
    label: typing.Literal[
        DocItemLabel.DOCUMENT_INDEX,
        DocItemLabel.TABLE,
    ] = DocItemLabel.TABLE

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export the table as a Pandas DataFrame."""
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
                    col_name = cell.text
                    if columns[j] != "":
                        col_name = f".{col_name}"
                    columns[j] += col_name

        # Create table data
        table_data = [
            [cell.text for cell in row] for row in self.data.grid[num_headers:]
        ]

        # Create DataFrame
        df = pd.DataFrame(table_data, columns=columns)

        return df

    def export_to_markdown(self) -> str:
        """Export the table as markdown."""
        table = []
        for row in self.data.grid:
            tmp = []
            for col in row:

                # make sure that md tables are not broken
                # due to newline chars in the text
                text = col.text
                text = text.replace("\n", " ")
                tmp.append(text)

            table.append(tmp)

        md_table = ""
        if len(table) > 1 and len(table[0]) > 0:
            try:
                md_table = tabulate(table[1:], headers=table[0], tablefmt="github")
            except ValueError:
                md_table = tabulate(
                    table[1:],
                    headers=table[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        return md_table

    def export_to_html(
        self,
        doc: Optional["DoclingDocument"] = None,
        add_caption: bool = True,
    ) -> str:
        """Export the table as html."""
        if doc is None:
            warnings.warn(
                "The `doc` argument will be mandatory in a future version. "
                "It must be provided to include a caption.",
                DeprecationWarning,
            )

        nrows = self.data.num_rows
        ncols = self.data.num_cols

        text = ""
        if doc is not None and add_caption and len(self.captions):
            text = html.escape(self.caption_text(doc))

        if len(self.data.table_cells) == 0:
            return ""

        body = ""

        for i in range(nrows):
            body += "<tr>"
            for j in range(ncols):
                cell: TableCell = self.data.grid[i][j]

                rowspan, rowstart = (
                    cell.row_span,
                    cell.start_row_offset_idx,
                )
                colspan, colstart = (
                    cell.col_span,
                    cell.start_col_offset_idx,
                )

                if rowstart != i:
                    continue
                if colstart != j:
                    continue

                content = html.escape(cell.text.strip())
                celltag = "td"
                if cell.column_header:
                    celltag = "th"

                opening_tag = f"{celltag}"
                if rowspan > 1:
                    opening_tag += f' rowspan="{rowspan}"'
                if colspan > 1:
                    opening_tag += f' colspan="{colspan}"'

                text_dir = get_text_direction(content)
                if text_dir == "rtl":
                    opening_tag += f' dir="{dir}"'

                body += f"<{opening_tag}>{content}</{celltag}>"
            body += "</tr>"

        # dir = get_text_direction(text)

        if len(text) > 0 and len(body) > 0:
            caption_text = get_html_tag_with_text_direction(
                html_tag="caption", text=text
            )
            body = f"<table>{caption_text}<tbody>{body}</tbody></table>"

        elif len(text) == 0 and len(body) > 0:
            body = f"<table><tbody>{body}</tbody></table>"
        elif len(text) > 0 and len(body) == 0:
            caption_text = get_html_tag_with_text_direction(
                html_tag="caption", text=text
            )
            body = f"<table>{caption_text}</table>"
        else:
            body = "<table></table>"

        return body

    def export_to_otsl(
        self,
        doc: "DoclingDocument",
        add_cell_location: bool = True,
        add_cell_text: bool = True,
        xsize: int = 500,
        ysize: int = 500,
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
                content = cell.text.strip()
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

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_cell_location: bool = True,
        add_cell_text: bool = True,
        add_caption: bool = True,
    ):
        r"""Export table to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_cell_location: bool:  (Default value = True)
        :param add_cell_text: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)

        """
        otsl_tag = DocumentToken.OTSL.value

        body = f"<{otsl_tag}>{new_line}"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
            )

        body += self.export_to_otsl(doc, add_cell_location, add_cell_text, xsize, ysize)

        if add_caption and len(self.captions):
            text = self.caption_text(doc)

            if len(text):
                body += f"<{DocItemLabel.CAPTION.value}>"
                for caption in self.captions:
                    body += caption.resolve(doc).get_location_tokens(
                        doc=doc,
                        new_line=new_line,
                        xsize=xsize,
                        ysize=ysize,
                    )
                body += f"{text.strip()}"
                body += f"</{DocItemLabel.CAPTION.value}>"
                body += f"{new_line}"

        body += f"</{otsl_tag}>\n"

        return body


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

    def _export_to_markdown(self) -> str:
        # TODO add actual implementation
        return "<!-- missing-key-value-item -->"


class FormItem(FloatingItem):
    """FormItem."""

    label: typing.Literal[DocItemLabel.FORM] = DocItemLabel.FORM

    graph: GraphData

    def _export_to_markdown(self) -> str:
        # TODO add actual implementation
        return "<!-- missing-form-item -->"


ContentItem = Annotated[
    Union[
        TextItem,
        SectionHeaderItem,
        ListItem,
        CodeItem,
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

    _HTML_DEFAULT_HEAD: str = r"""<head>
    <link rel="icon" type="image/png"
    href="https://ds4sd.github.io/docling/assets/logo.png"/>
    <meta charset="UTF-8">
    <title>
    Powered by Docling
    </title>
    <style>
    html {
    background-color: LightGray;
    }
    body {
    margin: 0 auto;
    width:800px;
    padding: 30px;
    background-color: White;
    font-family: Arial, sans-serif;
    box-shadow: 10px 10px 10px grey;
    }
    figure{
    display: block;
    width: 100%;
    margin: 0px;
    margin-top: 10px;
    margin-bottom: 10px;
    }
    img {
    display: block;
    margin: auto;
    margin-top: 10px;
    margin-bottom: 10px;
    max-width: 640px;
    max-height: 640px;
    }
    table {
    min-width:500px;
    background-color: White;
    border-collapse: collapse;
    cell-padding: 5px;
    margin: auto;
    margin-top: 10px;
    margin-bottom: 10px;
    }
    th, td {
    border: 1px solid black;
    padding: 8px;
    }
    th {
    font-weight: bold;
    }
    table tr:nth-child(even) td{
    background-color: LightGray;
    }
    math annotation {
    display: none;
    }
    .formula-not-decoded {
    background: repeating-linear-gradient(
    45deg, /* Angle of the stripes */
    LightGray, /* First color */
    LightGray 10px, /* Length of the first color */
    White 10px, /* Second color */
    White 20px /* Length of the second color */
    );
    margin: 0;
    text-align: center;
    }
    </style>
    </head>"""

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

    groups: List[GroupItem] = []
    texts: List[Union[SectionHeaderItem, ListItem, TextItem, CodeItem]] = []
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
    ):
        """add_list_item.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        marker = marker or "-"

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            enumerated=enumerated,
            marker=marker,
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
            )

        elif label in [DocItemLabel.LIST_ITEM]:
            return self.add_list_item(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
            )

        elif label in [DocItemLabel.SECTION_HEADER]:
            return self.add_heading(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
            )

        elif label in [DocItemLabel.CODE]:
            return self.add_code(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
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
            label=label, data=data, self_ref=cref, parent=parent.get_ref()
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
        annotations: List[PictureDataType] = [],
        image: Optional[ImageRef] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ):
        """add_picture.

        :param data: List[PictureData]: (Default value = [])
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
            annotations=annotations,
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
    ):
        """add_title.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        text_item = TextItem(
            label=DocItemLabel.TITLE,
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            text_item.prov.append(prov)
        if content_layer:
            text_item.content_layer = content_layer

        self.texts.append(text_item)
        parent.children.append(RefItem(cref=cref))

        return text_item

    def add_code(
        self,
        text: str,
        code_language: Optional[CodeLanguageLabel] = None,
        orig: Optional[str] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
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

    def add_heading(
        self,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
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

    def num_pages(self):
        """num_pages."""
        return len(self.pages.values())

    def validate_tree(self, root) -> bool:
        """validate_tree."""
        res = []
        for child_ref in root.children:
            child = child_ref.resolve(self)
            if child.parent.resolve(self) != root:
                return False
            res.append(self.validate_tree(child))

        return all(res) or len(res) == 0

    def iterate_items(
        self,
        root: Optional[NodeItem] = None,
        with_groups: bool = False,
        traverse_pictures: bool = False,
        page_no: Optional[int] = None,
        included_content_layers: set[ContentLayer] = DEFAULT_CONTENT_LAYERS,
        _level: int = 0,  # fixed parameter, carries through the node nesting level
    ) -> typing.Iterable[Tuple[NodeItem, int]]:  # tuple of node and level
        """iterate_elements.

        :param root: Optional[NodeItem]:  (Default value = None)
        :param with_groups: bool:  (Default value = False)
        :param traverse_pictures: bool:  (Default value = False)
        :param page_no: Optional[int]:  (Default value = None)
        :param _level:  (Default value = 0)
        :param # fixed parameter:
        :param carries through the node nesting level:
        """
        if not root:
            root = self.body

        # Yield non-group items or group items when with_groups=True

        # Combine conditions to have a single yield point
        should_yield = (
            (not isinstance(root, GroupItem) or with_groups)
            and (
                not isinstance(root, DocItem)
                or (
                    page_no is None
                    or any(prov.page_no == page_no for prov in root.prov)
                )
            )
            and root.content_layer in included_content_layers
        )

        if should_yield:
            yield root, _level

        # Handle picture traversal - only traverse children if requested
        if isinstance(root, PictureItem) and not traverse_pictures:
            return

        # Traverse children
        for child_ref in root.children:
            child = child_ref.resolve(self)
            if isinstance(child, NodeItem):
                yield from self.iterate_items(
                    child,
                    with_groups=with_groups,
                    traverse_pictures=traverse_pictures,
                    page_no=page_no,
                    _level=_level + 1,
                    included_content_layers=included_content_layers,
                )

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
        self, image_dir: Path, reference_path: Optional[Path] = None
    ) -> "DoclingDocument":
        """Document with images as refs.

        Creates a copy of this document where all picture data is
        saved to image_dir and referenced through file URIs.
        """
        result: DoclingDocument = copy.deepcopy(self)

        img_count = 0
        image_dir.mkdir(parents=True, exist_ok=True)

        if image_dir.is_dir():
            for item, level in result.iterate_items(with_groups=False):
                if isinstance(item, PictureItem):

                    if (
                        item.image is not None
                        and isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "data"
                        and item.image.pil_image is not None
                    ):
                        img = item.image.pil_image

                        hexhash = item._image_to_hexhash()

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

                            item.image.uri = Path(obj_path)

                        # if item.image._pil is not None:
                        #    item.image._pil.close()

                    img_count += 1

        return result

    def print_element_tree(self):
        """Print_element_tree."""
        for ix, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                print(
                    " " * level,
                    f"{ix}: {item.label.value} with name={item.name}",
                )
            elif isinstance(item, DocItem):
                print(" " * level, f"{ix}: {item.label.value}")

    def export_to_element_tree(self) -> str:
        """Export_to_element_tree."""
        texts = []
        for ix, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                texts.append(
                    " " * level + f"{ix}: {item.label.value} with name={item.name}"
                )
            elif isinstance(item, DocItem):
                texts.append(" " * level + f"{ix}: {item.label.value}")

        return "\n".join(texts)

    def save_as_json(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        indent: int = 2,
    ):
        """Save as json."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
        )

        out = new_doc.export_to_dict()
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Path) -> "DoclingDocument":
        """load_from_json.

        :param filename: The filename to load a saved DoclingDocument from a .json.
        :type filename: Path

        :returns: The loaded DoclingDocument.
        :rtype: DoclingDocument

        """
        with open(filename, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def save_as_yaml(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        default_flow_style: bool = False,
    ):
        """Save as yaml."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
        )

        out = new_doc.export_to_dict()
        with open(filename, "w", encoding="utf-8") as fw:
            yaml.dump(out, fw, default_flow_style=default_flow_style)

    @classmethod
    def load_from_yaml(cls, filename: Path) -> "DoclingDocument":
        """load_from_yaml.

        Args:
            filename: The filename to load a YAML-serialized DoclingDocument from.

        Returns:
            DoclingDocument: the loaded DoclingDocument
        """
        with open(filename, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return DoclingDocument.model_validate(data)

    def export_to_dict(
        self,
        mode: str = "json",
        by_alias: bool = True,
        exclude_none: bool = True,
    ) -> Dict:
        """Export to dict."""
        out = self.model_dump(mode=mode, by_alias=by_alias, exclude_none=exclude_none)

        return out

    def save_as_markdown(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        delim: str = "\n\n",  # TODO: deprecate
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        strict_text: bool = False,
        escaping_underscores: bool = True,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
        included_content_layers: set[ContentLayer] = DEFAULT_CONTENT_LAYERS,
    ):
        """Save to markdown."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
        )

        md_out = new_doc.export_to_markdown(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            strict_text=strict_text,
            escaping_underscores=escaping_underscores,
            image_placeholder=image_placeholder,
            image_mode=image_mode,
            indent=indent,
            text_width=text_width,
            page_no=page_no,
            included_content_layers=included_content_layers,
        )

        with open(filename, "w", encoding="utf-8") as fw:
            fw.write(md_out)

    def export_to_markdown(  # noqa: C901
        self,
        delim: str = "\n\n",  # TODO deprecate
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        strict_text: bool = False,
        escaping_underscores: bool = True,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
        included_content_layers: set[ContentLayer] = DEFAULT_CONTENT_LAYERS,
    ) -> str:
        r"""Serialize to Markdown.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole document.

        :param delim: Delimiter to use when concatenating the various
                Markdown parts. (Default value = "\n").
        :type delim: str = "\n"
        :param from_element: Body slicing start index (inclusive).
                (Default value = 0).
        :type from_element: int = 0
        :param to_element: Body slicing stop index
                (exclusive). (Default value = maxint).
        :type to_element: int = sys.maxsize
        :param labels: The set of document labels to include in the export.
        :type labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS
        :param strict_text: bool: Whether to only include the text content
            of the document. (Default value = False).
        :type strict_text: bool = False
        :param escaping_underscores: bool: Whether to escape underscores in the
            text content of the document. (Default value = True).
        :type escaping_underscores: bool = True
        :param image_placeholder: The placeholder to include to position
            images in the markdown. (Default value = "\<!-- image --\>").
        :type image_placeholder: str = "<!-- image -->"
        :param image_mode: The mode to use for including images in the
            markdown. (Default value = ImageRefMode.PLACEHOLDER).
        :type image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
        :param indent: The indent in spaces of the nested lists.
            (Default value = 4).
        :type indent: int = 4
        :returns: The exported Markdown representation.
        :rtype: str
        """
        comps = self._get_markdown_components(
            node=self.body,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            strict_text=strict_text,
            escaping_underscores=escaping_underscores,
            image_placeholder=image_placeholder,
            image_mode=image_mode,
            indent=indent,
            text_width=text_width,
            page_no=page_no,
            included_content_layers=included_content_layers,
            list_level=0,
            is_inline_scope=False,
            visited=set(),
        )
        return delim.join(comps)

    def _get_markdown_components(  # noqa: C901
        self,
        node: NodeItem,
        from_element: int,
        to_element: int,
        labels: set[DocItemLabel],
        strict_text: bool,
        escaping_underscores: bool,
        image_placeholder: str,
        image_mode: ImageRefMode,
        indent: int,
        text_width: int,
        page_no: Optional[int],
        included_content_layers: set[ContentLayer],
        list_level: int,
        is_inline_scope: bool,
        visited: set[str],  # refs of visited items
    ) -> list[str]:
        components: list[str] = []  # components to concatenate

        # Our export markdown doesn't contain any emphasis styling:
        # Bold, Italic, or Bold-Italic
        # Hence, any underscore that we print into Markdown is coming from document text
        # That means we need to escape it, to properly reflect content in the markdown
        # However, we need to preserve underscores in image URLs
        # to maintain their validity
        # For example: ![image](path/to_image.png) should remain unchanged
        def _escape_underscores(text):
            """Escape underscores but leave them intact in the URL.."""
            # Firstly, identify all the URL patterns.
            url_pattern = r"!\[.*?\]\((.*?)\)"
            # Matches both inline ($...$) and block ($$...$$) LaTeX equations:
            latex_pattern = r"\$\$?(?:\\.|[^$\\])*\$\$?"
            combined_pattern = f"({url_pattern})|({latex_pattern})"

            parts = []
            last_end = 0

            for match in re.finditer(combined_pattern, text):
                # Text to add before the URL (needs to be escaped)
                before_url = text[last_end : match.start()]
                parts.append(re.sub(r"(?<!\\)_", r"\_", before_url))

                # Add the full URL part (do not escape)
                parts.append(match.group(0))
                last_end = match.end()

            # Add the final part of the text (which needs to be escaped)
            if last_end < len(text):
                parts.append(re.sub(r"(?<!\\)_", r"\_", text[last_end:]))

            return "".join(parts)

        def _ingest_text(text: str, do_escape_html=True, do_escape_underscores=True):
            if do_escape_underscores and escaping_underscores:
                text = _escape_underscores(text)
            if do_escape_html:
                text = html.escape(text, quote=False)
            if text:
                components.append(text)

        for ix, (item, level) in enumerate(
            self.iterate_items(
                node,
                with_groups=True,
                page_no=page_no,
                included_content_layers=included_content_layers,
            )
        ):
            if item.self_ref in visited:
                continue
            else:
                visited.add(item.self_ref)

            if ix < from_element or to_element <= ix:
                continue  # skip as many items as you want

            elif (isinstance(item, DocItem)) and (item.label not in labels):
                continue  # skip any label that is not whitelisted

            elif isinstance(item, GroupItem):
                if item.label in [
                    GroupLabel.LIST,
                    GroupLabel.ORDERED_LIST,
                ]:
                    comps = self._get_markdown_components(
                        node=item,
                        from_element=from_element,
                        to_element=to_element,
                        labels=labels,
                        strict_text=strict_text,
                        escaping_underscores=escaping_underscores,
                        image_placeholder=image_placeholder,
                        image_mode=image_mode,
                        indent=indent,
                        text_width=text_width,
                        page_no=page_no,
                        included_content_layers=included_content_layers,
                        list_level=list_level + 1,
                        is_inline_scope=is_inline_scope,
                        visited=visited,
                    )
                    indent_str = list_level * indent * " "
                    is_ol = item.label == GroupLabel.ORDERED_LIST
                    text = "\n".join(
                        [
                            # avoid additional marker on already evaled sublists
                            (
                                c
                                if c and c[0] == " "
                                else f"{indent_str}{f'{i + 1}.' if is_ol else '-'} {c}"
                            )
                            for i, c in enumerate(comps)
                        ]
                    )
                    _ingest_text(
                        text=text,
                        # special chars have already been escaped as needed
                        do_escape_html=False,
                        do_escape_underscores=False,
                    )
                elif item.label == GroupLabel.INLINE:
                    comps = self._get_markdown_components(
                        node=item,
                        from_element=from_element,
                        to_element=to_element,
                        labels=labels,
                        strict_text=strict_text,
                        escaping_underscores=escaping_underscores,
                        image_placeholder=image_placeholder,
                        image_mode=image_mode,
                        indent=indent,
                        text_width=text_width,
                        page_no=page_no,
                        included_content_layers=included_content_layers,
                        list_level=list_level,
                        is_inline_scope=True,
                        visited=visited,
                    )
                    text = " ".join(comps)
                    _ingest_text(
                        text=text,
                        # special chars have already been escaped as needed
                        do_escape_html=False,
                        do_escape_underscores=False,
                    )
                else:
                    continue

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                marker = "" if strict_text else "#"
                text = f"{marker} {item.text}"
                _ingest_text(text.strip())

            elif (
                isinstance(item, TextItem)
                and item.label in [DocItemLabel.SECTION_HEADER]
            ) or isinstance(item, SectionHeaderItem):
                marker = ""
                if not strict_text:
                    marker = "#" * level
                    if len(marker) < 2:
                        marker = "##"
                text = f"{marker} {item.text}"
                _ingest_text(text.strip())

            elif isinstance(item, CodeItem):
                text = f"`{item.text}`" if is_inline_scope else f"```\n{item.text}\n```"
                _ingest_text(text, do_escape_underscores=False, do_escape_html=False)

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.FORMULA]:
                if item.text != "":
                    _ingest_text(
                        f"${item.text}$" if is_inline_scope else f"$${item.text}$$",
                        do_escape_underscores=False,
                        do_escape_html=False,
                    )
                elif item.orig != "":
                    _ingest_text(
                        "<!-- formula-not-decoded -->",
                        do_escape_underscores=False,
                        do_escape_html=False,
                    )

            elif isinstance(item, TextItem):
                if len(item.text) and text_width > 0:
                    text = item.text
                    wrapped_text = textwrap.fill(text, width=text_width)
                    _ingest_text(wrapped_text)
                elif len(item.text):
                    _ingest_text(item.text)

            elif isinstance(item, TableItem) and not strict_text:
                if caption_text := item.caption_text(self):
                    _ingest_text(caption_text)
                md_table = item.export_to_markdown()
                _ingest_text(md_table)

            elif isinstance(item, PictureItem) and not strict_text:
                _ingest_text(item.caption_text(self))

                line = item.export_to_markdown(
                    doc=self,
                    image_placeholder=image_placeholder,
                    image_mode=image_mode,
                )

                _ingest_text(line, do_escape_html=False, do_escape_underscores=False)

            elif isinstance(item, (KeyValueItem, FormItem)):
                text = item._export_to_markdown()
                _ingest_text(text, do_escape_html=False, do_escape_underscores=False)

            elif isinstance(item, DocItem):
                text = "<!-- missing-text -->"
                _ingest_text(text, do_escape_html=False, do_escape_underscores=False)

        return components

    def export_to_text(  # noqa: C901
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = 1000000,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
    ) -> str:
        """export_to_text."""
        return self.export_to_markdown(
            delim,
            from_element,
            to_element,
            labels,
            strict_text=True,
            escaping_underscores=False,
            image_placeholder="",
        )

    def save_as_html(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        formula_to_mathml: bool = True,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = _HTML_DEFAULT_HEAD,
        included_content_layers: set[ContentLayer] = DEFAULT_CONTENT_LAYERS,
    ):
        """Save to HTML."""
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            os.makedirs(artifacts_dir, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir, image_mode, reference_path=reference_path
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
        )

        with open(filename, "w", encoding="utf-8") as fw:
            fw.write(html_out)

    def _get_output_paths(
        self, filename: Path, artifacts_dir: Optional[Path] = None
    ) -> Tuple[Path, Optional[Path]]:
        if artifacts_dir is None:
            # Remove the extension and add '_pictures'
            artifacts_dir = filename.with_suffix("")
            artifacts_dir = artifacts_dir.with_name(artifacts_dir.name + "_artifacts")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = filename.parent
        return artifacts_dir, reference_path

    def _make_copy_with_refmode(
        self,
        artifacts_dir: Path,
        image_mode: ImageRefMode,
        reference_path: Optional[Path] = None,
    ):
        new_doc = None
        if image_mode == ImageRefMode.PLACEHOLDER:
            new_doc = self
        elif image_mode == ImageRefMode.REFERENCED:
            new_doc = self._with_pictures_refs(
                image_dir=artifacts_dir, reference_path=reference_path
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
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        formula_to_mathml: bool = True,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = _HTML_DEFAULT_HEAD,
        included_content_layers: set[ContentLayer] = DEFAULT_CONTENT_LAYERS,
    ) -> str:
        r"""Serialize to HTML."""

        def close_lists(
            curr_level: int,
            prev_level: int,
            in_ordered_list: List[bool],
            html_texts: list[str],
        ):

            if len(in_ordered_list) == 0:
                return (in_ordered_list, html_texts)

            while curr_level < prev_level and len(in_ordered_list) > 0:
                if in_ordered_list[-1]:
                    html_texts.append("</ol>")
                else:
                    html_texts.append("</ul>")

                prev_level -= 1
                in_ordered_list.pop()  # = in_ordered_list[:-1]

            return (in_ordered_list, html_texts)

        head_lines = [
            "<!DOCTYPE html>",
            f'<html lang="{html_lang}">',
            html_head,
        ]
        html_texts: list[str] = []

        prev_level = 0  # Track the previous item's level

        in_ordered_list: List[bool] = []  # False

        def _prepare_tag_content(
            text: str, do_escape_html=True, do_replace_newline=True
        ) -> str:
            if do_escape_html:
                text = html.escape(text, quote=False)
            if do_replace_newline:
                text = text.replace("\n", "<br>")
            return text

        for ix, (item, curr_level) in enumerate(
            self.iterate_items(
                self.body,
                with_groups=True,
                page_no=page_no,
                included_content_layers=included_content_layers,
            )
        ):
            # If we've moved to a lower level, we're exiting one or more groups
            if curr_level < prev_level and len(in_ordered_list) > 0:
                # Calculate how many levels we've exited
                # level_difference = previous_level - level
                # Decrement list_nesting_level for each list group we've exited
                # list_nesting_level = max(0, list_nesting_level - level_difference)

                in_ordered_list, html_texts = close_lists(
                    curr_level=curr_level,
                    prev_level=prev_level,
                    in_ordered_list=in_ordered_list,
                    html_texts=html_texts,
                )

            prev_level = curr_level  # Update previous_level for next iteration

            if ix < from_element or to_element <= ix:
                continue  # skip as many items as you want

            if (isinstance(item, DocItem)) and (item.label not in labels):
                continue  # skip any label that is not whitelisted

            if isinstance(item, GroupItem) and item.label in [
                GroupLabel.ORDERED_LIST,
            ]:

                text = "<ol>"
                html_texts.append(text)

                # Increment list nesting level when entering a new list
                in_ordered_list.append(True)

            elif isinstance(item, GroupItem) and item.label in [
                GroupLabel.LIST,
            ]:

                text = "<ul>"
                html_texts.append(text)

                # Increment list nesting level when entering a new list
                in_ordered_list.append(False)

            elif isinstance(item, GroupItem):
                continue

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                text_inner = _prepare_tag_content(item.text)
                text = get_html_tag_with_text_direction(html_tag="h1", text=text_inner)

                html_texts.append(text)

            elif isinstance(item, SectionHeaderItem):

                section_level: int = min(item.level + 1, 6)

                text = get_html_tag_with_text_direction(
                    html_tag=f"h{section_level}",
                    text=_prepare_tag_content(item.text),
                )
                html_texts.append(text)

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.FORMULA]:

                math_formula = _prepare_tag_content(
                    item.text, do_escape_html=False, do_replace_newline=False
                )
                text = ""

                def _image_fallback(item: TextItem):
                    item_image = item.get_image(doc=self)
                    if item_image is not None:
                        img_ref = ImageRef.from_pil(item_image, dpi=72)
                        return (
                            "<figure>"
                            f'<img src="{img_ref.uri}" alt="{item.orig}" />'
                            "</figure>"
                        )

                img_fallback = _image_fallback(item)

                # If the formula is not processed correcty, use its image
                if (
                    item.text == ""
                    and item.orig != ""
                    and image_mode == ImageRefMode.EMBEDDED
                    and len(item.prov) > 0
                    and img_fallback is not None
                ):
                    text = img_fallback

                # Building a math equation in MathML format
                # ref https://www.w3.org/TR/wai-aria-1.1/#math
                elif formula_to_mathml and len(math_formula) > 0:
                    try:
                        mathml_element = latex2mathml.converter.convert_to_element(
                            math_formula, display="block"
                        )
                        annotation = SubElement(
                            mathml_element, "annotation", dict(encoding="TeX")
                        )
                        annotation.text = math_formula
                        mathml = unescape(tostring(mathml_element, encoding="unicode"))
                        text = f"<div>{mathml}</div>"
                    except Exception as err:
                        _logger.warning(
                            "Malformed formula cannot be rendered. "
                            f"Error {err.__class__.__name__}, formula={math_formula}"
                        )
                        if (
                            image_mode == ImageRefMode.EMBEDDED
                            and len(item.prov) > 0
                            and img_fallback is not None
                        ):
                            text = img_fallback
                        else:
                            text = f"<pre>{math_formula}</pre>"

                elif math_formula != "":
                    text = f"<pre>{math_formula}</pre>"

                if text != "":
                    html_texts.append(text)
                else:
                    html_texts.append(
                        '<div class="formula-not-decoded">Formula not decoded</div>'
                    )

            elif isinstance(item, ListItem):
                text = get_html_tag_with_text_direction(
                    html_tag="li", text=_prepare_tag_content(item.text)
                )
                html_texts.append(text)

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.LIST_ITEM]:
                text = get_html_tag_with_text_direction(
                    html_tag="li", text=_prepare_tag_content(item.text)
                )
                html_texts.append(text)

            elif isinstance(item, CodeItem):
                code_text = _prepare_tag_content(
                    item.text, do_escape_html=False, do_replace_newline=False
                )
                text = f"<pre><code>{code_text}</code></pre>"
                html_texts.append(text)

            elif isinstance(item, TextItem):

                text = get_html_tag_with_text_direction(
                    html_tag="p", text=_prepare_tag_content(item.text)
                )
                html_texts.append(text)

            elif isinstance(item, TableItem):

                text = item.export_to_html(doc=self, add_caption=True)
                html_texts.append(text)

            elif isinstance(item, PictureItem):

                html_texts.append(
                    item.export_to_html(
                        doc=self, add_caption=True, image_mode=image_mode
                    )
                )

            elif isinstance(item, DocItem) and item.label in labels:
                continue

        html_texts.append("</html>")

        lines = []
        lines.extend(head_lines)
        lines.extend(html_texts)

        delim = "\n"
        html_text = (delim.join(lines)).strip()

        return html_text

    def save_as_document_tokens(
        self,
        filename: Path,
        delim: str = "",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DOCUMENT_TOKENS_EXPORT_LABELS,
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_text: bool = True,
    ):
        r"""Save the document content to a DocumentToken format."""
        out = self.export_to_document_tokens(
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
        )

        with open(filename, "w", encoding="utf-8") as fw:
            fw.write(out)

    def export_to_document_tokens(  # noqa: C901
        self,
        delim: str = "",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DOCUMENT_TOKENS_EXPORT_LABELS,
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_text: bool = True,
    ) -> str:
        r"""Exports the document content to a DocumentToken format.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole main_text.

        :param delim: str:  (Default value = "")
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
        :returns: The content of the document formatted as a DocTags string.
        :rtype: str
        """

        def _close_lists(
            current_level: int,
            previous_level: int,
            ordered_list_stack: List[bool],
            output_parts: List[str],
        ) -> List[bool]:
            """Close open list tags until the nesting level matches item's level."""
            while current_level < previous_level and ordered_list_stack:
                last_is_ordered = ordered_list_stack.pop()
                if last_is_ordered:
                    output_parts.append(f"</{DocumentToken.ORDERED_LIST.value}>\n")
                else:
                    output_parts.append(f"</{DocumentToken.UNORDERED_LIST.value}>\n")
                previous_level -= 1
            return ordered_list_stack

        def _add_page_break_if_needed(
            output_parts: List[str],
            item,
            prev_page_no,
            page_break_enabled: bool,
        ):
            """Inserts a page-break token.

            Inserts a page-break token if the item's page number is different
            from the previous item and page breaks are enabled.
            Returns the updated output_parts list and the current page number.
            """
            if not page_break_enabled:
                return output_parts, prev_page_no

            if not item.prov:
                return output_parts, prev_page_no

            current_page_no = item.prov[0].page_no
            if prev_page_no is None:
                return output_parts, current_page_no

            if current_page_no != prev_page_no:
                output_parts.append(f"<{DocumentToken.PAGE_BREAK.value}>\n")

            return output_parts, current_page_no

        def _get_standalone_captions(document_body):
            """Identify captions that are not attached to any table or figure."""
            all_captions = set()
            matched_captions = set()
            for item, _ in self.iterate_items(document_body, with_groups=True):
                if item.label == DocItemLabel.CAPTION:
                    all_captions.update([item.self_ref])
                if item.label in [DocItemLabel.PICTURE, DocItemLabel.TABLE]:
                    matched_captions.update([caption.cref for caption in item.captions])

            return all_captions - matched_captions

        # Initialization
        output_parts: List[str] = []
        ordered_list_stack: List[bool] = []
        previous_level = 0
        previous_page_no = None

        # Precompute standalone captions
        standalone_captions = _get_standalone_captions(self.body)

        # Begin document
        output_parts.append(f"<{DocumentToken.DOCUMENT.value}>{delim}")

        for ix, (item, current_level) in enumerate(
            self.iterate_items(
                self.body,
                with_groups=True,
                included_content_layers={
                    ContentLayer.BODY,
                    ContentLayer.FURNITURE,
                },
            )
        ):
            # Close lists if we've moved to a lower nesting level
            if current_level < previous_level and ordered_list_stack:
                ordered_list_stack = _close_lists(
                    current_level,
                    previous_level,
                    ordered_list_stack,
                    output_parts,
                )
            previous_level = current_level

            # Skip items outside the specified element range
            if ix < from_element or ix >= to_element:
                continue

            # Skip items whose label is not in the allowed set
            if isinstance(item, DocItem) and (item.label not in labels):
                continue

            # Skip captions that are not standalone as they will be included below
            # by the export functions of Table and Picture
            if (
                isinstance(item, TextItem)
                and item.label == DocItemLabel.CAPTION
                and item.self_ref not in standalone_captions
            ):
                continue

            # Handle list groups
            if isinstance(item, GroupItem):
                if item.label == GroupLabel.ORDERED_LIST:
                    output_parts.append(f"<{DocumentToken.ORDERED_LIST.value}>{delim}")
                    ordered_list_stack.append(True)
                elif item.label == GroupLabel.LIST:
                    output_parts.append(
                        f"<{DocumentToken.UNORDERED_LIST.value}>{delim}"
                    )
                    ordered_list_stack.append(False)
                continue

            # For other item types, optionally insert page-break if the page changed
            output_parts, previous_page_no = _add_page_break_if_needed(
                output_parts, item, previous_page_no, add_page_index
            )

            if isinstance(item, SectionHeaderItem):
                output_parts.append(
                    item.export_to_document_tokens(
                        doc=self,
                        new_line=delim,
                        xsize=xsize,
                        ysize=ysize,
                        add_location=add_location,
                        add_content=add_content,
                    )
                )
            elif isinstance(item, CodeItem):
                output_parts.append(
                    item.export_to_document_tokens(
                        doc=self,
                        new_line=delim,
                        xsize=xsize,
                        ysize=ysize,
                        add_location=add_location,
                        add_content=add_content,
                    )
                )
            elif isinstance(item, TextItem):
                output_parts.append(
                    item.export_to_document_tokens(
                        doc=self,
                        new_line=delim,
                        xsize=xsize,
                        ysize=ysize,
                        add_location=add_location,
                        add_content=add_content,
                    )
                )
            elif isinstance(item, TableItem):
                output_parts.append(
                    item.export_to_document_tokens(
                        doc=self,
                        new_line=delim,
                        xsize=xsize,
                        ysize=ysize,
                        add_location=add_location,
                        add_cell_location=add_table_cell_location,
                        add_cell_text=add_table_cell_text,
                        add_caption=True,
                    )
                )
            elif isinstance(item, PictureItem):
                output_parts.append(
                    item.export_to_document_tokens(
                        doc=self,
                        new_line=delim,
                        xsize=xsize,
                        ysize=ysize,
                        add_caption=True,
                        add_location=add_location,
                        add_content=add_content,
                    )
                )

        # End any lists that might still be open
        ordered_list_stack = _close_lists(
            0, previous_level, ordered_list_stack, output_parts
        )

        # End document
        output_parts.append(f"</{DocumentToken.DOCUMENT.value}>")

        return "".join(output_parts)

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
                                text = get_text(text=cell.text, max_text_len=16)
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

    @field_validator("version")
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this document version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, CURRENT_VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(
                f"incompatible version {v} with schema version {CURRENT_VERSION}"
            )
        else:
            return CURRENT_VERSION

    @model_validator(mode="after")  # type: ignore
    @classmethod
    def validate_document(cls, d: "DoclingDocument"):
        """validate_document."""
        if not d.validate_tree(d.body) or not d.validate_tree(d.furniture):
            raise ValueError("Document hierachy is inconsistent.")

        return d
