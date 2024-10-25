"""Models for the Docling Document data type."""

import base64
import mimetypes
import re
import sys
import typing
from io import BytesIO
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union

import pandas as pd
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
from docling_core.types.doc.labels import DocItemLabel, GroupLabel
from docling_core.types.legacy_doc.tokens import DocumentToken

Uint64 = typing.Annotated[int, Field(ge=0, le=(2**64 - 1))]
LevelNumber = typing.Annotated[int, Field(ge=1, le=100)]
CURRENT_VERSION: Final = "1.0.0"

DEFAULT_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.CAPTION,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
}


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


PictureDataType = Annotated[
    Union[
        PictureClassificationData,
        PictureDescriptionData,
        PictureMoleculeData,
        PictureMiscData,
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
    uri: AnyUrl
    _pil: Optional[PILImage.Image] = None

    @property
    def pil_image(self) -> PILImage.Image:
        """Return the PIL Image."""
        if self._pil is not None:
            return self._pil

        if str(self.uri).startswith("data:"):
            encoded_img = str(self.uri).split(",")[1]
            decoded_img = base64.b64decode(encoded_img)
            self._pil = PILImage.open(BytesIO(decoded_img))
        else:
            self._pil = PILImage.open(str(self.uri))

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


class NodeItem(BaseModel):
    """NodeItem."""

    self_ref: str = Field(pattern=_JSON_POINTER_REGEX)
    parent: Optional[RefItem] = None
    children: List[RefItem] = []

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
        xsize: int = 100,
        ysize: int = 100,
        add_page_index: bool = True,
    ) -> str:
        """Get the location string for the BaseCell."""
        if not len(self.prov):
            return ""

        location = ""
        for prov in self.prov:
            page_w, page_h = doc.pages[prov.page_no].size.as_tuple()

            page_i = -1
            if add_page_index:
                page_i = prov.page_no

            loc_str = DocumentToken.get_location(
                bbox=prov.bbox.to_bottom_left_origin(page_h).as_tuple(),
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
                page_i=page_i,
            )
            location += f"{loc_str}{new_line}"

        return location


class TextItem(DocItem):
    """TextItem."""

    orig: str  # untreated representation
    text: str  # sanitized representation

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "\n",
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str:  (Default value = "\n")
        :param xsize: int:  (Default value = 100)
        :param ysize: int:  (Default value = 100)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param add_page_index: bool:  (Default value = True)

        """
        body = f"<{self.label.value}>"

        # TODO: This must be done through an explicit mapping.
        # assert DocumentToken.is_known_token(
        #    body
        # ), f"failed DocumentToken.is_known_token({body})"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line="",
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_content and self.text is not None:
            body += self.text.strip()

        body += f"</{self.label.value}>{new_line}"

        return body


class SectionHeaderItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.SECTION_HEADER] = DocItemLabel.SECTION_HEADER
    level: LevelNumber


class ListItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.LIST_ITEM] = DocItemLabel.LIST_ITEM
    enumerated: bool = False
    marker: str  # The bullet or number symbol that prefixes this list item


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


class PictureItem(FloatingItem):
    """PictureItem."""

    label: typing.Literal[DocItemLabel.PICTURE] = DocItemLabel.PICTURE

    annotations: List[PictureDataType] = []

    def export_to_document_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "\n",
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,  # not used at the moment
        add_page_index: bool = True,
    ):
        r"""Export picture to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str:  (Default value = "\n")
        :param xsize: int:  (Default value = 100)
        :param ysize: int:  (Default value = 100)
        :param add_location: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param # not used at the momentadd_page_index: bool:  (Default value = True)

        """
        body = f"{DocumentToken.BEG_FIGURE.value}{new_line}"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_caption and len(self.captions):
            text = self.caption_text(doc)

            if len(text):
                body += f"{DocumentToken.BEG_CAPTION.value}"
                body += f"{text.strip()}"
                body += f"{DocumentToken.END_CAPTION.value}"
                body += f"{new_line}"

        body += f"{DocumentToken.END_FIGURE.value}{new_line}"

        return body


class TableItem(FloatingItem):
    """TableItem."""

    data: TableData
    label: typing.Literal[DocItemLabel.TABLE] = DocItemLabel.TABLE

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

    def export_to_html(self) -> str:
        """Export the table as html."""
        body = ""
        nrows = self.data.num_rows
        ncols = self.data.num_cols

        if not len(self.data.table_cells):
            return ""
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

                content = cell.text.strip()
                celltag = "td"
                if cell.column_header:
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
        doc: "DoclingDocument",
        new_line: str = "\n",
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
        r"""Export table to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str:  (Default value = "\n")
        :param xsize: int:  (Default value = 100)
        :param ysize: int:  (Default value = 100)
        :param add_location: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param add_cell_location: bool:  (Default value = True)
        :param add_cell_label: bool:  (Default value = True)
        :param add_cell_text: bool:  (Default value = True)
        :param add_page_index: bool:  (Default value = True)

        """
        body = f"{DocumentToken.BEG_TABLE.value}{new_line}"

        if add_location:
            body += self.get_location_tokens(
                doc=doc,
                new_line=new_line,
                xsize=xsize,
                ysize=ysize,
                add_page_index=add_page_index,
            )

        if add_caption and len(self.captions):
            text = self.caption_text(doc)

            if len(text):
                body += f"{DocumentToken.BEG_CAPTION.value}"
                body += f"{text.strip()}"
                body += f"{DocumentToken.END_CAPTION.value}"
                body += f"{new_line}"

        if add_content and len(self.data.table_cells) > 0:
            for i, row in enumerate(self.data.grid):
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
                        and len(self.prov) > 0
                    ):
                        page_w, page_h = doc.pages[self.prov[0].page_no].size.as_tuple()
                        cell_loc = DocumentToken.get_location(
                            bbox=col.bbox.to_bottom_left_origin(page_h).as_tuple(),
                            page_w=page_w,
                            page_h=page_h,
                            xsize=xsize,
                            ysize=ysize,
                            page_i=self.prov[0].page_no,
                        )
                    elif (
                        col.bbox is not None
                        and add_cell_location
                        and not add_page_index
                        and len(self.prov) > 0
                    ):
                        page_w, page_h = doc.pages[self.prov[0].page_no].size.as_tuple()

                        cell_loc = DocumentToken.get_location(
                            bbox=col.bbox.to_bottom_left_origin(page_h).as_tuple(),
                            page_w=page_w,
                            page_h=page_h,
                            xsize=xsize,
                            ysize=ysize,
                            page_i=-1,
                        )

                    cell_label = ""
                    if add_cell_label:
                        if col.column_header:
                            cell_label = "<col_header>"
                        elif col.row_header:
                            cell_label = "<row_header>"
                        elif col.row_section:
                            cell_label = "<row_section>"
                        else:
                            cell_label = "<body>"

                    body += f"<col_{j}>{cell_loc}{cell_label}{text}</col_{j}>"

                body += f"</row_{i}>{new_line}"

        body += f"{DocumentToken.END_TABLE.value}{new_line}"

        return body


class KeyValueItem(DocItem):
    """KeyValueItem."""


ContentItem = Union[
    TextItem, SectionHeaderItem, ListItem, PictureItem, TableItem, KeyValueItem
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

    furniture: GroupItem = GroupItem(
        name="_root_", self_ref="#/furniture"
    )  # List[RefItem] = []
    body: GroupItem = GroupItem(name="_root_", self_ref="#/body")  # List[RefItem] = []

    groups: List[GroupItem] = []
    texts: List[Union[SectionHeaderItem, ListItem, TextItem]] = []
    pictures: List[PictureItem] = []
    tables: List[TableItem] = []
    key_value_items: List[KeyValueItem] = []

    pages: Dict[int, PageItem] = {}  # empty as default

    def add_group(
        self,
        label: Optional[GroupLabel] = None,
        name: Optional[str] = None,
        parent: Optional[GroupItem] = None,
    ) -> GroupItem:
        """add_group.

        :param label: Optional[GroupLabel]:  (Default value = None)
        :param name: Optional[str]:  (Default value = None)
        :param parent: Optional[GroupItem]:  (Default value = None)

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
        parent: Optional[GroupItem] = None,
    ):
        """add_paragraph.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[GroupItem]:  (Default value = None)

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

        self.texts.append(list_item)
        parent.children.append(RefItem(cref=cref))

        return list_item

    def add_text(
        self,
        label: DocItemLabel,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[GroupItem] = None,
    ):
        """add_paragraph.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[GroupItem]:  (Default value = None)

        """
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

        self.texts.append(text_item)
        parent.children.append(RefItem(cref=cref))

        return text_item

    def add_table(
        self,
        data: TableData,
        caption: Optional[Union[TextItem, RefItem]] = None,  # This is not cool yet.
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[GroupItem] = None,
    ):
        """add_table.

        :param data: BaseTableData:
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param # This is not cool yet.prov: Optional[ProvenanceItem]
        :param parent: Optional[GroupItem]:  (Default value = None)

        """
        if not parent:
            parent = self.body

        table_index = len(self.tables)
        cref = f"#/tables/{table_index}"

        tbl_item = TableItem(
            label=DocItemLabel.TABLE, data=data, self_ref=cref, parent=parent.get_ref()
        )
        if prov:
            tbl_item.prov.append(prov)
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
        parent: Optional[GroupItem] = None,
    ):
        """add_picture.

        :param data: List[PictureData]: (Default value = [])
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[GroupItem]:  (Default value = None)

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
        if caption:
            fig_item.captions.append(caption.get_ref())

        self.pictures.append(fig_item)
        parent.children.append(RefItem(cref=cref))

        return fig_item

    def add_heading(
        self,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[GroupItem] = None,
    ):
        """add_heading.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[GroupItem]:  (Default value = None)

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

        self.texts.append(section_header_item)
        parent.children.append(RefItem(cref=cref))

        return section_header_item

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
        traverse_pictures: bool = True,
        page_no: Optional[int] = None,
        _level: int = 0,  # fixed parameter, carries through the node nesting level
    ) -> typing.Iterable[Tuple[NodeItem, int]]:  # tuple of node and level
        """iterate_elements.

        :param root: Optional[NodeItem]:  (Default value = None)
        :param with_groups: bool:  (Default value = False)
        :param traverse_pictures: bool:  (Default value = True)
        :param page_no: Optional[int]:  (Default value = None)
        :param _level:  (Default value = 0)
        :param # fixed parameter:
        :param carries through the node nesting level:
        """
        if not root:
            root = self.body

        if not isinstance(root, GroupItem) or with_groups:
            if isinstance(root, DocItem):
                if page_no is not None:
                    for prov in root.prov:
                        if prov.page_no == page_no:
                            yield root, _level
                else:
                    yield root, _level
            else:
                yield root, _level

        # Traverse children
        for child_ref in root.children:
            child = child_ref.resolve(self)

            if isinstance(child, NodeItem):
                # If the child is a NodeItem, recursively traverse it
                if not isinstance(child, PictureItem) or traverse_pictures:
                    yield from self.iterate_items(
                        child, _level=_level + 1, with_groups=with_groups
                    )

    def print_element_tree(self):
        """print_element_tree."""
        for ix, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                print(" " * level, f"{ix}: {item.label.value} with name={item.name}")
            elif isinstance(item, DocItem):
                print(" " * level, f"{ix}: {item.label.value}")

    def export_to_dict(self) -> Dict:
        """export_to_dict."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def export_to_markdown(  # noqa: C901
        self,
        delim: str = "\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        strict_text: bool = False,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
    ) -> str:
        r"""Serialize to Markdown.

        Operates on a slice of the document's main_text as defined through arguments
        main_text_start and main_text_stop; defaulting to the whole main_text.

        :param delim: Delimiter to use when concatenating the various
                Markdown parts. Defaults to "\n\n".
        :type delim: str
        :param from_element: Body slicing start index (inclusive).
                Defaults to 0.
        :type from_element: int
        :param to_element: Body slicing stop index
                (exclusive). Defaults to None.
        :type to_element: Optional[int]
        :param delim: str:  (Default value = "\n\n")
        :param from_element: int:  (Default value = 0)
        :param to_element: Optional[int]:  (Default value = None)
        :param labels: set[DocItemLabel]
        :param "subtitle-level-1":
        :param "paragraph":
        :param "caption":
        :param "table":
        :param "Text":
        :param "text":
        :param ]:
        :param strict_text: bool:  (Default value = False)
        :param image_placeholder str:  (Default value = "<!-- image -->")
            the placeholder to include to position images in the markdown.
        :param indent: int (default=4): indent of the nested lists
        :returns: The exported Markdown representation.
        :rtype: str
        """
        mdtexts: list[str] = []
        list_nesting_level = 0  # Track the current list nesting level
        previous_level = 0  # Track the previous item's level
        in_list = False  # Track if we're currently processing list items

        for ix, (item, level) in enumerate(
            self.iterate_items(self.body, with_groups=True)
        ):
            # If we've moved to a lower level, we're exiting one or more groups
            if level < previous_level:
                # Calculate how many levels we've exited
                level_difference = previous_level - level
                # Decrement list_nesting_level for each list group we've exited
                list_nesting_level = max(0, list_nesting_level - level_difference)

            previous_level = level  # Update previous_level for next iteration

            if ix < from_element and to_element <= ix:
                continue  # skip as many items as you want

            # Handle newlines between different types of content
            if (
                len(mdtexts) > 0
                and not isinstance(item, (ListItem, GroupItem))
                and in_list
            ):
                mdtexts[-1] += "\n"
                in_list = False

            if isinstance(item, GroupItem) and item.label in [
                GroupLabel.LIST,
                GroupLabel.ORDERED_LIST,
            ]:

                if list_nesting_level == 0:  # Check if we're on the top level.
                    # In that case a new list starts directly after another list.
                    mdtexts.append("\n")  # Add a blank line

                # Increment list nesting level when entering a new list
                list_nesting_level += 1
                in_list = True
                continue

            elif isinstance(item, GroupItem):
                continue

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                in_list = False
                marker = "" if strict_text else "#"
                text = f"{marker} {item.text}\n"
                mdtexts.append(text.strip())

            elif (
                isinstance(item, TextItem)
                and item.label in [DocItemLabel.SECTION_HEADER]
            ) or isinstance(item, SectionHeaderItem):
                in_list = False
                marker = ""
                if not strict_text:
                    marker = "#" * level
                    if len(marker) < 2:
                        marker = "##"
                text = f"{marker} {item.text}\n"
                mdtexts.append(text.strip() + "\n")

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:
                in_list = False
                text = f"```\n{item.text}\n```\n"
                mdtexts.append(text)

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CAPTION]:
                # captions are printed in picture and table ... skipping for now
                continue

            elif isinstance(item, ListItem) and item.label in [DocItemLabel.LIST_ITEM]:
                in_list = True
                # Calculate indent based on list_nesting_level
                # -1 because level 1 needs no indent
                list_indent = " " * (indent * (list_nesting_level - 1))

                marker = ""
                if strict_text:
                    marker = ""
                elif item.enumerated:
                    marker = item.marker
                else:
                    marker = "-"  # Markdown needs only dash as item marker.

                text = f"{list_indent}{marker} {item.text}"
                mdtexts.append(text)

            elif isinstance(item, TextItem) and item.label in labels:
                in_list = False
                if len(item.text):
                    text = f"{item.text}\n"
                    mdtexts.append(text)

            elif isinstance(item, TableItem) and not strict_text:
                in_list = False
                mdtexts.append(item.caption_text(self))
                md_table = item.export_to_markdown()
                mdtexts.append("\n" + md_table + "\n")

            elif isinstance(item, PictureItem) and not strict_text:
                in_list = False
                mdtexts.append(item.caption_text(self))

                if image_mode == ImageRefMode.PLACEHOLDER:
                    mdtexts.append("\n" + image_placeholder + "\n")
                elif image_mode == ImageRefMode.EMBEDDED and isinstance(
                    item.image, ImageRef
                ):
                    text = f"![Local Image]({item.image.uri})\n"
                    mdtexts.append(text)
                elif image_mode == ImageRefMode.EMBEDDED and not isinstance(
                    item.image, ImageRef
                ):
                    text = (
                        "<!-- 🖼️❌ Image not available. "
                        "Please use `PdfPipelineOptions(generate_picture_images=True)`"
                        " --> "
                    )
                    mdtexts.append(text)

            elif isinstance(item, DocItem) and item.label in labels:
                in_list = False
                text = "<missing-text>"
                mdtexts.append(text)

        mdtext = (delim.join(mdtexts)).strip()
        mdtext = re.sub(
            r"\n\n\n+", "\n\n", mdtext
        )  # remove cases of double or more empty lines.

        # Our export markdown doesn't contain any emphasis styling:
        # Bold, Italic, or Bold-Italic
        # Hence, any underscore that we print into Markdown is coming from document text
        # That means we need to escape it, to properly reflect content in the markdown
        def escape_underscores(text):
            # Replace "_" with "\_" only if it's not already escaped
            escaped_text = re.sub(r"(?<!\\)_", r"\_", text)
            return escaped_text

        mdtext = escape_underscores(mdtext)

        return mdtext

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
            image_placeholder="",
        )

    def export_to_document_tokens(
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: Optional[int] = None,
        labels: set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
        xsize: int = 100,
        ysize: int = 100,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_label: bool = True,
        add_table_cell_text: bool = True,
    ) -> str:
        r"""Exports the document content to an DocumentToken format.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole main_text.

        :param delim: str:  (Default value = "\n\n")
        :param from_element: int:  (Default value = 0)
        :param to_element: Optional[int]:  (Default value = None)
        :param labels: set[DocItemLabel]
        :param xsize: int:  (Default value = 100)
        :param ysize: int:  (Default value = 100)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param add_page_index: bool:  (Default value = True)
        :param # table specific flagsadd_table_cell_location: bool
        :param add_table_cell_label: bool:  (Default value = True)
        :param add_table_cell_text: bool:  (Default value = True)
        :returns: The content of the document formatted as a DocTags string.
        :rtype: str
        """
        new_line = ""
        if delim:
            new_line = "\n"

        doctags = f"{DocumentToken.BEG_DOCUMENT.value}{new_line}"

        # pagedims = self.get_map_to_page_dimensions()

        skip_count = 0
        for ix, (item, level) in enumerate(self.iterate_items(self.body)):
            if skip_count < from_element:
                skip_count += 1
                continue  # skip as many items as you want

            if to_element and ix >= to_element:
                break

            if not isinstance(item, DocItem):
                continue

            prov = item.prov

            page_i = -1

            if add_location and len(self.pages) and len(prov) > 0:

                page_i = prov[0].page_no
                page_dim = self.pages[page_i].size

                float(page_dim.width)
                float(page_dim.height)

            item_type = item.label
            if isinstance(item, TextItem) and (item_type in labels):

                doctags += item.export_to_document_tokens(
                    doc=self,
                    new_line=new_line,
                    xsize=xsize,
                    ysize=ysize,
                    add_location=add_location,
                    add_content=add_content,
                    add_page_index=add_page_index,
                )

            elif isinstance(item, TableItem) and (item_type in labels):

                doctags += item.export_to_document_tokens(
                    doc=self,
                    new_line=new_line,
                    xsize=xsize,
                    ysize=ysize,
                    add_caption=True,
                    add_location=add_location,
                    add_content=add_content,
                    add_cell_location=add_table_cell_location,
                    add_cell_label=add_table_cell_label,
                    add_cell_text=add_table_cell_text,
                    add_page_index=add_page_index,
                )

            elif isinstance(item, PictureItem) and (item_type in labels):

                doctags += item.export_to_document_tokens(
                    doc=self,
                    new_line=new_line,
                    xsize=xsize,
                    ysize=ysize,
                    add_caption=True,
                    add_location=add_location,
                    add_content=add_content,
                    add_page_index=add_page_index,
                )

        doctags += DocumentToken.END_DOCUMENT.value

        return doctags

    def _export_to_indented_text(
        self, indent="  ", max_text_len: int = -1, explicit_tables: bool = False
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

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CAPTION]:
                # captions are printed in picture and table ... skipping for now
                continue

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
