"""Table item and its annotation-type union."""

import logging
import typing
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Optional, Union

import pandas as pd
from pydantic import Field, model_validator, validate_call
from tabulate import _column_type, tabulate
from typing_extensions import Self, deprecated

from docling_core.types.doc.common.annotations import BaseAnnotation, DescriptionAnnotation, MiscAnnotation
from docling_core.types.doc.common.meta import DescriptionMetaField, FloatingMeta, MetaUtils
from docling_core.types.doc.items.node import FloatingItem
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.tokens import DocumentToken, TableToken

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_logger = logging.getLogger(__name__)


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

    annotations: Annotated[
        list[TableAnnotationType],
        deprecated("Field `annotations` is deprecated; use `meta` instead."),
    ] = []

    @model_validator(mode="after")
    def _migrate_annotations_to_meta(self) -> Self:
        """Migrate the `annotations` field to `meta`."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

            if self.annotations:
                _logger.info(
                    "Migrating deprecated `annotations` to `meta`; this will be removed in the future. "
                    "Note that only the first available instance of each annotation type will be migrated."
                )
                for ann in self.annotations:
                    # ensure meta field is present
                    if self.meta is None:
                        self.meta = FloatingMeta()

                    if isinstance(ann, DescriptionAnnotation) and self.meta.description is None:
                        self.meta.description = DescriptionMetaField(
                            text=ann.text,
                            created_by=ann.provenance,
                        )
                    elif not isinstance(ann, DescriptionAnnotation) and not hasattr(
                        self.meta,
                        MetaUtils.create_meta_field_name(
                            namespace=MetaUtils._META_FIELD_LEGACY_NAMESPACE,
                            name=ann.kind,
                        ),
                    ):
                        self.meta.set_custom_field(
                            namespace=MetaUtils._META_FIELD_LEGACY_NAMESPACE,
                            name=ann.kind,
                            value=(ann.content if isinstance(ann, MiscAnnotation) else ann.model_dump(mode="json")),
                        )

            return self

    def export_to_dataframe(self, doc: Optional["DoclingDocument"] = None) -> pd.DataFrame:
        """Export the table as a Pandas DataFrame."""
        return self._export_to_dataframe_with_options(doc=doc)

    def _export_to_dataframe_with_options(
        self,
        doc: Optional["DoclingDocument"] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Export the table as a Pandas DataFrame with contextual named arguments."""
        if doc is None:
            _logger.warning("Usage of TableItem.export_to_dataframe() without `doc` argument is deprecated.")

        if self.data.num_rows == 0 or self.data.num_cols == 0:
            return pd.DataFrame()

        # Count how many rows are column headers
        num_headers = 0
        for i, row in enumerate(self.data.grid):
            if len(row) == 0:
                raise RuntimeError(f"Invalid table. {len(row)=} but {self.data.num_cols=}.")

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
        columns: Optional[list[str]] = None
        if num_headers > 0:
            columns = ["" for _ in range(self.data.num_cols)]
            for i in range(num_headers):
                for j, cell in enumerate(self.data.grid[i]):
                    col_name = cell._get_text(doc=doc, **kwargs)
                    if columns[j] != "":
                        col_name = f".{col_name}"
                    columns[j] += col_name

        # Create table data
        table_data = [[cell._get_text(doc=doc, **kwargs) for cell in row] for row in self.data.grid[num_headers:]]

        # Create DataFrame
        table = pd.DataFrame(table_data, columns=columns)

        return table

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
                "Usage of TableItem.export_to_markdown() without `doc` argument is deprecated.",
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
                # Always disable numparse to prevent silent precision loss in numeric values
                # Use tabulate's _column_type to detect numeric columns for right-alignment
                colalign = []
                num_cols = len(table[0])
                for col_idx in range(num_cols):
                    col_values = [row[col_idx] if col_idx < len(row) else "" for row in table[1:]]
                    col_type = _column_type(col_values)
                    colalign.append("right" if col_type in (int, float) else "left")

                res = tabulate(
                    table[1:],
                    headers=table[0],
                    tablefmt="github",
                    disable_numparse=True,
                    colalign=tuple(colalign) if colalign else None,
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
                "Usage of TableItem.export_to_html() without `doc` argument is deprecated.",
            )
            return ""

    def export_to_otsl(
        self,
        doc: "DoclingDocument",
        add_cell_location: bool = True,
        add_cell_text: bool = True,
        xsize: int = 500,
        ysize: int = 500,
        self_closing: bool = False,
        **kwargs: Any,
    ) -> str:
        """Export the table as OTSL."""
        # Possible OTSL tokens...
        #
        # Empty and full cells:
        # "ecel", "fcel"
        #
        # Cell spans (horizontal, vertical, 2d):
        # "lcel", "ucel", "xcel"
        #
        # New line:
        # "nl"
        #
        # Headers (column, row, section row):
        # "ched", "rhed", "srow"

        from docling_core.transforms.serializer.doctags import DocTagsDocSerializer

        table_token = kwargs.get("table_token", TableToken)

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
                content = cell._get_text(doc=doc, doc_serializer=doc_serializer, **kwargs).strip()
                rowspan, rowstart = (
                    cell.row_span,
                    cell.start_row_offset_idx,
                )
                colspan, colstart = (
                    cell.col_span,
                    cell.start_col_offset_idx,
                )

                has_page_info = page_no in doc.pages

                cell_loc = ""
                if cell.bbox is not None and has_page_info:
                    page_w, page_h = doc.pages[page_no].size.as_tuple()
                    cell_loc = DocumentToken.get_location(
                        bbox=cell.bbox.to_bottom_left_origin(page_h).as_tuple(),
                        page_w=page_w,
                        page_h=page_h,
                        xsize=xsize,
                        ysize=ysize,
                        self_closing=self_closing,
                    )

                if rowstart == i and colstart == j:
                    if len(content) > 0:
                        if cell.column_header:
                            body.append(str(table_token.OTSL_CHED.value))
                        elif cell.row_header:
                            body.append(str(table_token.OTSL_RHED.value))
                        elif cell.row_section:
                            body.append(str(table_token.OTSL_SROW.value))
                        else:
                            body.append(str(table_token.OTSL_FCEL.value))
                        if add_cell_location:
                            body.append(str(cell_loc))
                        if add_cell_text:
                            body.append(str(content))
                    else:
                        body.append(str(table_token.OTSL_ECEL.value))
                else:
                    add_cross_cell = False
                    if rowstart != i:
                        if colspan == 1:
                            body.append(str(table_token.OTSL_UCEL.value))
                        else:
                            add_cross_cell = True
                    if colstart != j:
                        if rowspan == 1:
                            body.append(str(table_token.OTSL_LCEL.value))
                        else:
                            add_cross_cell = True
                    if add_cross_cell:
                        body.append(str(table_token.OTSL_XCEL.value))
            body.append(str(table_token.OTSL_NL.value))
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
