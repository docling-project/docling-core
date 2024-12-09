#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utilities for converting between legacy and new document format."""

import hashlib
import uuid
from typing import Union

from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableCell,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import ListItem
from docling_core.types.legacy_doc.base import (
    BaseCell,
    BaseText,
    Figure,
    GlmTableCell,
    PageDimensions,
    PageReference,
    Prov,
    Ref,
)
from docling_core.types.legacy_doc.base import Table as DsSchemaTable
from docling_core.types.legacy_doc.base import TableCell as DsTableCell
from docling_core.types.legacy_doc.document import (
    CCSDocumentDescription as DsDocumentDescription,
)
from docling_core.types.legacy_doc.document import CCSFileInfoObject as DsFileInfoObject
from docling_core.types.legacy_doc.document import ExportedCCSDocument as DsDocument


def _create_hash(string: str):
    hasher = hashlib.sha256()
    hasher.update(string.encode("utf-8"))

    return hasher.hexdigest()


def doc_item_label_to_legacy_type(label: DocItemLabel):
    """Convert the DocItemLabel to the legacy type."""
    _label_to_ds_type = {
        DocItemLabel.TITLE: "title",
        DocItemLabel.DOCUMENT_INDEX: "table-of-contents",
        DocItemLabel.SECTION_HEADER: "subtitle-level-1",
        DocItemLabel.CHECKBOX_SELECTED: "checkbox-selected",
        DocItemLabel.CHECKBOX_UNSELECTED: "checkbox-unselected",
        DocItemLabel.CAPTION: "caption",
        DocItemLabel.PAGE_HEADER: "page-header",
        DocItemLabel.PAGE_FOOTER: "page-footer",
        DocItemLabel.FOOTNOTE: "footnote",
        DocItemLabel.TABLE: "table",
        DocItemLabel.FORMULA: "equation",
        DocItemLabel.LIST_ITEM: "paragraph",
        DocItemLabel.CODE: "paragraph",
        DocItemLabel.PICTURE: "figure",
        DocItemLabel.TEXT: "paragraph",
        DocItemLabel.PARAGRAPH: "paragraph",
    }
    if label in _label_to_ds_type:
        return _label_to_ds_type[label]
    return label.value


def doc_item_label_to_legacy_name(label: DocItemLabel):
    """Convert the DocItemLabel to the legacy name."""
    _reverse_label_name_mapping = {
        DocItemLabel.CAPTION: "Caption",
        DocItemLabel.FOOTNOTE: "Footnote",
        DocItemLabel.FORMULA: "Formula",
        DocItemLabel.LIST_ITEM: "List-item",
        DocItemLabel.PAGE_FOOTER: "Page-footer",
        DocItemLabel.PAGE_HEADER: "Page-header",
        DocItemLabel.PICTURE: "Picture",
        DocItemLabel.SECTION_HEADER: "Section-header",
        DocItemLabel.TABLE: "Table",
        DocItemLabel.TEXT: "Text",
        DocItemLabel.TITLE: "Title",
        DocItemLabel.DOCUMENT_INDEX: "Document Index",
        DocItemLabel.CODE: "Code",
        DocItemLabel.CHECKBOX_SELECTED: "Checkbox-Selected",
        DocItemLabel.CHECKBOX_UNSELECTED: "Checkbox-Unselected",
        DocItemLabel.FORM: "Form",
        DocItemLabel.KEY_VALUE_REGION: "Key-Value Region",
        DocItemLabel.PARAGRAPH: "paragraph",
    }
    if label in _reverse_label_name_mapping:
        return _reverse_label_name_mapping[label]
    return label.value


def docling_document_to_legacy(doc: DoclingDocument, fallback_filaname: str = "file"):
    """Convert a DoclingDocument to the legacy format."""
    title = ""
    desc: DsDocumentDescription = DsDocumentDescription(logs=[])

    if doc.origin is not None:
        document_hash = _create_hash(str(doc.origin.binary_hash))
        filename = doc.origin.filename
    else:
        document_hash = _create_hash(str(uuid.uuid4()))
        filename = fallback_filaname

    page_hashes = [
        PageReference(
            hash=_create_hash(document_hash + ":" + str(p.page_no - 1)),
            page=p.page_no,
            model="default",
        )
        for p in doc.pages.values()
    ]

    file_info = DsFileInfoObject(
        filename=filename,
        document_hash=document_hash,
        num_pages=len(doc.pages),
        page_hashes=page_hashes,
    )

    main_text: list[Union[Ref, BaseText]] = []
    tables: list[DsSchemaTable] = []
    figures: list[Figure] = []
    equations: list[BaseCell] = []
    footnotes: list[BaseText] = []
    page_headers: list[BaseText] = []
    page_footers: list[BaseText] = []

    # TODO: populate page_headers page_footers from doc.furniture

    embedded_captions = set()
    for ix, (item, level) in enumerate(doc.iterate_items(doc.body)):

        if isinstance(item, (TableItem, PictureItem)) and len(item.captions) > 0:
            caption = item.caption_text(doc)
            if caption:
                embedded_captions.add(caption)

    for item, level in doc.iterate_items():
        if isinstance(item, DocItem):
            item_type = item.label

            if isinstance(item, (TextItem, ListItem, SectionHeaderItem)):

                if isinstance(item, ListItem) and item.marker:
                    text = f"{item.marker} {item.text}"
                else:
                    text = item.text

                # Can be empty.
                prov = [
                    Prov(
                        bbox=p.bbox.as_tuple(),
                        page=p.page_no,
                        span=[0, len(item.text)],
                    )
                    for p in item.prov
                ]
                main_text.append(
                    BaseText(
                        text=text,
                        obj_type=doc_item_label_to_legacy_type(item.label),
                        name=doc_item_label_to_legacy_name(item.label),
                        prov=prov,
                    )
                )

                # skip captions of they are embedded in the actual
                # floating object
                if item_type == DocItemLabel.CAPTION and text in embedded_captions:
                    continue

            elif isinstance(item, TableItem) and item.data:
                index = len(tables)
                ref_str = f"#/tables/{index}"
                main_text.append(
                    Ref(
                        name=doc_item_label_to_legacy_name(item.label),
                        obj_type=doc_item_label_to_legacy_type(item.label),
                        ref=ref_str,
                    ),
                )

                # Initialise empty table data grid (only empty cells)
                table_data = [
                    [
                        DsTableCell(
                            text="",
                            # bbox=[0,0,0,0],
                            spans=[[i, j]],
                            obj_type="body",
                        )
                        for j in range(item.data.num_cols)
                    ]
                    for i in range(item.data.num_rows)
                ]

                # Overwrite cells in table data for which there is actual cell content.
                for cell in item.data.table_cells:
                    for i in range(
                        min(cell.start_row_offset_idx, item.data.num_rows),
                        min(cell.end_row_offset_idx, item.data.num_rows),
                    ):
                        for j in range(
                            min(cell.start_col_offset_idx, item.data.num_cols),
                            min(cell.end_col_offset_idx, item.data.num_cols),
                        ):
                            celltype = "body"
                            if cell.column_header:
                                celltype = "col_header"
                            elif cell.row_header:
                                celltype = "row_header"
                            elif cell.row_section:
                                celltype = "row_section"

                            def _make_spans(cell: TableCell, table_item: TableItem):
                                for rspan in range(
                                    min(
                                        cell.start_row_offset_idx,
                                        table_item.data.num_rows,
                                    ),
                                    min(
                                        cell.end_row_offset_idx,
                                        table_item.data.num_rows,
                                    ),
                                ):
                                    for cspan in range(
                                        min(
                                            cell.start_col_offset_idx,
                                            table_item.data.num_cols,
                                        ),
                                        min(
                                            cell.end_col_offset_idx,
                                            table_item.data.num_cols,
                                        ),
                                    ):
                                        yield [rspan, cspan]

                            spans = list(_make_spans(cell, item))
                            table_data[i][j] = GlmTableCell(
                                text=cell.text,
                                bbox=(
                                    cell.bbox.as_tuple()
                                    if cell.bbox is not None
                                    else None
                                ),  # check if this is bottom-left
                                spans=spans,
                                obj_type=celltype,
                                col=j,
                                row=i,
                                row_header=cell.row_header,
                                row_section=cell.row_section,
                                col_header=cell.column_header,
                                row_span=[
                                    cell.start_row_offset_idx,
                                    cell.end_row_offset_idx,
                                ],
                                col_span=[
                                    cell.start_col_offset_idx,
                                    cell.end_col_offset_idx,
                                ],
                            )

                # Compute the caption
                caption = item.caption_text(doc)

                tables.append(
                    DsSchemaTable(
                        text=caption,
                        num_cols=item.data.num_cols,
                        num_rows=item.data.num_rows,
                        obj_type=doc_item_label_to_legacy_type(item.label),
                        data=table_data,
                        prov=[
                            Prov(
                                bbox=p.bbox.as_tuple(),
                                page=p.page_no,
                                span=[0, 0],
                            )
                            for p in item.prov
                        ],
                    )
                )

            elif isinstance(item, PictureItem):
                index = len(figures)
                ref_str = f"#/figures/{index}"
                main_text.append(
                    Ref(
                        name=doc_item_label_to_legacy_name(item.label),
                        obj_type=doc_item_label_to_legacy_type(item.label),
                        ref=ref_str,
                    ),
                )

                # Compute the caption
                caption = item.caption_text(doc)

                figures.append(
                    Figure(
                        prov=[
                            Prov(
                                bbox=p.bbox.as_tuple(),
                                page=p.page_no,
                                span=[0, len(caption)],
                            )
                            for p in item.prov
                        ],
                        obj_type=doc_item_label_to_legacy_type(item.label),
                        text=caption,
                        # data=[[]],
                    )
                )

    page_dimensions = [
        PageDimensions(page=p.page_no, height=p.size.height, width=p.size.width)
        for p in doc.pages.values()
    ]

    legacy_doc: DsDocument = DsDocument(
        name=title,
        description=desc,
        file_info=file_info,
        main_text=main_text,
        equations=equations,
        footnotes=footnotes,
        page_headers=page_headers,
        page_footers=page_footers,
        tables=tables,
        figures=figures,
        page_dimensions=page_dimensions,
    )

    return legacy_doc


# def legacy_to_docling_document(legacy_doc: DsDocument) -> DoclingDocument:
#     """Convert a legacy document to DoclingDocument."""
