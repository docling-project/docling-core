#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utilities for converting between legacy and new document format."""

import hashlib
import uuid
from pathlib import Path
from typing import Dict, Optional, Union

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    Size,
    TableCell,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import ContentLayer, GroupItem, ListItem, TableData
from docling_core.types.doc.labels import GroupLabel
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


def legacy_to_docling_document(legacy_doc: DsDocument) -> DoclingDocument:  # noqa: C901
    """Convert a legacy document to DoclingDocument.

    It is known that the following content will not be preserved in the transformation:
    - name of labels (upper vs lower case)
    - caption of figures are not in main-text anymore
    - s3_data removed
    - model metadata removed
    - logs removed
    - document hash cannot be preserved
    """

    def _transform_prov(item: BaseCell) -> Optional[ProvenanceItem]:
        """Create a new provenance from a legacy item."""
        prov: Optional[ProvenanceItem] = None
        if item.prov is not None and len(item.prov) > 0:
            prov = ProvenanceItem(
                page_no=int(item.prov[0].page),
                charspan=tuple(item.prov[0].span),
                bbox=BoundingBox.from_tuple(
                    tuple(item.prov[0].bbox), origin=CoordOrigin.BOTTOMLEFT
                ),
            )
        return prov

    origin = DocumentOrigin(
        mimetype="application/pdf",
        filename=legacy_doc.file_info.filename,
        binary_hash=legacy_doc.file_info.document_hash,
    )
    doc_name = Path(origin.filename).stem

    doc: DoclingDocument = DoclingDocument(name=doc_name, origin=origin)

    # define pages
    if legacy_doc.page_dimensions is not None:
        for page_dim in legacy_doc.page_dimensions:
            page_no = int(page_dim.page)
            size = Size(width=page_dim.width, height=page_dim.height)

            doc.add_page(page_no=page_no, size=size)

    # page headers
    if legacy_doc.page_headers is not None:
        for text_item in legacy_doc.page_headers:
            if text_item.text is None:
                continue
            prov = _transform_prov(text_item)
            doc.add_text(
                label=DocItemLabel.PAGE_HEADER,
                text=text_item.text,
                content_layer=ContentLayer.FURNITURE,
            )

    # page footers
    if legacy_doc.page_footers is not None:
        for text_item in legacy_doc.page_footers:
            if text_item.text is None:
                continue
            prov = _transform_prov(text_item)
            doc.add_text(
                label=DocItemLabel.PAGE_FOOTER,
                text=text_item.text,
                content_layer=ContentLayer.FURNITURE,
            )

    # footnotes
    if legacy_doc.footnotes is not None:
        for text_item in legacy_doc.footnotes:
            if text_item.text is None:
                continue
            prov = _transform_prov(text_item)
            doc.add_text(
                label=DocItemLabel.FOOTNOTE, text=text_item.text, parent=doc.furniture
            )

    # main-text content
    if legacy_doc.main_text is not None:
        item: Optional[Union[BaseCell, BaseText]]

        # collect all captions embedded in table and figure objects
        # to avoid repeating them
        embedded_captions: Dict[str, int] = {}
        for ix, orig_item in enumerate(legacy_doc.main_text):
            item = (
                legacy_doc._resolve_ref(orig_item)
                if isinstance(orig_item, Ref)
                else orig_item
            )
            if item is None:
                continue

            if isinstance(item, (DsSchemaTable, Figure)) and item.text:
                embedded_captions[item.text] = ix

        # build lookup from floating objects to their caption item
        floating_to_caption: Dict[int, BaseText] = {}
        for ix, orig_item in enumerate(legacy_doc.main_text):
            item = (
                legacy_doc._resolve_ref(orig_item)
                if isinstance(orig_item, Ref)
                else orig_item
            )
            if item is None:
                continue

            item_type = item.obj_type.lower()
            if (
                isinstance(item, BaseText)
                and (
                    item_type == "caption"
                    or (item.name is not None and item.name.lower() == "caption")
                )
                and item.text in embedded_captions
            ):
                floating_ix = embedded_captions[item.text]
                floating_to_caption[floating_ix] = item

        # main loop iteration
        current_list: Optional[GroupItem] = None
        for ix, orig_item in enumerate(legacy_doc.main_text):
            item = (
                legacy_doc._resolve_ref(orig_item)
                if isinstance(orig_item, Ref)
                else orig_item
            )
            if item is None:
                continue

            prov = _transform_prov(item)
            item_type = item.obj_type.lower()

            # if a group is needed, add it
            if isinstance(item, BaseText) and (
                item_type in "list-item-level-1" or item.name in {"list", "list-item"}
            ):
                if current_list is None:
                    current_list = doc.add_group(label=GroupLabel.LIST, name="list")
            else:
                current_list = None

            # add the document item in the document
            if isinstance(item, BaseText):
                text = item.text if item.text is not None else ""
                label_name = item.name if item.name is not None else "text"

                if item_type == "caption":
                    if text in embedded_captions:
                        # skip captions if they are embedded in the actual
                        # floating objects
                        continue
                    else:
                        # captions without a related object are inserted as text
                        doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)

                # first title match
                if item_type == "title":
                    doc.add_title(text=text, prov=prov)

                # secondary titles
                elif item_type in {
                    "subtitle-level-1",
                }:
                    doc.add_heading(text=text, prov=prov)

                # list item
                elif item_type in "list-item-level-1" or label_name in {
                    "list",
                    "list-item",
                }:
                    # TODO: Infer if this is a numbered or a bullet list item
                    doc.add_list_item(
                        text=text, enumerated=False, prov=prov, parent=current_list
                    )

                # normal text
                else:
                    label = DocItemLabel.TEXT
                    normalized_label_name = label_name.replace("-", "_")
                    if normalized_label_name is not None:
                        try:
                            label = DocItemLabel(normalized_label_name)
                        except ValueError:
                            pass
                    doc.add_text(label=label, text=text, prov=prov)

            elif isinstance(item, DsSchemaTable):

                table_data = TableData(num_cols=item.num_cols, num_rows=item.num_rows)
                if item.data is not None:
                    seen_spans = set()
                    for row_ix, row in enumerate(item.data):
                        for col_ix, orig_cell_data in enumerate(row):

                            cell_bbox: Optional[BoundingBox] = (
                                BoundingBox.from_tuple(
                                    tuple(orig_cell_data.bbox),
                                    origin=CoordOrigin.BOTTOMLEFT,
                                )
                                if orig_cell_data.bbox is not None
                                else None
                            )
                            cell = TableCell(
                                start_row_offset_idx=row_ix,
                                end_row_offset_idx=row_ix + 1,
                                start_col_offset_idx=col_ix,
                                end_col_offset_idx=col_ix + 1,
                                text=orig_cell_data.text,
                                bbox=cell_bbox,
                                column_header=(orig_cell_data.obj_type == "col_header"),
                                row_header=(orig_cell_data.obj_type == "row_header"),
                                row_section=(orig_cell_data.obj_type == "row_section"),
                            )

                            if orig_cell_data.spans is not None:
                                # convert to a tuple of tuples for hashing
                                spans_tuple = tuple(
                                    tuple(span) for span in orig_cell_data.spans
                                )

                                # skip repeated spans
                                if spans_tuple in seen_spans:
                                    continue

                                seen_spans.add(spans_tuple)

                                cell.start_row_offset_idx = min(
                                    s[0] for s in spans_tuple
                                )
                                cell.end_row_offset_idx = (
                                    max(s[0] for s in spans_tuple) + 1
                                )
                                cell.start_col_offset_idx = min(
                                    s[1] for s in spans_tuple
                                )
                                cell.end_col_offset_idx = (
                                    max(s[1] for s in spans_tuple) + 1
                                )

                                cell.row_span = (
                                    cell.end_row_offset_idx - cell.start_row_offset_idx
                                )
                                cell.col_span = (
                                    cell.end_col_offset_idx - cell.start_col_offset_idx
                                )

                            table_data.table_cells.append(cell)

                new_item = doc.add_table(data=table_data, prov=prov)
                if (caption_item := floating_to_caption.get(ix)) is not None:
                    if caption_item.text is not None:
                        caption_prov = _transform_prov(caption_item)
                        caption = doc.add_text(
                            label=DocItemLabel.CAPTION,
                            text=caption_item.text,
                            prov=caption_prov,
                            parent=new_item,
                        )
                        new_item.captions.append(caption.get_ref())

            elif isinstance(item, Figure):
                new_item = doc.add_picture(prov=prov)
                if (caption_item := floating_to_caption.get(ix)) is not None:
                    if caption_item.text is not None:
                        caption_prov = _transform_prov(caption_item)
                        caption = doc.add_text(
                            label=DocItemLabel.CAPTION,
                            text=caption_item.text,
                            prov=caption_prov,
                            parent=new_item,
                        )
                        new_item.captions.append(caption.get_ref())

            # equations
            elif (
                isinstance(item, BaseCell)
                and item.text is not None
                and item_type in {"formula", "equation"}
            ):
                doc.add_text(label=DocItemLabel.FORMULA, text=item.text, prov=prov)

    return doc
