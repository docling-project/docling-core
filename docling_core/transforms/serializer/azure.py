"""Define classes for Azure serialization.

This serializer exports a DoclingDocument to a JSON structure that mirrors
the Azure Document Intelligence layout output used in
`azure_document_intelligence.convert_azure_output_to_docling`.

It traverses the document similarly to the HTML/Markdown serializers but
accumulates structured JSON for:
- pages (number, width, height; words omitted by default)
- tables (with bounding regions and cells)
- figures (with bounding regions and optional footnotes)
- paragraphs (with optional Azure roles)

Notes:
- Word-level segmentation is not available in the DoclingDocument, so the
  exported `pages[*].words` array is left empty.
- Bounding boxes are normalized to TOPLEFT origin when page size is known.
"""

# TODO: module to be refactored to either align with Serializer framework or get consolidated into single
# BaseDocSerializer subclass

import json
from typing import Any, Optional

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    create_ser_result,
)
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    NodeItem,
    PictureItem,
    RefItem,
    RichTableCell,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel


def _bbox_to_polygon_coords(
    *,
    l: float,
    t: float,
    r: float,
    b: float,
) -> list[float]:
    """Create a flat polygon list [x1,y1, x2,y2, x3,y3, x4,y4] from bbox."""
    # Order: top-left, top-right, bottom-right, bottom-left
    return [l, t, r, t, r, b, l, b]


def _bbox_to_polygon_for_item(
    doc: DoclingDocument, item: DocItem
) -> Optional[list[float]]:
    """Compute a TOPLEFT-origin polygon for the first provenance of the item."""
    if not item.prov:
        return None

    prov = item.prov[0]
    page_no = prov.page_no
    bbox = prov.bbox
    if bbox is None:
        return None

    # Normalize to TOPLEFT origin when page height is known
    if page_no in doc.pages and doc.pages[page_no].size is not None:
        page_h = doc.pages[page_no].size.height
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(page_height=page_h)

    l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b
    return _bbox_to_polygon_coords(l=l, t=t, r=r, b=b)


class AzureParams(CommonParams):
    """Azure-specific serialization parameters.

    - include_words: whether to export page words (not supported; kept for future).
    """

    include_words: bool = False
    indent: Optional[int] = None


class _AzureBoundingRegion(BaseModel):
    """Bounding region on a page with polygon coordinates.

    Matches Azure's schema; field names use camelCase by design.
    """

    pageNumber: int  # noqa: N815
    polygon: list[float]


class _AzureParagraph(BaseModel):
    """Paragraph content with optional role and regions."""

    content: str
    boundingRegions: list["_AzureBoundingRegion"]  # noqa: N815
    role: Optional[str] = None


class _AzureTableCell(BaseModel):
    """Single table cell with position, span, and optional region."""

    content: str
    rowIndex: int  # noqa: N815
    columnIndex: int  # noqa: N815
    rowSpan: int = 1  # noqa: N815
    colSpan: int = 1  # noqa: N815
    kind: Optional[str] = None
    boundingRegions: Optional[list[_AzureBoundingRegion]] = None  # noqa: N815


class _AzureTable(BaseModel):
    """Table with dimensions, regions, and cells."""

    rowCount: int  # noqa: N815
    columnCount: int  # noqa: N815
    boundingRegions: list[_AzureBoundingRegion]  # noqa: N815
    cells: list[_AzureTableCell]


class _AzureImage(BaseModel):
    """Image/figure with bounding region and optional footnotes."""

    boundingRegions: list[_AzureBoundingRegion]  # noqa: N815
    footnotes: Optional[list[_AzureParagraph]] = None


class _AzurePage(BaseModel):
    """Page metadata used in the Azure-like output."""

    pageNumber: int  # noqa: N815
    width: float
    height: float
    # Words are not currently emitted; keep as untyped list
    words: list[Any] = Field(default_factory=list)


class _AzureOutput(BaseModel):
    """Pydantic model for the Azure-like accumulated output."""

    pages: list[_AzurePage] = Field(default_factory=list)
    tables: list[_AzureTable] = Field(default_factory=list)
    figures: list[_AzureImage] = Field(default_factory=list)
    paragraphs: list[_AzureParagraph] = Field(default_factory=list)


class _AzureTextSerializer(BaseModel, BaseTextSerializer):
    """Serializer that collects paragraphs with optional roles."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        **kwargs: Any,
    ) -> SerializationResult:
        assert isinstance(doc_serializer, AzureDocSerializer)

        # Lists may be represented either as TextItem(ListItem) or via groups;
        # we treat any TextItem as a paragraph-like entry.
        if item.prov:
            prov = item.prov[0]
            page_no = prov.page_no
            polygon = _bbox_to_polygon_for_item(doc, item)
        else:
            page_no = 1
            polygon = None

        role: Optional[str] = None
        if item.label == DocItemLabel.TITLE:
            role = "title"
        elif item.label == DocItemLabel.SECTION_HEADER:
            role = "sectionHeading"
        elif item.label == DocItemLabel.FOOTNOTE:
            role = "footnote"
        elif item.label == DocItemLabel.PAGE_HEADER:
            role = "pageHeader"
        elif item.label == DocItemLabel.PAGE_FOOTER:
            role = "pageFooter"
        # Other labels map to regular paragraphs without a specific role

        content = item.text

        if content != "" and polygon is not None:
            para = _AzureParagraph(
                content=content,
                boundingRegions=[
                    _AzureBoundingRegion(pageNumber=page_no, polygon=polygon)
                ],
                role=role,
            )

            doc_serializer._azure.paragraphs.append(para)

        # Nothing to emit as text; we just filled the accumulator
        return create_ser_result()


class _AzureTableSerializer(BaseTableSerializer):
    """Serializer that collects tables with cell metadata."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        assert isinstance(doc_serializer, AzureDocSerializer)

        if not item.prov:
            return create_ser_result()

        prov = item.prov[0]
        page_no = prov.page_no
        poly = _bbox_to_polygon_for_item(doc, item)
        if poly is None:
            return create_ser_result()

        table_obj = _AzureTable(
            rowCount=item.data.num_rows,
            columnCount=item.data.num_cols,
            boundingRegions=[_AzureBoundingRegion(pageNumber=page_no, polygon=poly)],
            cells=[],
        )

        # Serialize cells from the computed grid
        for i, row in enumerate(item.data.grid):
            for j, cell in enumerate(row):
                # Only materialize each spanning cell once at its anchor position
                if i != cell.start_row_offset_idx or j != cell.start_col_offset_idx:
                    continue

                # For RichTableCell, get textual content via helper
                if isinstance(cell, RichTableCell):
                    content_text = cell._get_text(
                        doc=doc, doc_serializer=doc_serializer
                    )
                else:
                    content_text = cell.text

                cell_poly: Optional[list[float]] = None
                if cell.bbox is not None:
                    # Normalize cell bbox to TOPLEFT origin
                    bbox = cell.bbox
                    if page_no in doc.pages and doc.pages[page_no].size is not None:
                        page_h = doc.pages[page_no].size.height
                        if bbox.coord_origin != CoordOrigin.TOPLEFT:
                            bbox = bbox.to_top_left_origin(page_height=page_h)
                    cell_poly = _bbox_to_polygon_coords(
                        l=bbox.l, t=bbox.t, r=bbox.r, b=bbox.b
                    )

                cell_obj = _AzureTableCell(
                    content=content_text.strip(),
                    rowIndex=cell.start_row_offset_idx,
                    columnIndex=cell.start_col_offset_idx,
                    rowSpan=max(cell.row_span, 1),
                    colSpan=max(cell.col_span, 1),
                    kind=(
                        "columnHeader"
                        if cell.column_header
                        else ("rowHeader" if cell.row_header else None)
                    ),
                    boundingRegions=(
                        [_AzureBoundingRegion(pageNumber=page_no, polygon=cell_poly)]
                        if cell_poly is not None
                        else None
                    ),
                )

                table_obj.cells.append(cell_obj)

        doc_serializer._azure.tables.append(table_obj)
        return create_ser_result()


class _AzurePictureSerializer(BasePictureSerializer):
    """Serializer that collects figures with optional footnotes."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        assert isinstance(doc_serializer, AzureDocSerializer)

        if not item.prov:
            return create_ser_result()

        prov = item.prov[0]
        page_no = prov.page_no
        poly = _bbox_to_polygon_for_item(doc, item)
        if poly is None:
            return create_ser_result()

        fig_obj = _AzureImage(
            boundingRegions=[_AzureBoundingRegion(pageNumber=page_no, polygon=poly)]
        )

        # Include picture footnotes if present
        foots = []
        for foot_ref in item.footnotes:
            if isinstance(foot_ref, RefItem):
                tgt = foot_ref.resolve(doc)
                if isinstance(tgt, TextItem) and tgt.prov:
                    f_poly = _bbox_to_polygon_for_item(doc, tgt)
                    if f_poly is not None:
                        foots.append(
                            _AzureParagraph(
                                content=tgt.text,
                                boundingRegions=[
                                    _AzureBoundingRegion(
                                        pageNumber=tgt.prov[0].page_no, polygon=f_poly
                                    )
                                ],
                            )
                        )

        if foots:
            fig_obj.footnotes = foots

        doc_serializer._azure.figures.append(fig_obj)
        return create_ser_result()


class _AzureKeyValueSerializer(BaseKeyValueSerializer):
    """No-op for Azure output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        # Azure JSON we target does not include KeyValue/Form regions; ignore.
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _AzureFormSerializer(BaseFormSerializer):
    """No-op for Azure output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _AzureListSerializer(BaseModel, BaseListSerializer):
    """Lists are flattened via their TextItem children; no direct output."""

    @override
    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        **kwargs: Any,
    ) -> SerializationResult:
        # Do not recurse here; the outer traversal in DocSerializer.get_parts
        # will visit children already. We emit no direct list structure.
        _ = (doc, list_level, is_inline_scope, item, doc_serializer, kwargs)
        return create_ser_result()


class _AzureInlineSerializer(BaseInlineSerializer):
    """Inline groups are flattened; no direct output."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        list_level: int = 0,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (doc, list_level, item, doc_serializer, kwargs)
        return create_ser_result()


class _AzureFallbackSerializer(BaseFallbackSerializer):
    """Fallback for groups; triggers traversal only."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        # No recursion; outer traversal covers children already.
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _AzureAnnotationSerializer(BaseAnnotationSerializer):
    """No-op for annotations in Azure output."""

    @override
    def serialize(self, *, item: DocItem, **kwargs: Any) -> SerializationResult:
        _ = (item, kwargs)
        return create_ser_result()


class AzureDocSerializer(DocSerializer):
    """Azure-specific document serializer.

    Produces a JSON string compatible with the inverse mapping in
    `azure_document_intelligence.convert_azure_output_to_docling`.
    """

    text_serializer: BaseTextSerializer = _AzureTextSerializer()
    table_serializer: BaseTableSerializer = _AzureTableSerializer()
    picture_serializer: BasePictureSerializer = _AzurePictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = _AzureKeyValueSerializer()
    form_serializer: BaseFormSerializer = _AzureFormSerializer()
    fallback_serializer: BaseFallbackSerializer = _AzureFallbackSerializer()

    list_serializer: BaseListSerializer = _AzureListSerializer()
    inline_serializer: BaseInlineSerializer = _AzureInlineSerializer()
    annotation_serializer: BaseAnnotationSerializer = _AzureAnnotationSerializer()

    params: AzureParams = AzureParams()

    # Accumulator for the Azure-like output (private)
    _azure: _AzureOutput = PrivateAttr(default_factory=_AzureOutput)

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],  # not used; traversal already filled state
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the collected state to an Azure-compatible JSON string.

        Initializes the internal accumulator when empty, exports page
        metadata (number and size) in input order, and returns a
        SerializationResult whose `text` contains the compact JSON.
        """
        # Ensure accumulator has expected top-level keys even if partially filled
        # during traversal by item serializers (which may set paragraphs/tables/figures).
        # Ensure accumulator exists (lists are initialized by default_factory)
        _ = self._azure

        # Pages: export number/size; words omitted by default
        # Keep original order by page number
        for page_no in sorted(self.doc.pages.keys()):
            page = self.doc.pages[page_no]
            if page.size is not None:
                self._azure.pages.append(
                    _AzurePage(
                        pageNumber=page_no,
                        width=page.size.width,
                        height=page.size.height,
                        words=[],
                    )
                )

        # Convert accumulated structure to compact JSON string
        # Produce a compact JSON string from the Pydantic model
        data = self._azure.model_dump(exclude_none=True)
        json_text = json.dumps(data, ensure_ascii=False, indent=self.params.indent)
        return create_ser_result(text=json_text, span_source=parts)

    # Formatting/hyperlink hooks are no-ops for JSON output
    @override
    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        """Return the input text unchanged; bold is not encoded in JSON."""
        return text

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Return the input text unchanged; italic is not encoded in JSON."""
        return text

    @override
    def serialize_underline(self, text: str, **kwargs: Any) -> str:
        """Return the input text unchanged; underline is not encoded in JSON."""
        return text

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Return the input text unchanged; strikethrough is not encoded in JSON."""
        return text

    @override
    def serialize_subscript(self, text: str, **kwargs: Any) -> str:
        """Return the input text unchanged; subscript is not encoded in JSON."""
        return text

    @override
    def serialize_superscript(self, text: str, **kwargs: Any) -> str:
        """Return the input text unchanged; superscript is not encoded in JSON."""
        return text

    @override
    def serialize_hyperlink(self, text: str, hyperlink, **kwargs: Any) -> str:
        """Return the input text unchanged; hyperlinks are not encoded in JSON."""
        return text

    @override
    def requires_page_break(self) -> bool:
        """Azure JSON output does not include page break markers."""
        return False
