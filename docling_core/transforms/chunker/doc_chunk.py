"""Data model for chunk metadata."""

from __future__ import annotations

import logging
import re
import warnings
from copy import deepcopy
from typing import Annotated, ClassVar, Final, Literal, Optional

from pydantic import Field, StringConstraints, field_validator

from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseMeta
from docling_core.transforms.serializer.common import DocSerializer
from docling_core.types.doc.document import DocItem, DoclingDocument, DocumentOrigin, InlineGroup, ListGroup, RefItem

_VERSION: Final = "1.0.0"

_KEY_SCHEMA_NAME = "schema_name"
_KEY_VERSION = "version"
_KEY_DOC_ITEMS = "doc_items"
_KEY_HEADINGS = "headings"
_KEY_CAPTIONS = "captions"
_KEY_ORIGIN = "origin"

_logger = logging.getLogger(__name__)


class DocMeta(BaseMeta):
    """Data model for Hierarchical Chunker chunk metadata."""

    schema_name: Literal["docling_core.transforms.chunker.DocMeta"] = Field(
        default="docling_core.transforms.chunker.DocMeta",
        alias=_KEY_SCHEMA_NAME,
    )
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = Field(
        default=_VERSION,
        alias=_KEY_VERSION,
    )
    doc_items: list[DocItem] = Field(
        alias=_KEY_DOC_ITEMS,
        min_length=1,
    )
    headings: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_HEADINGS,
        min_length=1,
    )
    captions: Optional[list[str]] = Field(  # deprecated
        deprecated=True,
        default=None,
        alias=_KEY_CAPTIONS,
        min_length=1,
    )
    origin: Optional[DocumentOrigin] = Field(
        default=None,
        alias=_KEY_ORIGIN,
    )

    excluded_embed: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]
    excluded_llm: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]

    @field_validator(_KEY_VERSION)
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this meta item version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, _VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(f"incompatible version {v} with schema version {_VERSION}")
        else:
            return _VERSION


class DocChunk(BaseChunk):
    """Data model for document chunks."""

    meta: DocMeta

    def _get_top_containing_items(self, doc: DoclingDocument) -> list[DocItem] | None:
        """Get top-level document items that contain this chunk's items.

        Traverses the document tree upward from each item in the chunk to find
        the top-level items (direct children of document body) that contain them.
        Maintains the original document reading order.

        Args:
            doc: The DoclingDocument containing this chunk.

        Returns:
            List of top-level DocItems in document order, or None if no items found.
        """

        items = {}
        ref_items = [item.self_ref for item in self.meta.doc_items]
        for item in ref_items:
            # traverse document tree till top level (body)
            top_item = RefItem(cref=item).resolve(doc)
            while top_item.parent != doc.body.get_ref():
                top_item = top_item.parent.resolve(doc)
            items[top_item.self_ref] = top_item

        # maintain the reading order as in the original document
        doc_body_refs = [ref.cref for ref in doc.body.children]
        doc_ordered_refs = [ref for ref in doc_body_refs if ref in items]
        if len(doc_ordered_refs) > 0:
            return [items[ref] for ref in doc_ordered_refs]
        return None

    def expand_to_item(self, dl_doc: DoclingDocument, serializer: DocSerializer) -> DocChunk:
        """Expand chunk to include complete top-level document items.

        Expands the chunk to contain full top-level items (sections, tables, lists)
        rather than partial content. This ensures semantic completeness by including
        all content from the top-level items that contain any part of the original chunk.

        Args:
            dl_doc: The DoclingDocument containing this chunk.
            serializer: Serializer to convert document items to text.

        Returns:
            New DocChunk with expanded content and updated metadata, or the original
            chunk if expansion fails or yields no content.

        Note:
            - It is recommended to use same serializer as the original document
        """
        top_items = self._get_top_containing_items(dl_doc)
        if not top_items:
            _logger.warning(f"error in getting top items of {self}")
            return self

        content = ""
        all_doc_items = []

        for top_item in top_items:
            if isinstance(top_item, ListGroup | InlineGroup | DocItem):
                try:
                    ser_res = serializer.serialize(item=top_item)
                    content += ser_res.text + "\n"
                    # Extract doc_items from serialization result
                    all_doc_items.extend(ser_res.get_unique_doc_items())

                except Exception as e:
                    _logger.warning(f"error in extacting text of {top_item}: {e}")

        if len(content.strip()) == 0:
            _logger.warning(f"expansion of {self} did not yield any text")
            return self

        meta = deepcopy(self.meta)
        meta.doc_items = all_doc_items
        return DocChunk(
            text=content,
            meta=meta,
        )

    def expand_to_page(self, doc: DoclingDocument, serializer: DocSerializer) -> DocChunk:
        """Expand chunk to include all content from its pages.

        Expands the chunk to contain all content from the pages it spans. This is
        useful for maintaining page-level context and ensuring complete page coverage
        in retrieval applications.

        Args:
            doc: The DoclingDocument containing this chunk.
            serializer: Serializer to convert document content to text.

        Returns:
            New DocChunk with all content from the chunk's pages and updated metadata,
            or the original chunk if expansion is not possible.

        Raises:
            UserWarning: If document has no pages or chunk items have no page provenance.

        Example:
            If a chunk spans pages 2-3, this expands it to include all content
            from both pages, not just the original chunk's items.

        Note:
            - It is recommended to use same serializer as the original document
        """

        page_ids = [i.page_no for item in self.meta.doc_items for i in item.prov]

        if len(doc.pages) == 0 or page_ids is None or len(page_ids) == 0:
            warnings.warn(
                f"cannot expand to page the following chunk: {self}. \n Probably pagination was not supported in document conversion"
            )
            return self

        page_serializer = deepcopy(serializer)  # avoid mutating the serializer
        page_serializer.params.pages = set(page_ids)
        ser_res = page_serializer.serialize()

        # Extract doc_items from serialization result
        expanded_doc_items = ser_res.get_unique_doc_items()

        # Update metadata
        meta = deepcopy(self.meta)
        meta.doc_items = expanded_doc_items
        return DocChunk(
            text=ser_res.text,
            meta=meta,
        )
