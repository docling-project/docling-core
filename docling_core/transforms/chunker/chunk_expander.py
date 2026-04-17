import logging
import warnings
from copy import deepcopy

from typing_extensions import override

from docling_core.transforms.chunker.base import BaseChunk, BaseChunkExpander, BaseMeta
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.serializer.common import DocSerializer
from docling_core.types.doc.document import DocItem, DoclingDocument, InlineGroup, ListGroup, RefItem

_logger = logging.getLogger(__name__)


class TreeChunkExpander(BaseChunkExpander):
    """A chunk expander that includes complete top-level document items.

    The TreeChunkExpander expands the chunk to contain full top-level items (sections, tables, lists) rather than
    partial content. This ensures semantic completeness by including all content from the top-level items that
    contain any part of the original chunk.
    """

    @staticmethod
    def _get_top_containing_items(meta: DocMeta, doc: DoclingDocument) -> list[DocItem] | None:
        """Get top-level document items that contain this chunk's items.

        Traverses the document tree upward from each item in the chunk to find the top-level items (direct children
        of document body) that contain them. Maintains the original document reading order.

        Args:
            meta: The chunk metadata containing doc_items to traverse.
            doc: The DoclingDocument containing this chunk.

        Returns:
            List of top-level DocItems in document order, or None if no items found.
        """

        items: dict[str, DocItem] = {}
        ref_items: list[str] = [item.self_ref for item in meta.doc_items]
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

    def expand(self, chunk: BaseChunk, dl_doc: DoclingDocument, serializer: BaseDocSerializer) -> BaseChunk:
        """Expand chunk to include complete top-level document items.

        Expands the chunk to contain full top-level items (sections, tables, lists) rather than partial content.
        This ensures semantic completeness by including all content from the top-level items that contain any part
        of the original chunk.

        Args:
            chunk: The chunk to expand. Must be a DocChunk instance.
            dl_doc: The DoclingDocument containing this chunk.
            serializer: Serializer to convert document items to text.

        Returns:
            New DocChunk with expanded content and updated metadata, or the original chunk if expansion fails,
            yields no content, or is not a DocChunk instance.

        Note:
            It is recommended to use the same serializer as used for the original document.
        """
        if not isinstance(chunk, DocChunk):
            warnings.warn(f"cannot expand chunk since it is not an instance of DocChunk: {chunk}")
            return chunk

        top_items = TreeChunkExpander._get_top_containing_items(chunk.meta, dl_doc)
        if not top_items:
            _logger.warning(f"error in getting top items of {self}")
            return chunk

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
                    _logger.warning(f"error in extracting text of {top_item}: {e}")
        if len(content.strip()) == 0:
            warnings.warn(f"expansion of {chunk} did not yield any text")
            return chunk

        meta = deepcopy(chunk.meta)
        meta.doc_items = all_doc_items
        return DocChunk(
            text=content,
            meta=meta,
        )


class PageChunkExpander(BaseChunkExpander):
    """A chunk expander that includes all content from its pages.

    Expands the chunk to contain all content from the pages it spans. This is useful for maintaining page-level
    context and ensuring complete page coverage in retrieval applications.
    """

    @override
    def expand(self, chunk: BaseChunk, dl_doc: DoclingDocument, serializer: BaseDocSerializer) -> BaseChunk:
        """Expand chunk to include all content from its pages.

        Expands the chunk to contain all content from the pages it spans. This is useful for maintaining page-level
        context and ensuring complete page coverage in retrieval applications.

        Args:
            chunk: The chunk to expand. Must be a DocChunk instance.
            dl_doc: The DoclingDocument containing this chunk.
            serializer: Serializer to convert document content to text.

        Returns:
            New DocChunk with all content from the chunk's pages and updated metadata, or the original chunk if
            expansion is not possible or chunk is not a DocChunk.

        Raises:
            UserWarning: If document has no pages or chunk items have no page provenance.

        Example:
            If a chunk spans pages 2-3, this expands it to include all content from both pages, not just the
            original chunk's items.

        Note:
            It is recommended to use the same serializer as used for the original document.
        """

        if not isinstance(chunk, DocChunk):
            return chunk

        page_ids = [i.page_no for item in chunk.meta.doc_items for i in item.prov]

        if len(dl_doc.pages) == 0 or page_ids is None or len(page_ids) == 0:
            warnings.warn(
                f"Cannot expand to page the following chunk: {chunk}.\n"
                "Probably pagination was not supported in document conversion."
            )
            return chunk

        page_serializer = deepcopy(serializer)  # avoid mutating the serializer
        if isinstance(page_serializer, DocSerializer):
            page_serializer.params.pages = set(page_ids)
        ser_res = page_serializer.serialize()

        # Extract doc_items from serialization result
        expanded_doc_items = ser_res.get_unique_doc_items()

        # Update metadata
        meta = deepcopy(chunk.meta)
        meta.doc_items = expanded_doc_items
        return DocChunk(
            text=ser_res.text,
            meta=meta,
        )
