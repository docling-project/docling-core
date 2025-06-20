"""Page-based chunker implementation: each chunk corresponds to a single page."""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import ConfigDict
from typing_extensions import override

from docling_core.transforms.chunker import BaseChunker, DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingSerializerProvider,
)
from docling_core.types import DoclingDocument as DLDocument


class PageChunker(BaseChunker):
    r"""Chunker implementation that yields one chunk per page."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    serializer_provider: ChunkingSerializerProvider = ChunkingSerializerProvider()

    @override
    def chunk(
        self,
        dl_doc: DLDocument,
        **kwargs: Any,
    ) -> Iterator[DocChunk]:
        """Chunk the provided document by page."""
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        if dl_doc.pages:
            # chunk by page
            for page_no in sorted(dl_doc.pages.keys()):
                ser_res = my_doc_ser.serialize(pages={page_no})
                if not ser_res.text:
                    continue
                yield DocChunk(
                    text=ser_res.text,
                    meta=DocMeta(
                        doc_items=ser_res.get_unique_doc_items(),
                        headings=None,
                        captions=None,
                        origin=dl_doc.origin,
                    ),
                )
        else:
            # if no pages, treat whole document as single chunk
            ser_res = my_doc_ser.serialize()
            if ser_res.text:
                yield DocChunk(
                    text=ser_res.text,
                    meta=DocMeta(
                        doc_items=ser_res.get_unique_doc_items(),
                        headings=None,
                        captions=None,
                        origin=dl_doc.origin,
                    ),
                )
