"""Base code chunking strategy."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from docling_core.transforms.chunker.code_chunking.code_chunk import CodeChunk
from docling_core.types.doc.document import CodeItem, DoclingDocument


class BaseCodeChunkingStrategy(ABC):
    """Base class for code chunking strategies."""

    @abstractmethod
    def chunk_code_item(
        self,
        *,
        item: CodeItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> Iterator[CodeChunk]:
        """Chunk a single code item."""
        ...
