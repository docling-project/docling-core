"""Define the chunker types."""

from docling_core.transforms.chunker.base import BaseChunk, BaseChunker, BaseMeta
from docling_core.transforms.chunker.code_chunking.base_code_chunking_strategy import (
    BaseCodeChunkingStrategy,
)
from docling_core.transforms.chunker.code_chunking.code_chunk import (
    CodeChunk,
    CodeChunkType,
    CodeDocMeta,
)
from docling_core.transforms.chunker.code_chunking.standard_code_chunking_strategy import (
    StandardCodeChunkingStrategy,
)
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.types.doc.labels import CodeLanguageLabel
