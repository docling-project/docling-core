from enum import Enum
from typing import Optional

from pydantic import Field

from docling_core.transforms.chunker.base import BaseChunk, BaseMeta
from docling_core.types.doc.document import DocumentOrigin


class CodeDocMeta(BaseMeta):
    """Data model for CodeChunker metadata."""

    part_name: Optional[str] = Field(default=None)
    docstring: Optional[str] = Field(default=None)
    sha256: Optional[int] = Field(default=None)
    start_line: Optional[int] = Field(default=None)
    end_line: Optional[int] = Field(default=None)
    end_line_signature: Optional[int] = Field(default=None)
    origin: Optional[DocumentOrigin] = Field(default=None)
    chunk_type: Optional[str] = Field(default=None)


class ChunkType(str, Enum):
    """Chunk type"""

    FUNCTION = "function"
    METHOD = "method"
    PREAMBLE = "preamble"
    CLASS = "class"


class CodeChunk(BaseChunk):
    """Data model for code chunks."""

    meta: CodeDocMeta
