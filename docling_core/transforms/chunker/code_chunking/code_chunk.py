"""Data model for code chunks."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import Field

from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.doc_chunk import _KEY_SCHEMA_NAME, DocMeta


class CodeChunkType(str, Enum):
    """Chunk type."""

    FUNCTION = "function"
    METHOD = "method"
    PREAMBLE = "preamble"
    CLASS = "class"
    CODE_BLOCK = "code_block"


class CodeDocMeta(DocMeta):
    """Data model for code chunk metadata."""

    schema_name: Literal["docling_core.transforms.chunker.CodeDocMeta"] = Field(  # type: ignore[assignment]
        default="docling_core.transforms.chunker.CodeDocMeta",
        alias=_KEY_SCHEMA_NAME,
    )
    part_name: Optional[str] = Field(default=None)
    docstring: Optional[str] = Field(default=None)
    sha256: Optional[int] = Field(default=None)
    start_line: Optional[int] = Field(default=None)
    end_line: Optional[int] = Field(default=None)
    end_line_signature: Optional[int] = Field(default=None)
    chunk_type: CodeChunkType = Field(default=CodeChunkType.CODE_BLOCK)


class CodeChunk(BaseChunk):
    """Data model for code chunks."""

    meta: CodeDocMeta
