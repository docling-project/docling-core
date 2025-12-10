"""Data model for code chunks."""

from __future__ import annotations

from enum import Enum
from typing import Literal

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
    part_name: str | None = Field(default=None)
    docstring: str | None = Field(default=None)
    sha256: int | None = Field(default=None)
    start_line: int | None = Field(default=None)
    end_line: int | None = Field(default=None)
    end_line_signature: int | None = Field(default=None)
    chunk_type: CodeChunkType = Field(default=CodeChunkType.CODE_BLOCK)


class CodeChunk(BaseChunk):
    """Data model for code chunks."""

    meta: CodeDocMeta
