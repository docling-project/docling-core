#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define base classes for chunking."""
import re
from abc import ABC, abstractmethod
from typing import Final, Iterator, Optional

from pydantic import BaseModel, Field, field_validator

from docling_core.types import BoundingBox, Document
from docling_core.types.base import _JSON_POINTER_REGEX

# (subset of) JSONPath format, e.g. "$.main-text[84]" (for migration purposes)
_DEPRECATED_JSON_PATH_PATTERN: Final = re.compile(r"^\$\.([\w-]+)\[(\d+)\]$")


def _create_path(pos: int, path_prefix: str = "main-text") -> str:
    return f"#/{path_prefix}/{pos}"


class Chunk(BaseModel):
    """Data model for Chunk."""

    path: str = Field(pattern=_JSON_POINTER_REGEX)
    text: str
    heading: Optional[str] = None

    @field_validator("path", mode="before")
    @classmethod
    def _json_pointer_from_json_path(cls, path: str):
        if (match := _DEPRECATED_JSON_PATH_PATTERN.match(path)) is not None:
            groups = match.groups()
            if len(groups) == 2 and groups[0] is not None and groups[1] is not None:
                return _create_path(
                    pos=int(groups[1]),
                    path_prefix=groups[0],
                )
        return path


class ChunkWithMetadata(Chunk):
    """Data model for Chunk including metadata."""

    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None


class BaseChunker(BaseModel, ABC):
    """Base class for Chunker."""

    @abstractmethod
    def chunk(self, dl_doc: Document, **kwargs) -> Iterator[Chunk]:
        """Chunk the provided document.

        Args:
            dl_doc (Document): document to chunk

        Raises:
            NotImplementedError: in this abstract implementation

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        raise NotImplementedError()

    @classmethod
    def _create_path(cls, pos: int, path_prefix: str = "main-text") -> str:
        return _create_path(
            pos=pos,
            path_prefix=path_prefix,
        )
