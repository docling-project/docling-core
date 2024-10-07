#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define base classes for chunking."""
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from pydantic import BaseModel, model_validator

from docling_core.types import BoundingBox, Document


def _create_path(pos: int, path_prefix: str = "main-text") -> str:
    return f"#/{path_prefix}/{pos}"


class Chunk(BaseModel):
    """Data model for Chunk."""

    path: str
    text: str
    heading: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _json_pointer_from_json_path(cls, data):
        path = data.get("path")
        if path.startswith("$."):
            parts = path.split("[")
            data["path"] = _create_path(
                pos=parts[1][:-1],
                path_prefix=parts[0][2:],
            )
        return data


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
