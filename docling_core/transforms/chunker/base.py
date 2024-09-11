#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define base classes for chunking."""
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from pydantic import BaseModel

from docling_core.types import BoundingBox, Document


class Chunk(BaseModel):
    """Data model for Chunk."""

    path: str
    text: str


class ChunkWithMetadata(Chunk):
    """Data model for Chunk including metadata."""

    page: Optional[int]
    bbox: Optional[BoundingBox]


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
