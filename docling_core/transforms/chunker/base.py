#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define base classes for chunking."""
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterator

from pydantic import BaseModel

from docling_core.types.doc import DoclingDocument as DLDocument


class BaseMeta(BaseModel):
    """Chunk metadata base class."""

    excluded_embed: ClassVar[list[str]] = []
    excluded_llm: ClassVar[list[str]] = []

    def export_json_dict(self) -> dict[str, Any]:
        """Helper method for exporting non-None keys to JSON mode.

        Returns:
            dict[str, Any]: The exported dictionary.
        """
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class BaseChunk(BaseModel):
    """Chunk base class."""

    text: str
    meta: BaseMeta

    def export_json_dict(self) -> dict[str, Any]:
        """Helper method for exporting non-None keys to JSON mode.

        Returns:
            dict[str, Any]: The exported dictionary.
        """
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class BaseChunker(BaseModel, ABC):
    """Chunker base class."""

    @abstractmethod
    def chunk(self, dl_doc: DLDocument, **kwargs) -> Iterator[BaseChunk]:
        """Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Raises:
            NotImplementedError: in this abstract implementation

        Yields:
            Iterator[BaseChunk]: iterator over extracted chunks
        """
        raise NotImplementedError()
