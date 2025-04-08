#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define base classes for chunking."""
import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterator

from pydantic import BaseModel
from typing_extensions import deprecated

from docling_core.types.doc import DoclingDocument as DLDocument

DFLT_DELIM = "\n"


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

    delim: str = DFLT_DELIM

    @abstractmethod
    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Raises:
            NotImplementedError: in this abstract implementation

        Yields:
            Iterator[BaseChunk]: iterator over extracted chunks
        """
        raise NotImplementedError()

    def contextualize(self, chunk: BaseChunk) -> str:
        """Contextualize the given chunk. This implementation is embedding-targeted.

        Args:
            chunk: chunk to serialize

        Returns:
            str: the serialized form of the chunk
        """
        meta = chunk.meta.export_json_dict()

        items = []
        for k in meta:
            if k not in chunk.meta.excluded_embed:
                if isinstance(meta[k], list):
                    items.append(
                        self.delim.join(
                            [
                                d if isinstance(d, str) else json.dumps(d)
                                for d in meta[k]
                            ]
                        )
                    )
                else:
                    items.append(json.dumps(meta[k]))
        items.append(chunk.text)

        return self.delim.join(items)

    @deprecated("Use contextualize() instead.")
    def serialize(self, chunk: BaseChunk) -> str:
        """Contextualize the given chunk. This implementation is embedding-targeted."""
        return self.contextualize(chunk=chunk)
