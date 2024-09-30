#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Base metadata extractor module."""


from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from docling_core.types import Document as DLDocument


class BaseMetadataExtractor(BaseModel, ABC):
    """Metadata extractor base class."""

    @abstractmethod
    def get_metadata(
        self, doc: DLDocument, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Extract metadata for the given document.

        Args:
            doc (DLDocument): document to extract metadata for

        Raises:
            NotImplementedError: in this abstract implementation

        Returns:
            dict[str, Any]: the extracted metadata
        """
        raise NotImplementedError()

    @abstractmethod
    def get_excluded_embed_metadata_keys(self) -> list[str]:
        """Get metadata keys to exclude from embedding.

        Raises:
            NotImplementedError: in this abstract implementation

        Returns:
            list[str]: the metadata to exclude
        """
        raise NotImplementedError()

    @abstractmethod
    def get_excluded_llm_metadata_keys(self) -> list[str]:
        """Get metadata keys to exclude from LLM generation.

        Raises:
            NotImplementedError: in this abstract implementation

        Returns:
            list[str]: the metadata to exclude
        """
        raise NotImplementedError()
