#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Simple metadata extractor module."""


from enum import Enum
from typing import Any

from docling_core.transforms.metadata_extractor import BaseMetadataExtractor
from docling_core.types import Document as DLDocument


class SimpleMetadataExtractor(BaseMetadataExtractor):
    """Simple metadata extractor class."""

    class _Keys(str, Enum):
        DL_DOC_HASH = "dl_doc_hash"
        ORIGIN = "origin"

    include_origin: bool = False

    def get_metadata(
        self, doc: DLDocument, origin: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Extract metadata for the given document.

        Args:
            doc (DLDocument): document to extract metadata for
            origin (str): the document origin

        Returns:
            dict[str, Any]: the extracted metadata
        """
        meta: dict[str, Any] = {
            self._Keys.DL_DOC_HASH: doc.file_info.document_hash,
        }
        if self.include_origin:
            meta[self._Keys.ORIGIN] = origin
        return meta

    def get_excluded_embed_metadata_keys(self) -> list[str]:
        """Get metadata keys to exclude from embedding.

        Returns:
            list[str]: the metadata to exclude
        """
        excl_keys: list[str] = [self._Keys.DL_DOC_HASH]
        if self.include_origin:
            excl_keys.append(self._Keys.ORIGIN)
        return excl_keys

    def get_excluded_llm_metadata_keys(self) -> list[str]:
        """Get metadata keys to exclude from LLM generation.

        Returns:
            list[str]: the metadata to exclude
        """
        return self.get_excluded_embed_metadata_keys()
