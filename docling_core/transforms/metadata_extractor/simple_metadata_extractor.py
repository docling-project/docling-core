#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Simple metadata extractor module."""


from typing import Any, Final

from docling_core.transforms.metadata_extractor import BaseMetadataExtractor
from docling_core.types import Document as DLDocument

_DL_DOC_HASH: Final[str] = "dl_doc_hash"
_ORIGIN: Final[str] = "origin"


class SimpleMetadataExtractor(BaseMetadataExtractor):
    """Simple metadata extractor class."""

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
            _DL_DOC_HASH: doc.file_info.document_hash,
        }
        if self.include_origin:
            meta[_ORIGIN] = origin
        return meta

    def get_excluded_embed_metadata_keys(self) -> list[str]:
        """Get metadata keys to exclude from embedding.

        Returns:
            list[str]: the metadata to exclude
        """
        excl_keys: list[str] = [_DL_DOC_HASH]
        if self.include_origin:
            excl_keys.append(_ORIGIN)
        return excl_keys

    def get_excluded_llm_metadata_keys(self) -> list[str]:
        """Get metadata keys to exclude from LLM generation.

        Returns:
            list[str]: the metadata to exclude
        """
        return self.get_excluded_embed_metadata_keys()
