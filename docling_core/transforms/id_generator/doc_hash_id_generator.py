#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Doc-hash-based ID generator module."""


from typing import Any

from docling_core.transforms.id_generator import BaseIDGenerator
from docling_core.types import Document as DLDocument


class DocHashIDGenerator(BaseIDGenerator):
    """Doc-hash-based ID generator class."""

    def generate_id(self, doc: DLDocument, *args: Any, **kwargs: Any) -> str:
        """Generate an ID for the given document.

        Args:
            doc (DLDocument): document to generate ID for

        Returns:
            str: the generated ID
        """
        return doc.file_info.document_hash
