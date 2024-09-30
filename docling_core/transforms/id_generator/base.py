#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Base document ID generator module."""

from abc import ABC, abstractmethod
from typing import Any

from docling_core.types import Document as DLDocument


class BaseIDGenerator(ABC):
    """Document ID generator base class."""

    @abstractmethod
    def generate_id(self, doc: DLDocument, *args: Any, **kwargs: Any) -> str:
        """Generate an ID for the given document.

        Args:
            doc (DLDocument): document to generate ID for

        Raises:
            NotImplementedError: in this abstract implementation

        Returns:
            str: the generated ID
        """
        raise NotImplementedError()
