#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""UUID-based ID generator module."""

from random import Random
from typing import Annotated, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from docling_core.transforms.id_generator import BaseIDGenerator
from docling_core.types import Document as DLDocument


class UUIDGenerator(BaseModel, BaseIDGenerator):
    """UUID-based ID generator class."""

    seed: Optional[int] = None
    uuid_version: Annotated[int, Field(strict=True, ge=1, le=5)] = 4

    def generate_id(self, doc: DLDocument, *args: Any, **kwargs: Any) -> str:
        """Generate an ID for the given document.

        Args:
            doc (DLDocument): document to generate ID for

        Returns:
            str: the generated ID
        """
        rd = Random(x=self.seed)
        return str(UUID(int=rd.getrandbits(128), version=self.uuid_version))
