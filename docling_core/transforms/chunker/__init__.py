#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the chunker types."""

from docling_core.transforms.chunker.base import BaseChunk, BaseChunker, BaseMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
