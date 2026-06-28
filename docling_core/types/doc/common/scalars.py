"""Shared scalar type aliases used across the Docling document model."""

import typing
from typing import Annotated

from pydantic import Field

Uint64 = typing.Annotated[int, Field(ge=0, le=(2**64 - 1))]
LevelNumber = typing.Annotated[int, Field(ge=1, le=100)]
CharSpan = Annotated[tuple[int, int], Field(description="Character span (0-indexed)")]
