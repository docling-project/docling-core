"""Generic annotation models shared across document items."""

from typing import Any, Literal

from pydantic import BaseModel


class BaseAnnotation(BaseModel):
    """Base class for all annotation types."""

    kind: str


class DescriptionAnnotation(BaseAnnotation):
    """DescriptionAnnotation."""

    kind: Literal["description"] = "description"
    text: str
    provenance: str


class MiscAnnotation(BaseAnnotation):
    """MiscAnnotation."""

    kind: Literal["misc"] = "misc"
    content: dict[str, Any]
