"""Define the deserializer types."""

from docling_core.transforms.deserializer.base import BaseDocDeserializer
from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer

__all__ = [
    "BaseDocDeserializer",
    "DocLangDocDeserializer",
]
