"""Define classes for plain-text serialization."""

from pathlib import Path
from typing import Any, Union

from pydantic import AnyUrl
from typing_extensions import override

from docling_core.transforms.serializer.base import BaseTextSerializer
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTextSerializer,
)
from docling_core.types.doc.document import (
    SectionHeaderItem,
    TitleItem,
)


class PlainTextParams(MarkdownParams):
    """Plain-text serialization parameters."""

    escape_underscores: bool = False
    escape_html: bool = False
    image_placeholder: str = ""
    format_code_blocks: bool = False


class PlainTextTextSerializer(MarkdownTextSerializer):
    """Text serializer that emits headings and titles without ``#`` markers."""

    @override
    def _format_heading(
        self,
        text: str,
        item: Union[TitleItem, SectionHeaderItem],
    ) -> str:
        return text


class PlainTextDocSerializer(MarkdownDocSerializer):
    """Document serializer that produces clean plain text.

    Strips all Markdown decoration — heading markers, bold/italic/strikethrough
    markers, and hyperlink syntax — while keeping list bullets (``-``), ordered
    list numbers, and table-cell separators (``|``) intact.
    """

    text_serializer: BaseTextSerializer = PlainTextTextSerializer()
    params: PlainTextParams = PlainTextParams()

    @override
    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        """Apply plain-text bold serialization."""
        return text

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Apply plain-text italic serialization."""
        return text

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Apply plain-text strikethrough serialization."""
        return text

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ) -> str:
        """Return the link label only, discarding the URL."""
        return text
