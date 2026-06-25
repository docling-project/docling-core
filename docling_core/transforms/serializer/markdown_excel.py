"""Markdown serialization for MS Excel documents."""

from typing import Any

from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownFallbackSerializer,
)
from docling_core.types.doc.document import (
    ContentLayer,
    DoclingDocument,
    GroupItem,
    NodeItem,
)


class MsExcelMarkdownFallbackSerializer(MarkdownFallbackSerializer):
    """Fallback serializer that renders sheet groups as Markdown headings.

    When a ``GroupItem`` with ``content_layer=ContentLayer.SHEET`` is encountered
    the group's ``name`` is emitted as a level-2 Markdown heading (``##``) before
    the group's children, matching the visual structure of the original workbook.
    """

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        if isinstance(item, GroupItem) and item.content_layer == ContentLayer.SHEET:
            parts = doc_serializer.get_parts(item=item, **kwargs)
            content = "\n\n".join(p.text for p in parts if p.text)
            heading = f"## {item.name}"
            text = f"{heading}\n\n{content}" if content else heading
            return create_ser_result(text=text, span_source=parts)
        return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)


class MsExcelMarkdownDocSerializer(MarkdownDocSerializer):
    r"""``MarkdownDocSerializer`` variant for Excel-sourced ``DoclingDocument``\\s.

    Swap in :class:`MsExcelMarkdownFallbackSerializer` so that worksheet
    groups (``ContentLayer.SHEET``) are rendered with their name as a Markdown
    heading without requiring heading nodes to be injected into the document
    model by the backend.

    Usage::

        from docling_core.transforms.serializer.markdown_excel import (
            MsExcelMarkdownDocSerializer,
        )

        serializer = MsExcelMarkdownDocSerializer(doc=result.document)
        md = serializer.serialize().text
    """

    fallback_serializer: BaseFallbackSerializer = MsExcelMarkdownFallbackSerializer()
