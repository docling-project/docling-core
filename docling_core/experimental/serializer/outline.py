"""Markdown document summary serializers (outline and TOC).

This module provides a Markdown-focused serializer that emits a compact
document outline or a table of contents derived from a Docling document.
"""

import json
from enum import Enum
from typing import Annotated, Any

from pydantic import Field, model_validator
from typing_extensions import Self, override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BaseMetaSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownMetaSerializer,
    MarkdownParams,
    MarkdownTextSerializer,
)
from docling_core.types.doc import (
    BaseMeta,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FormItem,
    GroupItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableItem,
    TextItem,
    TitleItem,
)


def _default_prepend(item: NodeItem) -> str:
    if isinstance(item, DocItem | GroupItem):
        return f"{item.label.value} "
    else:
        raise ValueError("item is neither DocItem nor GroupItem")


def _default_outline_node(item: NodeItem) -> str:
    return f"[reference={item.self_ref}]"


def _default_summary(summary: str) -> str:
    return f"(summary={summary})"


def _default_text(item: NodeItem, doc: DoclingDocument, **kwargs: Any) -> str:
    if isinstance(item, ListItem):
        return ""

    params = OutlineParams(**kwargs)

    # For JSON format, return a JSON string representation
    if params.format == OutlineFormat.JSON:
        data: dict[str, Any] = {
            "ref": item.self_ref,
        }

        # Add title if include_non_meta is True
        if params.include_non_meta and isinstance(item, TitleItem | SectionHeaderItem):
            _md_serializer = MarkdownDocSerializer(doc=doc)
            _serializer = MarkdownTextSerializer()
            res = _serializer.serialize(item=item, doc_serializer=_md_serializer, doc=doc, **kwargs)
            data["title"] = res.text.strip()

        # Always include summary if available
        if item.meta and item.meta.summary:
            data["summary"] = item.meta.summary.text

        return json.dumps(data, ensure_ascii=False)

    # For Markdown format, build text parts
    text_parts = []

    # Only include prepend (actual text content) if include_non_meta is True
    if params.include_non_meta:
        prepend = _default_prepend(item)
        if isinstance(item, TitleItem | SectionHeaderItem):
            # MarkdownDocSerializer requires a doc instance; pass through current doc
            _md_serializer = MarkdownDocSerializer(doc=doc)
            _serializer = MarkdownTextSerializer()

            res = _serializer.serialize(item=item, doc_serializer=_md_serializer, doc=doc, **kwargs)
            prepend = res.text
        text_parts.append(prepend)

    # Always include reference (structure)
    reference = _default_outline_node(item)
    text_parts.append(reference)

    # Always include summary (metadata) if available
    if item.meta and item.meta.summary:
        summary = _default_summary(item.meta.summary.text)
        text_parts.append(summary)

    text = " ".join(text_parts)

    return text.strip()


class OutlineMode(str, Enum):
    """Display mode for document summary output."""

    OUTLINE = "outline"
    TABLE_OF_CONTENTS = "table_of_contents"


class OutlineFormat(str, Enum):
    """Output format for outline serialization."""

    MARKDOWN = "markdown"
    JSON = "json"


class OutlineParams(MarkdownParams):
    """Markdown-specific serialization parameters for outline.

    Inherits MarkdownParams to retain Markdown behaviors (escaping, links, etc.).
    """

    mode: Annotated[
        OutlineMode,
        Field(
            description="Display mode: 'outline' includes all document elements, 'table_of_contents' shows only titles and section headers"
        ),
    ] = OutlineMode.OUTLINE
    format: Annotated[
        OutlineFormat,
        Field(description="Output format: 'markdown' for human-readable text, 'json' for structured data"),
    ] = OutlineFormat.MARKDOWN

    @model_validator(mode="after")
    def adjust_allowed_labels(self) -> Self:
        """Adjust the allowed labels based on the selected mode."""
        if self.mode == OutlineMode.TABLE_OF_CONTENTS:
            if "labels" not in self.model_fields_set:
                self.labels = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
        return self


class _OutlineTextSerializer(BaseTextSerializer):
    """_Outline class for text item serializers."""

    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the passed item."""
        # Pass the original params from doc_serializer to respect include_non_meta
        # Remove include_non_meta from kwargs if present (it was overridden to True)
        # and use the original value from doc_serializer.params
        kwargs_copy = {k: v for k, v in kwargs.items() if k != "include_non_meta"}
        include_non_meta = (
            doc_serializer.params.include_non_meta if isinstance(doc_serializer, MarkdownDocSerializer) else True
        )
        text = _default_text(item=item, doc=doc, include_non_meta=include_non_meta, **kwargs_copy)
        return create_ser_result(text=text)


class _OutlineTableSerializer(BaseTableSerializer):
    """_Outline class for table item serializers."""

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.TABLE not in params.labels:
            return create_ser_result()

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlinePictureSerializer(BasePictureSerializer):
    """_Outline class for picture item serializers."""

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.PICTURE not in params.labels:
            return create_ser_result()

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineKeyValueSerializer(BaseKeyValueSerializer):
    """_Outline class for key value item serializers."""

    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.KEY_VALUE_REGION not in params.labels:
            return create_ser_result()

        print("label: ", item.label)

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineFormSerializer(BaseFormSerializer):
    """_Outline class for form item serializers."""

    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.FORM not in params.labels:
            return create_ser_result()

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineListSerializer(BaseListSerializer):
    """_Outline class for list serializers."""

    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the passed item."""
        # Intentionally skip list containers in outlines
        return create_ser_result()


class _OutlineInlineSerializer(BaseInlineSerializer):
    """_Outline class for inline serializers."""

    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the passed item."""
        return create_ser_result()


class _OutlineFallbackSerializer(BaseFallbackSerializer):
    """_Outline fallback class for item serializers."""

    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineMetaSerializer(MarkdownMetaSerializer):
    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc: DoclingDocument,
        level: int | None = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's meta."""
        return create_ser_result()

    @override
    def _serialize_meta_field(self, meta: BaseMeta, name: str, mark_meta: bool) -> str | None:
        if (field_val := getattr(meta, name)) is not None and isinstance(field_val, SummaryMetaField):
            txt = field_val.text
            return f"[{self._humanize_text(name, title=True)}] {txt}" if mark_meta else txt
        else:
            return None


class OutlineDocSerializer(MarkdownDocSerializer):
    """Markdown-based serializer for outlines and tables of contents."""

    text_serializer: BaseTextSerializer = _OutlineTextSerializer()
    table_serializer: BaseTableSerializer = _OutlineTableSerializer()
    picture_serializer: BasePictureSerializer = _OutlinePictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = _OutlineKeyValueSerializer()
    form_serializer: BaseFormSerializer = _OutlineFormSerializer()
    fallback_serializer: BaseFallbackSerializer = _OutlineFallbackSerializer()

    list_serializer: BaseListSerializer = _OutlineListSerializer()
    inline_serializer: BaseInlineSerializer = _OutlineInlineSerializer()

    meta_serializer: BaseMetaSerializer = _OutlineMetaSerializer()

    params: OutlineParams = OutlineParams()

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts.

        For JSON format, combines individual JSON objects into a JSON array.
        For Markdown format, uses the default behavior.
        """
        params = self.params.merge_with_patch(patch=kwargs)

        if params.format == OutlineFormat.JSON:
            # Parse each part as JSON and combine into an array
            json_objects = []
            for part in parts:
                if part.text:
                    try:
                        json_objects.append(json.loads(part.text))
                    except json.JSONDecodeError:
                        # Skip invalid JSON
                        pass

            # Return the array as a JSON string
            text_res = json.dumps(json_objects, ensure_ascii=False, indent=2)
            return create_ser_result(text=text_res, span_source=parts)
        else:
            # Use default Markdown behavior
            return super().serialize_doc(parts=parts, **kwargs)

    @override
    def get_parts(
        self,
        item: NodeItem | None = None,
        *,
        traverse_pictures: bool = False,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: set[str] | None = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Get serialization parts for the document.

        Override to ensure outline items are always processed regardless of
        include_non_meta setting. The _default_text function will handle
        what content to include based on include_non_meta.
        """
        kwargs_with_meta = {**kwargs, "include_non_meta": True}
        return super().get_parts(
            item=item,
            traverse_pictures=traverse_pictures,
            list_level=list_level,
            is_inline_scope=is_inline_scope,
            visited=visited,
            **kwargs_with_meta,
        )
