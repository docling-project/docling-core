"""Markdown document summary serializers (outline and TOC).

This module provides a Markdown-focused serializer that emits a compact
document outline or a table of contents derived from a Docling document.
"""

from enum import Enum
from typing import Any, Optional

from typing_extensions import override

from docling_core.transforms.serializer.base import (
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
    DoclingDocument,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    MetaFieldName,
    NodeItem,
    PictureItem,
    SummaryMetaField,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

def _default_outline_node(item: NodeItem) -> str:
    # return f"[{item.self_ref}] [{item.__class__.__name__}:{item.label.value}]"
    return f"[reference={item.self_ref}]"

def _default_summary(summary:str) -> str:
    return f"(summary={summary})"

class OutlineMode(str, Enum):
    """Display mode for document summary output."""

    OUTLINE = "outline"
    TABLE_OF_CONTENTS = "table_of_contents"


class OutlineParams(MarkdownParams):
    """Markdown-specific serialization parameters for outline.

    Inherits MarkdownParams to retain Markdown behaviors (escaping, links, etc.).
    """

    mode: OutlineMode = OutlineMode.OUTLINE

    
class _OutlineTextSerializer(BaseTextSerializer):
    """_Outline class for text item serializers."""

    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        prepend = ""
        if isinstance(item, TitleItem) or isinstance(item, SectionHeaderItem):
            # MarkdownDocSerializer requires a doc instance; pass through current doc
            _md_serializer = MarkdownDocSerializer(doc=doc)
            _serializer = MarkdownTextSerializer()

            res = _serializer.serialize(item=item, doc_serializer=_md_serializer, doc=doc)
            prepend = res.text
            
        summary = ""
        if item.meta and \
           (field_val := getattr(item.meta, MetaFieldName.SUMMARY)) is not None and \
           isinstance(field_val, SummaryMetaField):
            summary = _default_summary(field_val.text)
            
        reference = _default_outline_node(item)
        
        text = " ".join([prepend, reference, summary])
        
        return create_ser_result(
            text=text
        )

    """
    def _serialize_meta_field(
        self, meta: BaseMeta, name: str, mark_meta: bool
    ) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None and isinstance(
            field_val, SummaryMetaField
        ):
            txt = field_val.text
            return (
                f"[{self._humanize_text(name, title=True)}] {txt}"
                if mark_meta
                else txt
            )
        else:
            return None
    """

class _OutlineTableSerializer(BaseTableSerializer):
    """_Outline class for table item serializers."""

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(
            text=_default_outline_node(item)
        )


class _OutlinePictureSerializer(BasePictureSerializer):
    """_Outline class for picture item serializers."""

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(
            text=_default_outline_node(item)
        )


class _OutlineKeyValueSerializer(BaseKeyValueSerializer):
    """_Outline class for key value item serializers."""

    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(
            text=_default_outline_node(item)
        )


class _OutlineFormSerializer(BaseFormSerializer):
    """_Outline class for form item serializers."""

    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(
            text=_default_outline_node(item)
        )


class _OutlineListSerializer(BaseListSerializer):
    """_Outline class for list serializers."""

    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(
            text=_default_outline_node(item)
        )


class _OutlineInlineSerializer(BaseInlineSerializer):
    """_Outline class for inline serializers."""

    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(text="")


class _OutlineFallbackSerializer(BaseFallbackSerializer):
    """_Outline fallback class for item serializers."""

    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "_OutlineDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result(text="")


    
class _OutlineMetaSerializer(MarkdownMetaSerializer):

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc: DoclingDocument,
        level: Optional[int] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's meta."""
        params = MarkdownParams(**kwargs)
        return create_ser_result(
            text="\n\n".join(
                [
                    f"{'  ' * (level or 0)}[{item.self_ref}] [{item.__class__.__name__}:{item.label.value}] {tmp}"  # type:ignore[attr-defined]
                    for key in (
                        list(item.meta.__class__.model_fields)
                        + list(item.meta.get_custom_part())
                    )
                    if (
                        tmp := self._serialize_meta_field(
                            item.meta, key, params.mark_meta
                        )
                    )
                ]
                if item.meta
                else []
            ),
            span_source=item if isinstance(item, DocItem) else [],
        )

    def _serialize_meta_field(
        self, meta: BaseMeta, name: str, mark_meta: bool
    ) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None and isinstance(
            field_val, SummaryMetaField
        ):
            txt = field_val.text
            return (
                f"[{self._humanize_text(name, title=True)}] {txt}"
                if mark_meta
                else txt
            )
        else:
            return None

class OutlineDocSerializer(MarkdownDocSerializer):

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
