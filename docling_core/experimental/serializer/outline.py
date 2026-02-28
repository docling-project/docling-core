"""Markdown document summary serializers (outline and TOC).

This module provides a Markdown-focused serializer that emits a compact
document outline or a table of contents derived from a Docling document.
"""

from enum import Enum
from typing import Any, Optional

from typing_extensions import override

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
    MetaFieldName,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableItem,
    TextItem,
    TitleItem,
)


def _default_prepend(item: NodeItem) -> str:
    if isinstance(item, DocItem) or isinstance(item, GroupItem):
        return f"{item.label.value} "
    else:
        raise ValueError("item is nor DocItem nor GroupItem")
    # return f"[{item.self_ref}] [{item.__class__.__name__}:{item.label.value}]"
    # return f"[reference={item.self_ref}]"


def _default_outline_node(item: NodeItem) -> str:
    # return f"[{item.self_ref}] [{item.__class__.__name__}:{item.label.value}]"
    return f"[reference={item.self_ref}]"


def _default_summary(summary: str) -> str:
    return f"(summary={summary})"


def _default_text(item: NodeItem, doc: DoclingDocument, **kwargs: Any) -> str:
    if isinstance(item, ListItem):
        return ""

    prepend = _default_prepend(item)
    if isinstance(item, TitleItem) or isinstance(item, SectionHeaderItem):
        # MarkdownDocSerializer requires a doc instance; pass through current doc
        _md_serializer = MarkdownDocSerializer(doc=doc)
        _serializer = MarkdownTextSerializer()

        res = _serializer.serialize(
            item=item, doc_serializer=_md_serializer, doc=doc, **kwargs
        )
        prepend = res.text

    summary = ""
    if (
        item.meta
        and (field_val := getattr(item.meta, MetaFieldName.SUMMARY)) is not None
        and isinstance(field_val, SummaryMetaField)
    ):
        summary = _default_summary(field_val.text)

    reference = _default_outline_node(item)

    text = " ".join([prepend, reference, summary])

    return text.strip()


class OutlineMode(str, Enum):
    """Display mode for document summary output."""

    OUTLINE = "outline"
    TABLE_OF_CONTENTS = "table_of_contents"


class OutlineParams(MarkdownParams):
    """Markdown-specific serialization parameters for outline.

    Inherits MarkdownParams to retain Markdown behaviors (escaping, links, etc.).
    """

    mode: OutlineMode = OutlineMode.OUTLINE

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        """Adjust allowed labels based on the selected mode."""
        # Adjust allowed labels based on mode
        if self.mode == OutlineMode.TABLE_OF_CONTENTS:
            self.labels = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}


class _OutlineTextSerializer(BaseTextSerializer):
    """_Outline class for text item serializers."""

    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # print(kwargs)

        text = _default_text(item=item, doc=doc, **kwargs)
        return create_ser_result(text=text)


class _OutlineTableSerializer(BaseTableSerializer):
    """_Outline class for table item serializers."""

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.TABLE not in params.labels:
            return create_ser_result(text="")

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlinePictureSerializer(BasePictureSerializer):
    """_Outline class for picture item serializers."""

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.PICTURE not in params.labels:
            return create_ser_result(text="")

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineKeyValueSerializer(BaseKeyValueSerializer):
    """_Outline class for key value item serializers."""

    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.KEY_VALUE_REGION not in params.labels:
            return create_ser_result(text="")

        print("label: ", item.label)

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineFormSerializer(BaseFormSerializer):
    """_Outline class for form item serializers."""

    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = OutlineParams(**kwargs)
        if DocItemLabel.FORM not in params.labels:
            return create_ser_result(text="")

        text = _default_text(item=item, doc=doc)
        return create_ser_result(text=text)


class _OutlineListSerializer(BaseListSerializer):
    """_Outline class for list serializers."""

    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # Intentionally skip list containers in outlines
        return create_ser_result(text="")


class _OutlineInlineSerializer(BaseInlineSerializer):
    """_Outline class for inline serializers."""

    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
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
        doc_serializer: "BaseDocSerializer",
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
        level: Optional[int] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's meta."""
        return create_ser_result(text="")

    def _serialize_meta_field(
        self, meta: BaseMeta, name: str, mark_meta: bool
    ) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None and isinstance(
            field_val, SummaryMetaField
        ):
            txt = field_val.text
            return (
                f"[{self._humanize_text(name, title=True)}] {txt}" if mark_meta else txt
            )
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
