"""Markdown document summary serializers (outline and TOC).

This module provides a Markdown-focused serializer that emits a compact
document outline or a table of contents derived from a Docling document.
"""

import json
import re
from enum import Enum
from typing import Annotated, Any

from pydantic import ConfigDict, Field, TypeAdapter, model_validator
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
from docling_core.types.base import _JSON_POINTER_REGEX
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
from docling_core.types.doc.document import _ExtraAllowingModel


class OutlineItemData(_ExtraAllowingModel):
    """Data model for outline item JSON representation.

    Allows extra fields for custom metadata from SummaryMetaField.
    """

    ref: Annotated[
        str,
        Field(
            pattern=_JSON_POINTER_REGEX,
            description="JSON pointer reference to the item in the document",
        ),
    ]
    item: Annotated[str, Field(description="Label of the item")]
    title: Annotated[str | None, Field(description="Title or heading text of the item")] = None
    summary: Annotated[str | None, Field(description="Summary text of the item")] = None
    level: Annotated[
        int | None,
        Field(description="Hierarchical level of the item (for titles and section headers)"),
    ] = None


def _default_prepend(item: NodeItem) -> str:
    if isinstance(item, DocItem | GroupItem):
        return f"{item.label.value} "
    else:
        raise ValueError("item is neither DocItem nor GroupItem")


def _default_outline_node(item: NodeItem) -> str:
    return f"\\[ref={item.self_ref}\\]"


def _default_summary(summary: str) -> str:
    return summary


def _serialize_text_item(item: TextItem, doc: DoclingDocument, **kwargs: Any) -> str:
    """Serialize a text item using Markdown serializers.

    Args:
        item: The text item (e.g., title or section header item) to serialize
        doc: The document containing the item
        **kwargs: Additional serialization parameters

    Returns:
        The serialized title text, stripped of leading/trailing whitespace
    """
    md_serializer = MarkdownDocSerializer(doc=doc)
    text_serializer = MarkdownTextSerializer()
    result = text_serializer.serialize(item=item, doc_serializer=md_serializer, doc=doc, **kwargs)
    return result.text.strip()


def _format_indented_text_line(item: OutlineItemData, indent_size: int = 2, max_summary_length: int = 100) -> str:
    """Format a single item as an indented text line.

    Args:
        item: An outline data point
        indent_size: Number of spaces per indentation level
        max_summary_length: Maximum length for summary text before truncation

    Returns:
        Formatted line with indentation based on level
    """
    level = item.level if item.level is not None else 1
    indent = " " * (indent_size * level)
    summary = item.summary if item.summary else ""

    # Format: [ref=...] [title] summary
    parts = [f"[ref={item.ref}]"]
    if item.title is not None:
        parts.append(f"[{item.title}]")
    if summary:
        # Truncate summary if too long, keep first part
        if len(summary) <= max_summary_length:
            summary_text = summary
        else:
            summary_text = summary[: max_summary_length - 3] + "..."
        parts.append(summary_text)

    return indent + " ".join(parts)


def _extract_ref_from_markdown(text: str) -> tuple[str | None, int | None]:
    """Extract reference and level from Markdown outline text.

    Args:
        text: Markdown text containing a reference like \\[ref=#/texts/1\\]

    Returns:
        Tuple of (reference, level) where level is extracted from heading markers
    """
    # The text contains \\[ref=...\\] (escaped brackets in Markdown)
    # Match pattern: \[ref=#/path/to/item\]
    ref_pattern = r"\\+\[ref=(#/[\w/-]+)\\+\]"
    ref_match = re.search(ref_pattern, text)
    ref = ref_match.group(1) if ref_match else None

    # Extract level from heading markers (# for level 1, ## for level 2, etc.)
    # Title is represented as # (level 1), section headers as ## (level 1), ### (level 2), etc.
    level = None
    lines = text.split("\n")
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("#"):
            # Count the number of # characters
            hash_count = len(first_line) - len(first_line.lstrip("#"))
            if hash_count == 1:
                # Single # is the title (level 1)
                level = 1
            else:
                # Multiple ## represent section headers (level = hash_count - 1)
                level = hash_count - 1

    return ref, level


def _extract_item_info_from_part(part: SerializationResult) -> tuple[str | None, int | None, str | None]:
    """Extract item reference, level, and type from a serialization part.

    Args:
        part: A serialization result part

    Returns:
        Tuple of (item_ref, item_level, item_type)
    """
    if not part.text:
        return None, None, None

    item_ref = None
    item_level = None
    item_type = None

    # First try JSON parsing (for JSON/ITXT formats)
    try:
        item_data = json.loads(part.text)
        item_ref = item_data.get("ref")
        item_level = item_data.get("level")
        item_type = item_data.get("item")
        return item_ref, item_level, item_type
    except (json.JSONDecodeError, ValueError):
        pass

    # Not JSON, try to get from spans (for Markdown format)
    if part.spans and len(part.spans) > 0:
        first_span = part.spans[0]
        item_ref = first_span.item.self_ref
        # Get level and type if it's a section header or title
        if isinstance(first_span.item, SectionHeaderItem):
            item_level = first_span.item.level
            item_type = "section_header"
        elif isinstance(first_span.item, TitleItem):
            item_level = 1
            item_type = "title"
        elif isinstance(first_span.item, DocItem | GroupItem):
            item_type = first_span.item.label.value if hasattr(first_span.item, "label") else None
    else:
        # Spans are empty, extract from Markdown text
        item_ref, extracted_level = _extract_ref_from_markdown(part.text)
        if extracted_level is not None:
            item_level = extracted_level
            # Determine item type based on level
            if extracted_level == 1:
                item_type = "title"
            else:
                item_type = "section_header"

    return item_ref, item_level, item_type


def _default_text(item: NodeItem, doc: DoclingDocument, **kwargs: Any) -> str:
    if isinstance(item, ListItem):
        return ""

    params = OutlineParams(**kwargs)

    # Extract title/heading text once if needed
    title_text: str | None = None
    if params.include_non_meta and isinstance(item, TitleItem | SectionHeaderItem):
        title_text = (
            item.text
            if params.format in (OutlineFormat.JSON, OutlineFormat.ITXT)
            else _serialize_text_item(item, doc, **kwargs)
        )

    # For JSON and ITXT formats, return structured data as JSON string
    # (ITXT will be assembled at document level in serialize_doc)
    if params.format in (OutlineFormat.JSON, OutlineFormat.ITXT):
        # Prepare base fields
        label: str | None = item.label if isinstance(item, DocItem | GroupItem) else None
        data_dict: dict[str, Any] = {
            "ref": item.self_ref,
            "item": label,
            "title": title_text,
            "summary": item.meta.summary.text if item.meta and item.meta.summary else None,
        }

        # Add level for hierarchical items (for ITXT indentation)
        if isinstance(item, TitleItem):
            data_dict["level"] = 1
        elif isinstance(item, SectionHeaderItem):
            data_dict["level"] = item.level

        # Add extra custom fields from summary metadata
        if item.meta and item.meta.summary:
            extra_dict: dict[str, Any] = item.meta.summary.get_custom_part()
            if extra_dict:
                data_dict.update(extra_dict)

        # Build the outline item data model with all fields (including extras)
        outline_data = OutlineItemData(**data_dict)
        data = outline_data.model_dump(exclude_none=True)

        # Return as JSON string (will be parsed and reassembled in serialize_doc)
        return json.dumps(data, ensure_ascii=False)

    text_parts = []

    # Only include prepend (actual text content) if include_non_meta is True
    if params.include_non_meta:
        if title_text is not None:
            text_parts.append(title_text)
        else:
            text_parts.append(_default_prepend(item))

    # Add two trailing spaces for Markdown line break.
    text_parts.append(_default_outline_node(item) + "  ")

    # Always include summary (metadata) if available
    if item.meta and item.meta.summary:
        text_parts.append(_default_summary(item.meta.summary.text))

    return "\n".join(text_parts).strip()


class OutlineMode(str, Enum):
    """Display modes for document outline serialization.

    Controls which document elements are included in the serialized output.
    Choose based on whether you need a complete document overview or just navigation structure.

    Attributes:
        OUTLINE: Outline mode includes all document elements (text, tables, figures, etc.).
            Provides a comprehensive view of the entire document structure.
        TABLE_OF_CONTENTS: Table of contents mode shows only titles and section headers.
            Ideal for navigation, document structure analysis, or generating a TOC.
    """

    OUTLINE = "outline"
    TABLE_OF_CONTENTS = "table_of_contents"


class OutlineFormat(str, Enum):
    """Output formats for outline serialization.

    Controls how the outline data is formatted and presented.
    Choose based on your use case: human readability, machine processing, or debugging.

    Attributes:
        MARKDOWN: Markdown format produces human-readable text with proper formatting.
            Each item appears on separate lines with title, reference, and summary.
            Best for documentation, reports, or human consumption.
        JSON: JSON format provides structured data as a flat array of objects.
            Each object contains ref, title, summary, level, and optional custom fields.
            Best for programmatic processing, APIs, or data exchange.
        ITXT: Indented text format creates a hierarchical view using indentation.
            Items are indented based on their level (3 spaces per level).
            Format: [ref=...] [title] summary (truncated to configurable length).
            Best for debugging, logging, or quick visual inspection of document structure.
    """

    MARKDOWN = "markdown"
    JSON = "json"
    ITXT = "itxt"


class OutlineParams(MarkdownParams):
    """Markdown-specific serialization parameters for outline.

    Inherits MarkdownParams to retain Markdown behaviors (escaping, links, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Annotated[
        OutlineMode,
        Field(
            description=(
                "Display mode: 'outline' includes all document elements, "
                "'table_of_contents' shows only titles and section headers"
            )
        ),
    ] = OutlineMode.OUTLINE
    format: Annotated[
        OutlineFormat,
        Field(
            description=(
                "Output format: 'markdown' for human-readable text, "
                "'json' for structured data, 'itxt' for hierarchical indented text"
            )
        ),
    ] = OutlineFormat.MARKDOWN
    itxt_max_summary_length: Annotated[
        int,
        Field(
            description=(
                "Maximum length for summary text in ITXT format. "
                "Summaries longer than this will be truncated with '...'"
            ),
            ge=10,
        ),
    ] = 100
    start_item: Annotated[
        NodeItem | None,
        Field(
            description=(
                "Optional starting node item for the outline. "
                "If provided, only this item and its children (recursively) will be included. "
                "If None (default), the outline starts from the document root (#/body)."
            ),
        ),
    ] = None
    max_level: Annotated[
        int | None,
        Field(
            description=(
                "Optional maximum heading level to include in the outline. "
                "If provided, only headings with level <= max_level will be included "
                "(along with their children). Level 0 represents the document root. "
                "If None (default), all heading levels are included."
            ),
            ge=0,
        ),
    ] = None

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

        text = _default_text(item=item, doc=doc, **kwargs)
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

        text = _default_text(item=item, doc=doc, **kwargs)
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

        text = _default_text(item=item, doc=doc, **kwargs)
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

        text = _default_text(item=item, doc=doc, **kwargs)
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
        text = _default_text(item=item, doc=doc, **kwargs)
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
    def serialize(
        self,
        *,
        item: NodeItem | None = None,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: set[str] | None = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a given node."""

        # Always use parent implementation - it will call get_parts which handles filtering
        return super().serialize(
            item=item,
            list_level=list_level,
            is_inline_scope=is_inline_scope,
            visited=visited,
            **kwargs,
        )

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts.

        For JSON format, combines individual JSON objects into a flat array.
        For ITXT format, creates indented text based on level field.
        For Markdown format, uses the default behavior.
        """
        params = self.params.merge_with_patch(patch=kwargs)
        OutlineT = TypeAdapter(list[OutlineItemData])

        # Check if we're serializing from body by looking at kwargs
        # If 'from_non_body' is set, don't add body-level summary
        from_non_body = kwargs.get("_from_non_body", False)

        if params.format in (OutlineFormat.JSON, OutlineFormat.ITXT):
            outline: list[OutlineItemData] = []

            # Add body-level summary if present and we're serializing from body
            # Don't add it if start_item is specified (we're filtering to a subtree)
            if not from_non_body and not params.start_item and self.doc.body.meta and self.doc.body.meta.summary:
                body_data: dict[str, Any] = {
                    "ref": self.doc.body.self_ref,
                    "item": DocItemLabel.SECTION_HEADER,
                    "title": self.doc.name if params.include_non_meta else None,
                    "summary": self.doc.body.meta.summary.text,
                    "level": 0,
                }
                extra_dict: dict[str, Any] = self.doc.body.meta.summary.get_custom_part()
                if extra_dict:
                    body_data.update(extra_dict)

                outline_data = OutlineItemData(**body_data)
                outline.append(outline_data)

            # Add all the other parts
            for part in parts:
                if part.text:
                    # Check if the part is already a JSON array (from recursive serialize_doc call)
                    try:
                        parsed = json.loads(part.text)
                        if isinstance(parsed, list):
                            # It's already a list of OutlineItemData, extend our outline
                            for item_dict in parsed:
                                outline.append(OutlineItemData(**item_dict))
                        else:
                            # It's a single OutlineItemData object
                            outline.append(OutlineItemData.model_validate_json(part.text))
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, skip it
                        pass

            if params.format == OutlineFormat.JSON:
                text_res: str = OutlineT.dump_json(outline, exclude_none=True, ensure_ascii=False, indent=2).decode()
            else:
                # For ITXT format, normalize levels when start_item is specified
                # so that the starting item has level 0 (no indentation)
                if params.start_item and outline:
                    # Find the minimum level in the outline
                    min_level = min(item.level if item.level is not None else 0 for item in outline)
                    # Adjust all levels by subtracting the minimum
                    for item in outline:
                        if item.level is not None:
                            item.level = item.level - min_level

                lines = [
                    _format_indented_text_line(item, max_summary_length=params.itxt_max_summary_length)
                    for item in outline
                ]
                text_res = "\n".join(lines)

            return create_ser_result(text=text_res, span_source=parts)
        else:
            all_parts = []

            if not from_non_body and not params.start_item and self.doc.body.meta and self.doc.body.meta.summary:
                body_text_parts = []

                if params.include_non_meta:
                    body_text_parts.append(f"# {self.doc.name}")
                # Add reference with two trailing spaces for Markdown line break
                body_text_parts.append(f"\\[ref={self.doc.body.self_ref}\\]  ")
                body_text_parts.append(_default_summary(self.doc.body.meta.summary.text))
                body_text = "\n".join(body_text_parts).strip()
                all_parts.append(create_ser_result(text=body_text))

            all_parts.extend(parts)

            # Use default Markdown behavior with all parts
            return super().serialize_doc(parts=all_parts, **kwargs)

    def _filter_by_start_item(
        self, parts: list[SerializationResult], start_item: NodeItem
    ) -> list[SerializationResult]:
        """Filter parts to include only the start item and its descendants.

        Args:
            parts: List of serialization parts to filter
            start_item: The item to start from

        Returns:
            Filtered list of parts
        """
        filtered_parts = []
        found_start = False
        start_level = None

        # Get the level of the start item if it's a section header
        if isinstance(start_item, SectionHeaderItem):
            start_level = start_item.level

        for part in parts:
            item_ref, item_level, _ = _extract_item_info_from_part(part)

            # Skip if we couldn't extract item_ref
            if item_ref is None:
                continue

            # Check if this is the start item
            if not found_start:
                if item_ref == start_item.self_ref:
                    found_start = True
                    filtered_parts.append(part)
                continue

            # After finding start, include descendants based on level
            if start_level is not None and item_level is not None:
                if item_level <= start_level:
                    break
                filtered_parts.append(part)
            else:
                filtered_parts.append(part)

        return filtered_parts

    def _filter_by_max_level(self, parts: list[SerializationResult], max_level: int) -> list[SerializationResult]:
        """Filter parts to include only headings up to max_level and their children.

        Args:
            parts: List of serialization parts to filter
            max_level: Maximum heading level to include

        Returns:
            Filtered list of parts
        """
        filtered_parts = []
        include_children = True  # Track whether to include children of current heading

        for part in parts:
            _, item_level, item_type = _extract_item_info_from_part(part)

            # Skip if we couldn't extract item_type
            if item_type is None:
                continue

            # Check if it's a heading item
            if item_type == "section_header" and item_level is not None:
                include_children = item_level <= max_level

                # Include this heading if it's within max_level
                if include_children:
                    filtered_parts.append(part)
            elif item_type == "title":
                # Titles are always included if max_level >= 1
                if max_level >= 1:
                    filtered_parts.append(part)
                    include_children = True
                else:
                    include_children = False
            else:
                # For non-heading items, include only if we're including children
                # of the current heading
                if include_children:
                    filtered_parts.append(part)

        return filtered_parts

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

        Also handles filtering based on start_item and max_level parameters.
        """
        # Get start_item BEFORE merge_with_patch to avoid serialization issues
        start_item_filter = self.params.start_item

        params = self.params.merge_with_patch(patch=kwargs)

        kwargs_with_meta = {**kwargs, "include_non_meta": True}

        # Always get parts from body (or the item parameter if provided)
        # We'll filter for start_item afterwards
        all_parts = super().get_parts(
            item=item,
            traverse_pictures=traverse_pictures,
            list_level=list_level,
            is_inline_scope=is_inline_scope,
            visited=visited,
            **kwargs_with_meta,
        )

        # Filter based on start_item if specified
        if start_item_filter is not None:
            all_parts = self._filter_by_start_item(all_parts, start_item_filter)

        # Filter based on max_level if specified
        if params.max_level is not None:
            all_parts = self._filter_by_max_level(all_parts, params.max_level)

        return all_parts
