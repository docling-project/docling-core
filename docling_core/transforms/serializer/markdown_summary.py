from typing import Any, Optional, Union
from pathlib import Path

from pydantic import AnyUrl
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    create_ser_result,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownAnnotationSerializer,
    MarkdownFallbackSerializer,
    MarkdownFormSerializer,
    MarkdownInlineSerializer,
    MarkdownKeyValueSerializer,
    MarkdownListSerializer,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
    MarkdownTextSerializer,
)
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)


class MarkdownSummaryParams(CommonParams):
    """Markdown-specific serialization parameters for outline."""

    use_markdown_headers: bool = False


class MarkdownSummarySerializer(DocSerializer):
    """Markdown-specific document summary serializer."""

    # Provide required serializer attributes to satisfy DocSerializer’s model
    text_serializer: BaseTextSerializer = MarkdownTextSerializer()
    table_serializer: BaseTableSerializer = MarkdownTableSerializer()
    picture_serializer: BasePictureSerializer = MarkdownPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = MarkdownKeyValueSerializer()
    form_serializer: BaseFormSerializer = MarkdownFormSerializer()
    fallback_serializer: BaseFallbackSerializer = MarkdownFallbackSerializer()

    list_serializer: BaseListSerializer = MarkdownListSerializer()
    inline_serializer: BaseInlineSerializer = MarkdownInlineSerializer()

    annotation_serializer: BaseAnnotationSerializer = MarkdownAnnotationSerializer()

    params: MarkdownSummaryParams = MarkdownSummaryParams()

    @override
    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        return f"**{text}**"

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        return f"*{text}*"

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        return f"~~{text}~~"

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ) -> str:
        return f"[{text}]({str(hyperlink)})"

    @override
    def get_parts(
        self,
        item: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Return a single part containing the document (or subtree) outline."""
        outline = self._create_document_outline(root=item, **kwargs)
        return [create_ser_result(text=outline, span_source=[])] if outline else []

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        text_res = "\n\n".join([p.text for p in parts if p.text])
        return create_ser_result(text=text_res, span_source=parts)

    def _create_document_outline(
        self,
        *,
        root: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> str:
        """Create an outline, respecting params and recursive traversal."""
        params = self.params.merge_with_patch(patch=kwargs)
        excluded = self.get_excluded_refs(**kwargs)

        label_counter: dict[DocItemLabel, int] = {
            DocItemLabel.TABLE: 0,
            DocItemLabel.PICTURE: 0,
            DocItemLabel.TEXT: 0,
        }
        lines: list[str] = []
        visited: set[str] = set()

        # Iterate depth-first with groups, similar to MarkdownSerializer
        for node, level in self.doc.iterate_items(root=root, with_groups=True):
            if node.self_ref in visited:
                continue
            visited.add(node.self_ref)

            # Skip list items in outline
            if isinstance(node, ListItem):
                continue

            # Respect excluded refs and skip caption text items
            if isinstance(node, DocItem):
                if node.self_ref in excluded:
                    continue
                if isinstance(node, TextItem) and node.self_ref in self._captions_of_some_item:
                    continue

            if isinstance(node, TitleItem):
                if params.use_markdown_headers:
                    lines.append(f"# {node.text}")
                else:
                    lines.append(f"title (reference={node.self_ref}): {node.text}")
            elif isinstance(node, SectionHeaderItem):
                if params.use_markdown_headers:
                    hashes = "#" * (node.level + 1)
                    lines.append(f"{hashes} {node.text}")
                else:
                    lines.append(
                        f"section-header (level={node.level}, reference={node.self_ref}): {node.text}"
                    )
            elif isinstance(node, TextItem):
                lines.append(f"{node.label} (reference={node.self_ref})")
            elif isinstance(node, TableItem):
                label_counter[DocItemLabel.TABLE] += 1
                lines.append(
                    f"{node.label} {label_counter[DocItemLabel.TABLE]} (reference={node.self_ref})"
                )
            elif isinstance(node, PictureItem):
                label_counter[DocItemLabel.PICTURE] += 1
                lines.append(
                    f"{node.label} {label_counter[DocItemLabel.PICTURE]} (reference={node.self_ref})"
                )

        return "\n\n".join(lines)
