from typing import Any, Optional, Union
from pathlib import Path
from enum import Enum

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
    CodeItem,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FormItem,
    ListGroup,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

class MarkdownSummaryMode(str, Enum):
    
    OUTLINE = "outline"
    TABLE_OF_CONTENTS = "table_of_contents"
    
class MarkdownSummaryParams(CommonParams):
    """Markdown-specific serialization parameters for outline."""

    mode: MarkdownSummaryMode = MarkdownSummaryMode.OUTLINE
    
    use_markdown_headers: bool = False

    add_label_counter: bool = False
    add_references: bool = True
    add_summary: bool = True

    # Indentation control: when enabled, indent each line according to
    # the latest encountered section-header level (title treated as level 0).
    indent_by_section_level: bool = False
    indent_size: int = 2

    toc_labels: list[DocItemLabel] = [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER]
    

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
    def requires_page_break(self) -> bool:
        """Whether to add page breaks."""
        return False
    
    @override
    def get_parts(
        self,
        item: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Return a single part containing the document (or subtree) outline."""
        return self._create_document_outline(root=item, **kwargs)
    #return [create_ser_result(text=outline, span_source=[])] if outline else []

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
    ) -> list[SerializationResult]:
        """Create an outline, respecting params and recursive traversal."""
        params = self.params.merge_with_patch(patch=kwargs)
        excluded = self.get_excluded_refs(**kwargs)

        # Per-label counters; used consistently when params.add_label_counter is True
        # and always for table/picture numbering.
        label_counter: dict[DocItemLabel, int] = {}
        lines: list[str] = []
        visited: set[str] = set()

        result: list[SerializationResult] = []

        # Track latest section header level for indentation
        current_section_level: int = 0

        # Helper to increment and fetch the counter for a given label
        def _next_idx(lbl: DocItemLabel) -> int:
            label_counter[lbl] = label_counter.get(lbl, 0) + 1
            return label_counter[lbl]
        
        # Helper to identify if the label should be included in the table-of-contents
        def _include(lbl: DocItemLabel) -> int:
            if params.mode==MarkdownSummaryMode.TABLE_OF_CONTENTS and \
               (lbl not in params.toc_labels):
                return False

            return True
        
        # Iterate depth-first with groups, similar to MarkdownSerializer
        for node, level in self.doc.iterate_items(root=root, with_groups=True):
            if node.self_ref in visited:
                continue
            
            visited.add(node.self_ref)

            if not _include(lbl=node.label):
                continue
            
            summary = ""
            if params.add_summary and \
               (node.summary is not None) and \
               isinstance(node.summary, str):
                summary = node.summary
            
            # Skip list items in outline
            if isinstance(node, ListItem):
                continue

            # Respect excluded refs
            if isinstance(node, DocItem):
                if node.self_ref in excluded:
                    continue
                if isinstance(node, TextItem) and node.self_ref in self._captions_of_some_item:
                    continue

            line:str = ""

            # Base label string (normalize underscores to hyphens)
            node_label = str(node.label).replace("_", "-")
            if params.add_label_counter and not isinstance(node, (TableItem, PictureItem)):
                # Apply generic counters to non-table/picture items
                node_label = f"{node_label} {_next_idx(node.label)}"

            # Build optional reference snippet only when enabled
            ref_part = f" (reference={node.self_ref})" if params.add_references else ""
                
            if isinstance(node, TitleItem):

                raw_text = self.text_serializer.serialize(
                    item=node, doc_serializer=self, doc=self.doc
                ).text

                if params.use_markdown_headers:
                    # raw_text already includes the heading marker
                    text = raw_text.lstrip()
                    line = f"{text}{ref_part}"
                else:
                    # strip leading markdown header markers for verbose representation
                    text = raw_text.lstrip().lstrip("# ") if raw_text.startswith("#") else raw_text
                    if params.add_references:
                        line = f"{node_label}{ref_part}: {text}"
                    else:
                        line = f"{node_label}: {text}"

            elif isinstance(node, SectionHeaderItem):

                raw_text = self.text_serializer.serialize(
                    item=node, doc_serializer=self, doc=self.doc
                ).text

                if params.use_markdown_headers:
                    # raw_text already includes the correct number of '#'
                    text = raw_text.lstrip()
                    if params.add_references:
                        line = f"{text} (level={node.level}, reference={node.self_ref})"
                    else:
                        line = f"{text} (level={node.level})"
                else:
                    # strip leading markdown header markers for verbose representation
                    stripped = raw_text.lstrip()
                    while stripped.startswith("#"):
                        stripped = stripped.lstrip("#").lstrip()
                    text = stripped
                    if params.add_references:
                        line = f"{node_label} (level={node.level}, reference={node.self_ref}): {text}"
                    else:
                        line = f"{node_label} (level={node.level}): {text}"

                # Update current section level for subsequent items
                current_section_level = node.level

            elif isinstance(node, ListGroup):
                # Skip listing list groups in summary to avoid leading list noise
                line = ""
                    
            elif isinstance(node, TextItem):
                line = f"{node_label}{ref_part}"

            elif isinstance(node, FormItem):
                line = f"{node_label}{ref_part}"

            elif isinstance(node, CodeItem):
                line = f"{node_label}{ref_part}"
                
            elif isinstance(node, TableItem):
                # Tables are always numbered in the summary
                line = f"{node_label} {_next_idx(DocItemLabel.TABLE)}{ref_part}"
                
            elif isinstance(node, PictureItem):
                # Pictures are always numbered in the summary
                line = f"{node_label} {_next_idx(DocItemLabel.PICTURE)}{ref_part}"

            if len(summary)>0:
                line += f" (summary={summary})"

            # Apply indentation based on latest section level if enabled
            if params.indent_by_section_level:
                indent_level = current_section_level
                # For a section-header, indent by its own level
                if isinstance(node, SectionHeaderItem):
                    indent_level = node.level
                indent = " " * (params.indent_size * indent_level)
                line = f"{indent}{line}" if line else line

            if line:
                result.append(
                    create_ser_result(
                        text=line,
                        span_source=node if isinstance(node, DocItem) else [],
                    )
                )
            
        return result
