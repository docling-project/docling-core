"""Markdown document summary serializers (outline and TOC).

This module provides a Markdown-focused serializer that emits a compact
document outline or a table of contents derived from a Docling document.
"""

from enum import Enum
from typing import Any, Optional

from typing_extensions import override

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc import (
    CodeItem,
    DocItem,
    DocItemLabel,
    FormItem,
    GroupItem,
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
    """Display mode for document summary output."""

    OUTLINE = "outline"
    TABLE_OF_CONTENTS = "table_of_contents"


class MarkdownSummaryParams(MarkdownParams):
    """Markdown-specific serialization parameters for outline.

    Inherits MarkdownParams to retain Markdown behaviors (escaping, links, etc.).
    """

    mode: MarkdownSummaryMode = MarkdownSummaryMode.OUTLINE

    use_markdown_headers: bool = False

    add_label_counter: bool = False
    add_references: bool = True
    add_summary: bool = True

    toc_labels: list[DocItemLabel] = [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER]


class MarkdownSummarySerializer(MarkdownDocSerializer):
    """Markdown-specific document summary serializer.

    Inherits MarkdownDocSerializer to reuse Markdown formatting/post-processing
    and sub-serializers; overrides only the parts selection logic.
    """

    params: MarkdownSummaryParams = MarkdownSummaryParams()

    @override
    def get_parts(
        self,
        item: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Return a single part containing the document (or subtree) outline."""
        return self._create_document_outline(root=item, **kwargs)

    # return [create_ser_result(text=outline, span_source=[])] if outline else []

    # -------------------------
    # Helper methods (internal)
    # -------------------------

    def _next_idx(
        self, *, lbl: DocItemLabel, label_counter: dict[DocItemLabel, int]
    ) -> int:
        label_counter[lbl] = label_counter.get(lbl, 0) + 1
        return label_counter[lbl]

    def _include_label(
        self, *, params: MarkdownSummaryParams, lbl: DocItemLabel
    ) -> bool:
        """Return True if label should be included (esp. for TOC mode)."""
        if (
            params.mode == MarkdownSummaryMode.TABLE_OF_CONTENTS
            and lbl not in params.toc_labels
        ):
            return False
        return True

    def _is_node_excluded(
        self,
        *,
        node: NodeItem,
        excluded: set[str],
        params: MarkdownSummaryParams,
    ) -> bool:
        """Centralize exclusion logic applied to nodes in the outline."""
        if isinstance(node, DocItem):
            if node.self_ref in excluded:
                return True
            if (
                isinstance(node, TextItem)
                and node.self_ref in self._captions_of_some_item
            ):
                return True
            if not self._include_label(params=params, lbl=node.label):
                return True
        return False

    def _compose_node_label(
        self,
        *,
        node: NodeItem,
        params: MarkdownSummaryParams,
        label_counter: dict[DocItemLabel, int],
    ) -> str:
        """Compute the textual label for a node (without refs).

        - When ``add_label_counter`` is True, add counters for non-table/picture
          DocItems.
        - Tables/pictures are numbered separately when building the final line.
        - For groups, expose the raw normalized label but do not emit a line.
        """
        node_label = ""
        if (
            params.add_label_counter
            and isinstance(node, DocItem)
            and not isinstance(node, (TableItem, PictureItem))
        ):
            base = str(node.label).replace("_", "-")
            lbl_cnt = self._next_idx(lbl=node.label, label_counter=label_counter)
            node_label = f"{base} {lbl_cnt}"
        elif isinstance(node, (DocItem, GroupItem)):
            node_label = str(node.label).replace("_", "-")
        return node_label

    def _ref_part(self, *, node: NodeItem, params: MarkdownSummaryParams) -> str:
        return f" (reference={node.self_ref})" if params.add_references else ""

    def _strip_md_header_prefix(self, text: str) -> str:
        stripped = text.lstrip()
        while stripped.startswith("#"):
            stripped = stripped.lstrip("#").lstrip()
        return stripped

    def _line_for_title(
        self,
        *,
        node: TitleItem,
        params: MarkdownSummaryParams,
        node_label: str,
        ref_part: str,
    ) -> str:
        raw_text = self.text_serializer.serialize(
            item=node, doc_serializer=self, doc=self.doc
        ).text
        if params.use_markdown_headers:
            text = raw_text.lstrip()
            return f"{text}{ref_part}"
        text = raw_text.lstrip().lstrip("# ") if raw_text.startswith("#") else raw_text
        return (
            f"{node_label}{ref_part}: {text}"
            if params.add_references
            else f"{node_label}: {text}"
        )

    def _line_for_section_header(
        self,
        *,
        node: SectionHeaderItem,
        params: MarkdownSummaryParams,
        node_label: str,
    ) -> str:
        raw_text = self.text_serializer.serialize(
            item=node, doc_serializer=self, doc=self.doc
        ).text
        if params.use_markdown_headers:
            text = raw_text.lstrip()
            if params.add_references:
                return f"{text} (level={node.level}, reference={node.self_ref})"
            return f"{text} (level={node.level})"
        stripped = self._strip_md_header_prefix(raw_text)
        if params.add_references:
            return f"{node_label} (level={node.level}, reference={node.self_ref}): {stripped}"
        return f"{node_label} (level={node.level}): {stripped}"

    def _line_for_simple_label(self, *, node_label: str, ref_part: str) -> str:
        return f"{node_label}{ref_part}"

    def _line_for_table(
        self, *, node_label: str, ref_part: str, label_counter: dict[DocItemLabel, int]
    ) -> str:
        lbl_cnt = self._next_idx(lbl=DocItemLabel.TABLE, label_counter=label_counter)
        return f"{node_label} {lbl_cnt}{ref_part}"

    def _line_for_picture(
        self, *, node_label: str, ref_part: str, label_counter: dict[DocItemLabel, int]
    ) -> str:
        lbl_cnt = self._next_idx(lbl=DocItemLabel.PICTURE, label_counter=label_counter)
        return f"{node_label} {lbl_cnt}{ref_part}"

    def _get_summary(self, *, node: NodeItem, params: MarkdownSummaryParams) -> str:
        if (
            params.add_summary
            and (node.summary is not None)
            and isinstance(node.summary, str)
        ):
            return node.summary
        return ""

    def _create_document_outline(
        self,
        *,
        root: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Create an outline, respecting params and recursive traversal."""
        params = self.params.merge_with_patch(patch=kwargs)
        excluded = self.get_excluded_refs(**kwargs)

        label_counter: dict[DocItemLabel, int] = {}
        visited: set[str] = set()
        result: list[SerializationResult] = []

        for node, _level in self.doc.iterate_items(root=root, with_groups=True):
            if node.self_ref in visited:
                continue
            visited.add(node.self_ref)

            # Skip list items in outline
            if isinstance(node, ListItem):
                continue

            # Respect exclusion logic
            if self._is_node_excluded(node=node, excluded=excluded, params=params):
                continue

            summary = self._get_summary(node=node, params=params)
            node_label = self._compose_node_label(
                node=node, params=params, label_counter=label_counter
            )
            ref_part = self._ref_part(node=node, params=params)

            line = ""
            if isinstance(node, TitleItem):
                line = self._line_for_title(
                    node=node, params=params, node_label=node_label, ref_part=ref_part
                )
            elif isinstance(node, SectionHeaderItem):
                line = self._line_for_section_header(
                    node=node, params=params, node_label=node_label
                )
            elif isinstance(node, ListGroup):
                line = ""  # intentionally skip
            elif isinstance(node, (TextItem, FormItem, CodeItem)):
                line = self._line_for_simple_label(
                    node_label=node_label, ref_part=ref_part
                )
            elif isinstance(node, TableItem):
                line = self._line_for_table(
                    node_label=node_label,
                    ref_part=ref_part,
                    label_counter=label_counter,
                )
            elif isinstance(node, PictureItem):
                line = self._line_for_picture(
                    node_label=node_label,
                    ref_part=ref_part,
                    label_counter=label_counter,
                )

            if summary:
                line = f"{line} (summary={summary})" if line else line

            if line:
                result.append(
                    create_ser_result(
                        text=line,
                        span_source=node if isinstance(node, DocItem) else [],
                    )
                )

        return result
