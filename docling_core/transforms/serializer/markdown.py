"""Define classes for Markdown serialization."""

import html
import logging
import re
import textwrap
from enum import Enum
from pathlib import Path, PurePath
from typing import Annotated, Any, Final, Optional, Union
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import AnyUrl, BaseModel, Field, PositiveInt
from tabulate import _column_type, tabulate
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
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
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    _get_annotation_text,
    _PageBreakSerResult,
    _should_use_legacy_annotations,
    create_ser_result,
)
from docling_core.types.doc import (
    BaseMeta,
    CodeItem,
    ContentLayer,
    DescriptionAnnotation,
    DescriptionMetaField,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FieldItem,
    FieldRegionItem,
    FloatingItem,
    Formatting,
    FormItem,
    FormulaItem,
    GroupItem,
    ImageRef,
    ImageRefMode,
    InlineGroup,
    KeyValueItem,
    KeywordsMetaField,
    LanguageMetaField,
    ListGroup,
    ListItem,
    MoleculeMetaField,
    NodeItem,
    PictureClassificationData,
    PictureClassificationMetaField,
    PictureItem,
    PictureMoleculeData,
    PictureTabularChartData,
    RichTableCell,
    SectionHeaderItem,
    SummaryMetaField,
    TableItem,
    TabularChartMetaField,
    TextItem,
    TitleItem,
    TopicsMetaField,
)

_logger = logging.getLogger(__name__)


def _cell_content_has_table(item: NodeItem, doc: DoclingDocument) -> bool:
    """Return True if *item* is, or has a descendant that is, a TableItem."""
    if isinstance(item, TableItem):
        return True
    elif isinstance(item, NodeItem):
        for child_ref in item.children:
            if _cell_content_has_table(child_ref.resolve(doc=doc), doc):
                return True
    return False


def _mark_subtree_visited(
    item: NodeItem,
    doc: DoclingDocument,
    visited: set[str],
) -> None:
    """Recursively add *item* and all its descendants to *visited*.

    When a nested table inside a RichTableCell is flattened, its items are
    never passed through the normal serialize() path that would mark them
    visited.  Calling this keeps the visited set consistent so the document
    serializer does not emit those items again at the top level.
    """
    if isinstance(item, NodeItem):
        visited.add(item.self_ref)
        for child_ref in item.children:
            _mark_subtree_visited(child_ref.resolve(doc=doc), doc, visited)


def _collect_subtree_text(item: NodeItem, doc: DoclingDocument) -> str:
    """Collect all text from *item*'s subtree, flattening nested tables.

    Returns a space-joined string of every piece of text found so that the
    content of a nested table is preserved in a flat, readable form.

    For TableItems the text is pulled from ``data.grid`` cells directly;
    children are *not* recursed into because they duplicate the grid content
    for RichTableCells.  For all other items, ``.text`` is collected and
    children are visited recursively.
    """
    parts: list[str] = []

    if isinstance(item, TableItem):
        for row in item.data.grid:
            for cell in row:
                if cell.text:
                    parts.append(cell.text)
        return " ".join(parts)

    if isinstance(item, TextItem) and item.text:
        parts.append(item.text)

    if isinstance(item, NodeItem):
        for child_ref in item.children:
            child = child_ref.resolve(doc=doc)
            child_text = _collect_subtree_text(child, doc)
            if child_text:
                parts.append(child_text)

    return " ".join(parts)


class OrigListItemMarkerMode(str, Enum):
    """Display mode for original list item marker."""

    NEVER = "never"
    ALWAYS = "always"
    AUTO = "auto"


class MarkdownParams(CommonParams):
    """Markdown-specific serialization parameters."""

    layers: set[ContentLayer] = {ContentLayer.BODY}
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
    image_placeholder: str = "<!-- image -->"
    enable_chart_tables: bool = True
    indent: int = 4
    wrap_width: Optional[PositiveInt] = None
    page_break_placeholder: Optional[str] = None  # e.g. "<!-- page break -->"
    escape_underscores: bool = True
    escape_html: bool = True
    mark_meta: bool = Field(default=False, description="Mark meta sections.")
    include_annotations: bool = Field(
        default=True,
        description="Include item annotations.",
        deprecated="Use include_meta instead.",
    )
    mark_annotations: bool = Field(
        default=False,
        description="Mark annotation sections.",
        deprecated="Use mark_meta instead.",
    )
    orig_list_item_marker_mode: OrigListItemMarkerMode = OrigListItemMarkerMode.AUTO
    ensure_valid_list_item_marker: bool = True
    format_code_blocks: bool = Field(
        default=True,
        description="Whether to wrap code items in markdown code block formatting (```). ",
    )
    compact_tables: Annotated[
        bool,
        Field(
            description=(
                "Whether to use compact table format without column padding. "
                "When False (default), tables use padded columns for better visual formatting. "
                "When True, tables use minimal whitespace, which is better for large tables and downstream processing."
            )
        ),
    ] = False
    include_picture_classification: bool = Field(
        default=True,
        description="Include the picture classification prediction (the image's predicted class).",
    )


class MarkdownTextSerializer(BaseModel, BaseTextSerializer):
    """Markdown-specific text item serializer."""

    def _md_line_breaks(self, text: str) -> str:
        """Replace single newlines with GFM hard line breaks (two trailing spaces).

        A single `\\n` becomes `"  \\n"` (two trailing spaces) so Markdown
        renderers honour the line break.  Double newlines (`\\n\\n`) are left
        intact because they represent a paragraph break, which is already handled
        by the document serializer joining parts with `"\\n\\n"`.
        Override to disable or change this behaviour in subclasses.
        """
        paragraphs = text.split("\n\n")
        processed = [para.replace("\n", "  \n") for para in paragraphs]
        return "\n\n".join(processed)

    def _heading_line_breaks(self, text: str) -> str:
        """Replace newlines in heading text with a space.

        GFM headings cannot span multiple lines, so `\\n` is collapsed to a
        space rather than a hard line break.  Override to change this behaviour
        in subclasses.
        """
        return text.replace("\n", " ")

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        in_table_cell: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the passed text item to Markdown.

        Args:
            item: The text item to serialize.
            doc_serializer: The parent document serializer.
            doc: The document the item belongs to.
            is_inline_scope: Whether serialization happens in an inline context
                (e.g. inside an InlineGroup). Affects delimiter and code/formula
                wrapping.
            in_table_cell: Whether the item is being rendered inside a table
                cell. When ``True``, heading markers are suppressed because the
                Markdown spec does not allow headings inside tables.
            visited: Set of already-visited item refs used to prevent duplicate
                serialization.
            **kwargs: Additional keyword arguments forwarded to
                ``MarkdownParams``.

        Returns:
            The serialization result containing the rendered Markdown text.
        """
        my_visited = visited if visited is not None else set()
        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []
        escape_html = True
        escape_underscores = True

        has_inline_repr = (
            item.text == ""
            and len(item.children) == 1
            and isinstance((child_group := item.children[0].resolve(doc)), InlineGroup)
        )
        if has_inline_repr:
            text = doc_serializer.serialize(item=child_group, visited=my_visited).text
            processing_pending = False
        else:
            text = item.text
            processing_pending = True

        if item.label == DocItemLabel.CHECKBOX_SELECTED:
            text = f"- [x] {text}"
        if item.label == DocItemLabel.CHECKBOX_UNSELECTED:
            text = f"- [ ] {text}"
        if isinstance(item, ListItem | TitleItem | SectionHeaderItem):
            if not has_inline_repr:
                # case where processing/formatting should be applied first (in inner scope)
                if isinstance(item, TitleItem | SectionHeaderItem):
                    # Headings cannot span multiple lines; replace newlines with a
                    # space so "Hello\nWorld" becomes "# Hello World", not "# Hello\nWorld".
                    text = self._heading_line_breaks(text)
                elif isinstance(item, ListItem):
                    # Apply GFM hard line breaks inside list item text before
                    # post_process wraps it in formatting/hyperlink markers.
                    text = self._md_line_breaks(text)
                text = doc_serializer.post_process(
                    text=text,
                    escape_html=escape_html,
                    escape_underscores=escape_underscores,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
                )
                processing_pending = False

            if isinstance(item, ListItem):
                pieces: list[str] = []
                case_auto = params.orig_list_item_marker_mode == OrigListItemMarkerMode.AUTO and bool(
                    re.search(r"[a-zA-Z0-9]", item.marker)
                )
                case_already_valid = (
                    params.ensure_valid_list_item_marker
                    and params.orig_list_item_marker_mode != OrigListItemMarkerMode.NEVER
                    and (item.marker in ["-", "*", "+"] or re.fullmatch(r"\d+\.", item.marker))
                )

                # wrap with outer marker (if applicable)
                if params.ensure_valid_list_item_marker and not case_already_valid:
                    md_marker = "-"
                    if item.parent is None:
                        _logger.warning(f"ListItem {item} must have a parent")
                    else:
                        list_group = item.parent.resolve(doc)
                        if not isinstance(list_group, ListGroup):
                            _logger.warning(f"Expected ListGroup, got {type(list_group)}")
                        elif list_group.first_item_is_enumerated(doc) and (
                            params.orig_list_item_marker_mode != OrigListItemMarkerMode.AUTO or not item.marker
                        ):
                            pos = -1
                            for i, child in enumerate(list_group.children):
                                if child.resolve(doc) == item:
                                    pos = i
                                    break
                            md_marker = f"{pos + 1}."
                    pieces.append(md_marker)

                # include original marker (if applicable)
                if item.marker and (
                    params.orig_list_item_marker_mode == OrigListItemMarkerMode.ALWAYS
                    or case_auto
                    or case_already_valid
                ):
                    pieces.append(item.marker)

                pieces.append(text)
                text_part = " ".join(pieces)
            else:
                text_part = self._format_heading(text, item, in_table_cell=in_table_cell)
        elif isinstance(item, CodeItem):
            if params.format_code_blocks:
                # inline items and all hyperlinks: use single backticks
                bt = is_inline_scope or (params.include_hyperlinks and item.hyperlink)
                text_part = f"`{text}`" if bt else f"```\n{text}\n```"
            else:
                text_part = text
            escape_html = False
            escape_underscores = False
        elif isinstance(item, FormulaItem):
            if text:
                text_part = f"${text}$" if is_inline_scope else f"$${text}$$"
            elif item.orig:
                text_part = "<!-- formula-not-decoded -->"
            else:
                text_part = ""
            escape_html = False
            escape_underscores = False
        elif params.wrap_width:
            # although wrapping is not guaranteed if post-processing makes changes
            text_part = textwrap.fill(text, width=params.wrap_width)
        else:
            # Apply GFM hard line breaks: single \n -> "  \n", \n\n preserved.
            text_part = self._md_line_breaks(text)

        if text_part:
            text_res = create_ser_result(text=text_part, span_source=item)
            res_parts.append(text_res)

        if isinstance(item, FloatingItem):
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        text = (" " if is_inline_scope else "\n\n").join([r.text for r in res_parts])
        if processing_pending:
            text = doc_serializer.post_process(
                text=text,
                escape_html=escape_html,
                escape_underscores=escape_underscores,
                formatting=item.formatting,
                hyperlink=item.hyperlink,
            )
        return create_ser_result(text=text, span_source=res_parts)

    def _format_heading(
        self,
        text: str,
        item: Union[TitleItem, SectionHeaderItem],
        in_table_cell: bool = False,
    ) -> str:
        """Format a heading or title item as a Markdown heading string.

        Override this method to customize heading representation in subclasses.

        Args:
            text: The heading text content, already post-processed.
            item: The title or section header item being formatted.
            in_table_cell: When ``True``, returns plain text without ``#``
                markers because headings are not valid inside Markdown tables
                per the Markdown spec.

        Returns:
            The formatted heading string, e.g. ``"## My heading"`` for a
            level-1 section header, or plain ``text`` when inside a table cell.
        """
        if in_table_cell:
            return text
        num_hashes = 1 if isinstance(item, TitleItem) else item.level + 1
        return f"{num_hashes * '#'} {text}"


class MarkdownMetaSerializer(BaseModel, BaseMetaSerializer):
    """Markdown-specific meta serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's meta."""
        params = MarkdownParams(**kwargs)
        return create_ser_result(
            text="\n\n".join(
                [
                    tmp
                    for key in (list(item.meta.__class__.model_fields) + list(item.meta.get_custom_part()))
                    if (
                        (params.allowed_meta_names is None or key in params.allowed_meta_names)
                        and (key not in params.blocked_meta_names)
                        and (
                            tmp := self._serialize_meta_field(
                                item.meta,
                                key,
                                params.mark_meta,
                                include_picture_classification=params.include_picture_classification,
                            )
                        )
                    )
                ]
                if item.meta
                else []
            ),
            span_source=item if isinstance(item, DocItem) else [],
            # NOTE for now using an empty span source for GroupItems
        )

    def _serialize_meta_field(
        self,
        meta: BaseMeta,
        name: str,
        mark_meta: bool,
        *,
        include_picture_classification: bool = True,
        **kwargs: Any,
    ) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None:
            if isinstance(field_val, SummaryMetaField):
                txt = field_val.text
            elif isinstance(field_val, KeywordsMetaField | TopicsMetaField):
                txt = ", ".join(field_val.values)
            elif isinstance(field_val, DescriptionMetaField):
                txt = field_val.text
            elif isinstance(field_val, LanguageMetaField):
                # A language code is metadata, not prose: a bare "de" paragraph
                # is indistinguishable from content, so it only renders when
                # meta sections are marked ("[Language] de").
                if not mark_meta:
                    return None
                txt = field_val.code.value
            elif isinstance(field_val, PictureClassificationMetaField):
                if not include_picture_classification:
                    return None
                txt = self._humanize_text(field_val.get_main_prediction().class_name)
            elif isinstance(field_val, MoleculeMetaField):
                txt = field_val.smi
            elif isinstance(field_val, TabularChartMetaField):
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(data=field_val.chart_data)
                table_content = temp_table.export_to_markdown(temp_doc).strip()
                if table_content:
                    txt = table_content
                else:
                    return None
            elif tmp := str(field_val or ""):
                txt = tmp
            else:
                return None
            return f"[{self._humanize_text(name, title=True)}] {txt}" if mark_meta else txt
        else:
            return None


class MarkdownAnnotationSerializer(BaseModel, BaseAnnotationSerializer):
    """Markdown-specific annotation serializer."""

    @override
    def serialize(
        self,
        *,
        item: DocItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's annotations."""
        params = MarkdownParams(**kwargs)

        res_parts: list[SerializationResult] = []
        for ann in item.get_annotations():
            if isinstance(ann, PictureClassificationData) and not params.include_picture_classification:
                continue
            if isinstance(
                ann,
                PictureClassificationData | DescriptionAnnotation | PictureMoleculeData,
            ):
                if ann_text := _get_annotation_text(ann):
                    ann_res = create_ser_result(
                        text=(
                            (f'<!--<annotation kind="{ann.kind}">-->{ann_text}<!--<annotation/>-->')
                            if params.mark_annotations
                            else ann_text
                        ),
                        span_source=item,
                    )
                    res_parts.append(ann_res)
        return create_ser_result(
            text="\n\n".join([r.text for r in res_parts if r.text]),
            span_source=item,
        )


class MarkdownTableSerializer(BaseTableSerializer):
    """Markdown-specific table item serializer."""

    _SEPARATOR_ROW_RE: re.Pattern = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")
    """Matches a Markdown table separator row, e.g. ``| - | :---: | --: |``."""

    @override
    def get_header_and_body_lines(
        self,
        *,
        table_text: str,
        **kwargs: Any,
    ) -> tuple[list[str], list[str]]:
        """Split a serialized Markdown table into header and body lines.

        Locates the separator row (``| - | - |``) to identify the boundary
        between preamble, header, and body.  Any content before the header row
        — including captions that themselves start with ``|`` — is treated as
        preamble and excluded from the returned header lines.
        Returns ``([], all_lines)`` when no separator row can be found or the
        separator is on the first line (no header row above it).

        Args:
            table_text: A serialized Markdown table, possibly preceded by a
                caption or blank lines.

        Returns:
            A tuple ``(header_lines, body_lines)`` where ``header_lines`` holds
            the header row and its separator row, and ``body_lines`` holds the
            remaining data rows.
        """
        all_lines = table_text.splitlines(True)
        sep_idx = next((i for i, l in enumerate(all_lines) if self._SEPARATOR_ROW_RE.match(l.rstrip("\n"))), None)
        if sep_idx is None or sep_idx == 0:
            return [], all_lines
        header_lines = all_lines[sep_idx - 1 : sep_idx + 1]
        body_lines = all_lines[sep_idx + 1 :]
        return header_lines, body_lines

    @staticmethod
    def _compact_table(table_text: str) -> str:
        """Remove padding from a markdown table.

        Args:
            table_text: Padded markdown table string

        Returns:
            Compact markdown table string
        """
        lines = table_text.split("\n")
        compact_lines = []

        for i, line in enumerate(lines):
            if not line:
                continue

            parts = line.split("|")[1:-1]

            # For separator line (second line), preserve alignment marks
            if i == 1:
                compact_parts = []
                for part in parts:
                    p = part.strip()
                    if p.startswith(":") and p.endswith(":"):
                        compact_parts.append(":-:")
                    elif p.startswith(":"):
                        compact_parts.append(":-")
                    elif p.endswith(":"):
                        compact_parts.append("-:")
                    else:
                        compact_parts.append("-")
            else:
                compact_parts = [part.strip() for part in parts]

            compact_lines.append("| " + " | ".join(compact_parts) + " |")

        return "\n".join(compact_lines)

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        if kwargs.get("_nested_in_table"):
            visited: set[str] = kwargs.get("visited") or set()
            _mark_subtree_visited(item, doc, visited)
            return create_ser_result(
                text=_collect_subtree_text(item, doc),
                span_source=item,
            )

        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if _should_use_legacy_annotations(params=params, item=item):
                ann_res = doc_serializer.serialize_annotations(
                    item=item,
                    **kwargs,
                )
                if ann_res.text:
                    res_parts.append(ann_res)

            rows = []
            for row in item.data.grid:
                rendered_row = []
                for col in row:
                    if isinstance(col, RichTableCell):
                        ref_item = col.ref.resolve(doc=doc)
                        inner_kwargs = {**kwargs, "_nested_in_table": True, "in_table_cell": True}
                        cell_text = doc_serializer.serialize(
                            item=ref_item,
                            **inner_kwargs,
                        ).text
                    else:
                        cell_text = col.text or ""
                    # Newlines and pipes must be escaped in every cell so the
                    # markdown table stays valid.
                    rendered_row.append(cell_text.replace("\n", " ").replace("|", "&#124;"))
                rows.append(rendered_row)
            if len(rows) > 0:
                # Always disable numparse to prevent silent precision loss in numeric values
                # Use tabulate's _column_type to detect numeric columns for right-alignment
                colalign = []
                if len(rows) > 1:  # Need at least header + 1 data row
                    num_cols = len(rows[0])
                    for col_idx in range(num_cols):
                        col_values = [row[col_idx] if col_idx < len(row) else "" for row in rows[1:]]
                        col_type = _column_type(col_values)
                        colalign.append("right" if col_type in (int, float) else "left")
                table_text = tabulate(
                    rows[1:],
                    headers=rows[0],
                    tablefmt="github",
                    disable_numparse=True,
                    colalign=tuple(colalign) if colalign else None,
                )

                if params.compact_tables:
                    table_text = self._compact_table(table_text)
            else:
                table_text = ""
            if table_text:
                res_parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in res_parts])

        return create_ser_result(text=text_res, span_source=res_parts)


class MarkdownPictureSerializer(BasePictureSerializer):
    """Markdown-specific picture item serializer."""

    _URI_KEEP_CHARS: Final[str] = "/%:@+,;=~$!&'*"
    """Characters that survive percent-encoding in a Markdown link destination.

    Includes the RFC 3986 reserved characters that carry meaning in a URI, plus
    `%` so that an already-encoded destination is not encoded a second time.
    Whitespace and parentheses are deliberately absent: they would end a Markdown
    inline link.
    """

    _WINDOWS_DRIVE_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]:/")
    """Matches the drive prefix of an absolute Windows path, e.g. `C:/`."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)

        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if _should_use_legacy_annotations(params=params, item=item):
                ann_res = doc_serializer.serialize_annotations(
                    item=item,
                    **kwargs,
                )
                if ann_res.text:
                    res_parts.append(ann_res)

            img_res = self._serialize_image_part(
                item=item,
                doc=doc,
                image_mode=params.image_mode,
                image_placeholder=params.image_placeholder,
            )
            if img_res.text:
                res_parts.append(img_res)

        if params.enable_chart_tables and _should_use_legacy_annotations(
            params=params,
            item=item,
            kind=PictureTabularChartData.model_fields["kind"].default,
        ):
            # Check if picture has attached PictureTabularChartData
            tabular_chart_annotations = [ann for ann in item.annotations if isinstance(ann, PictureTabularChartData)]
            if len(tabular_chart_annotations) > 0:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(data=tabular_chart_annotations[0].chart_data)
                md_table_content = temp_table.export_to_markdown(temp_doc)
                if len(md_table_content) > 0:
                    res_parts.append(create_ser_result(text=md_table_content, span_source=item))
        text_res = "\n\n".join([r.text for r in res_parts if r.text])

        return create_ser_result(text=text_res, span_source=res_parts)

    def _serialize_image_part(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        image_placeholder: str,
        **kwargs: Any,
    ) -> SerializationResult:
        error_response = (
            "<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->"
        )
        if image_mode == ImageRefMode.PLACEHOLDER:
            text_res = image_placeholder
        elif image_mode == ImageRefMode.EMBEDDED:
            # short-cut: we already have the image in base64
            if (
                isinstance(item.image, ImageRef)
                and isinstance(item.image.uri, AnyUrl)
                and item.image.uri.scheme == "data"
            ):
                text = f"![Image]({item.image.uri})"
                text_res = text
            else:
                # get the item.image._pil or crop it out of the page-image
                img = item.get_image(doc=doc)

                if img is not None:
                    imgb64 = item._image_to_base64(img)
                    text = f"![Image](data:image/png;base64,{imgb64})"

                    text_res = text
                else:
                    text_res = error_response
        elif image_mode == ImageRefMode.REFERENCED:
            if not isinstance(item.image, ImageRef) or (
                isinstance(item.image.uri, AnyUrl) and item.image.uri.scheme == "data"
            ):
                text_res = image_placeholder
            else:
                text_res = f"![Image]({self._escape_uri_path(item.image.uri)})"
        else:
            text_res = image_placeholder

        return create_ser_result(text=text_res, span_source=item)

    @staticmethod
    def _escape_uri_path(value: Union[AnyUrl, PurePath]) -> str:
        """Encode a URL or filesystem path as a Markdown link destination.

        Handles URLs of any scheme (https/s3/ftp/...) as well as POSIX and Windows
        paths, keeps relative paths relative, and never double-encodes. A Windows path
        is recognized by either flavour, so a document authored on Windows still
        resolves when it is exported on POSIX.

        The only destination this gives a ``file://`` scheme to is an absolute Windows
        path, where it is the sole spelling a renderer cannot misread as a URL scheme
        (``C:``) or as an authority (``//server``). A URL that already carries the
        scheme is passed through, since dropping it would turn an absolute filesystem
        reference into a root-relative URL.

        Known limitation: a backslash in a `PosixPath` string is ambiguous.
        It may be a Windows separator surviving a JSON round-trip (correct to
        convert) or a literal filename character (where converting it to `/`
        would split one component into two). The two cases are indistinguishable
        from `str()`. In practice this is not a concern because `ImageRef.uri`
        is always populated from native filesystem operations, so a `PosixPath`
        can only carry a literal backslash if the caller explicitly constructed one.

        Args:
            value: The URL or path to encode.

        Returns:
            A percent-encoded Markdown link destination.
        """

        keep = MarkdownPictureSerializer._URI_KEEP_CHARS
        # A backslash is both the Windows separator and a Markdown escape character, and
        # is read as a separator whatever flavour the path arrives in: a document authored
        # on Windows keeps its backslashes once it is re-read on POSIX, where the flavour
        # can no longer tell. A URL is unaffected, as pydantic normalizes backslashes away
        # when parsing.
        s = str(value).replace("\\", "/")

        if s.startswith("//"):  # In case of a fileshare (//someserver/somefolder)
            host, _, tail = s.lstrip("/").partition("/")  # get the end of the path
            # file://<host>/<path>, with <host> being a possibly empty string.
            return urlunsplit(("file", host, quote(f"/{tail}", safe=keep), "", ""))
        if MarkdownPictureSerializer._WINDOWS_DRIVE_RE.match(s):  # In case of a Windows filename with drive letter
            # file://<full_path_with_filename>
            return urlunsplit(("file", "", quote(f"/{s}", safe=keep), "", ""))

        # A URL keeps its scheme, authority and delimiters; only its components are
        # encoded. A single-character scheme cannot be real, so it is read as a path.
        parts = urlsplit(s)
        if len(parts.scheme) > 1:
            return urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    quote(parts.path, safe=keep),
                    quote(parts.query, safe=keep + "="),
                    quote(parts.fragment, safe=keep),
                )
            )

        # A relative or root-relative local path.
        return quote(s, safe=keep)


class MarkdownKeyValueSerializer(BaseKeyValueSerializer):
    """Markdown-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        if item.self_ref not in doc_serializer.get_excluded_refs():
            return create_ser_result(
                text="<!-- missing-key-value-item -->",
                span_source=item,
            )
        else:
            return create_ser_result()


class MarkdownFormSerializer(BaseFormSerializer):
    """Markdown-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        if item.self_ref not in doc_serializer.get_excluded_refs():
            return create_ser_result(
                text="<!-- missing-form-item -->",
                span_source=item,
            )
        else:
            return create_ser_result()


class MarkdownListSerializer(BaseModel, BaseListSerializer):
    """Markdown-specific list serializer."""

    @override
    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)
        my_visited = visited if visited is not None else set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        sep = "\n"
        my_parts: list[SerializationResult] = []
        for p in parts:
            if (
                my_parts
                and p.text
                and p.spans
                and p.spans[0].item.parent
                and isinstance(p.spans[0].item.parent.resolve(doc), InlineGroup)
            ):
                my_parts[-1].text = f"{my_parts[-1].text}{p.text}"  # append to last
                my_parts[-1].spans.extend(p.spans)
            else:
                my_parts.append(p)

        indent_str = list_level * params.indent * " "
        my_texts = [
            # avoid additional marker on already evaled sublists
            (c.text if c.text and c.text[0] == " " else f"{indent_str}{c.text}")
            for c in my_parts
        ]
        text_res = ""
        for i, text in enumerate(my_texts):
            if i:
                # a page break is a block-level marker, so it gets the document
                # scope separator on both sides instead of the list one
                text_res += (
                    "\n\n"
                    if isinstance(my_parts[i], _PageBreakSerResult) or isinstance(my_parts[i - 1], _PageBreakSerResult)
                    else sep
                )
            text_res += text
        return create_ser_result(text=text_res, span_source=my_parts)


class MarkdownInlineSerializer(BaseInlineSerializer):
    """Markdown-specific inline group serializer."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )
        text_res = " ".join([p.text for p in parts if p.text])
        return create_ser_result(text=text_res, span_source=parts)


class MarkdownFallbackSerializer(BaseFallbackSerializer):
    """Markdown-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        if isinstance(item, GroupItem):
            parts = doc_serializer.get_parts(item=item, **kwargs)
            text_res = "\n\n".join([p.text for p in parts if p.text])
            return create_ser_result(text=text_res, span_source=parts)
        elif isinstance(item, (FieldRegionItem, FieldItem)):
            return create_ser_result()
        else:
            return create_ser_result(
                text="<!-- missing-text -->",
                span_source=item if isinstance(item, DocItem) else [],
            )


class MarkdownDocSerializer(DocSerializer):
    """Markdown-specific document serializer."""

    text_serializer: BaseTextSerializer = MarkdownTextSerializer()
    table_serializer: BaseTableSerializer = MarkdownTableSerializer()
    picture_serializer: BasePictureSerializer = MarkdownPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = MarkdownKeyValueSerializer()
    form_serializer: BaseFormSerializer = MarkdownFormSerializer()
    fallback_serializer: BaseFallbackSerializer = MarkdownFallbackSerializer()

    list_serializer: BaseListSerializer = MarkdownListSerializer()
    inline_serializer: BaseInlineSerializer = MarkdownInlineSerializer()

    meta_serializer: BaseMetaSerializer = MarkdownMetaSerializer()
    annotation_serializer: BaseAnnotationSerializer = MarkdownAnnotationSerializer()

    params: MarkdownParams = MarkdownParams()

    @override
    def serialize_bold(self, text: str, **kwargs: Any):
        """Apply Markdown-specific bold serialization."""
        return f"**{text}**"

    @override
    def serialize_italic(self, text: str, **kwargs: Any):
        """Apply Markdown-specific italic serialization."""
        return f"*{text}*"

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any):
        """Apply Markdown-specific strikethrough serialization."""
        return f"~~{text}~~"

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ):
        """Apply Markdown-specific hyperlink serialization."""
        return f"[{text}]({hyperlink!s})"

    @classmethod
    def _escape_underscores(cls, text: str):
        """Escape underscores but leave them intact in the URL.."""
        # Firstly, identify all the URL patterns.
        url_pattern = r"!\[.*?\]\((.*?)\)"

        parts = []
        last_end = 0

        for match in re.finditer(url_pattern, text):
            # Text to add before the URL (needs to be escaped)
            before_url = text[last_end : match.start()]
            parts.append(re.sub(r"(?<!\\)_", r"\_", before_url))

            # Add the full URL part (do not escape)
            parts.append(match.group(0))
            last_end = match.end()

        # Add the final part of the text (which needs to be escaped)
        if last_end < len(text):
            parts.append(re.sub(r"(?<!\\)_", r"\_", text[last_end:]))

        return "".join(parts)
        # return text.replace("_", r"\_")

    def post_process(
        self,
        text: str,
        *,
        escape_html: bool = True,
        escape_underscores: bool = True,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """Apply some text post-processing steps."""
        res = text
        params = self.params.merge_with_patch(patch=kwargs)
        if escape_underscores and params.escape_underscores:
            res = self._escape_underscores(text)
        if escape_html and params.escape_html:
            res = html.escape(res, quote=False)
        res = super().post_process(
            text=res,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return res

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts."""
        text_res = "\n\n".join([p.text for p in parts if p.text])
        if self.requires_page_break():
            page_sep = self.params.page_break_placeholder or ""
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        return create_ser_result(text=text_res, span_source=parts)

    @override
    def requires_page_break(self) -> bool:
        """Whether to add page breaks."""
        return self.params.page_break_placeholder is not None

    @override
    def serialize(
        self,
        *,
        item: Optional[NodeItem] = None,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a given node."""
        return super().serialize(
            item=item,
            list_level=list_level,
            is_inline_scope=is_inline_scope,
            visited=visited,
            **(dict(delim="\n\n") | kwargs),
        )
