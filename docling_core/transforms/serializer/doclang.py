"""Define classes for DocLang serialization.

Aligned to the DocLang specification version ``_DOCLANG_VERSION``.
"""

import copy
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Annotated, Any, Optional, Union, cast

from defusedxml.ElementTree import fromstring
from defusedxml.minidom import parseString
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.networks import AnyUrl
from typing_extensions import override

from docling_core.transforms.serializer._doclang_utils import (
    _DOCLANG_LABEL_UNDEFINED,
    _DOCLANG_VERSION,
    DOCLANG_DFLT_RESOLUTION,
    DOCLANG_NAMESPACE,
    DocLangAttributeKey,
    DocLangAttributeValue,
    DocLangToken,
    DocLangVocabulary,
    _append_textual_fragment,
    _code_language_label_from_doclang,
    _code_language_label_to_doclang,
    _create_location_tokens_for_bbox,
    _create_location_tokens_for_item,
    _merge_table_data,
    _picture_classification_label_from_doclang,
    _picture_classification_label_to_doclang,
    _provenance_with_charspan,
    _quantize_to_resolution,
    _thread_table_merge_offset,
    _wrap,
    _wrap_field_kv_markup_if_needed,
    _wrap_in_field_item_if_needed,
    _wrap_in_field_region_if_needed,
    _wrap_token,
    _xml_error_context,
)
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
    _PageBreakNode,
    create_ser_result,
)
from docling_core.types.doc import (
    BaseMeta,
    BoundingBox,
    CodeItem,
    ContentLayer,
    DescriptionMetaField,
    DocItem,
    DoclingDocument,
    FloatingItem,
    Formatting,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    ListItem,
    MetaFieldName,
    MoleculeMetaField,
    NodeItem,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
    ProvenanceItem,
    Script,
    SectionHeaderItem,
    Size,
    SummaryMetaField,
    TableCell,
    TableData,
    TableItem,
    TabularChartMetaField,
    TextItem,
)
from docling_core.types.doc.base import CoordOrigin, ImageRefMode
from docling_core.types.doc.document import (
    FieldHeadingItem,
    FieldItem,
    FieldRegionItem,
    FieldValueItem,
    FormulaItem,
    GroupItem,
    RichTableCell,
    TitleItem,
)
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    GroupLabel,
    PictureClassificationLabel,
)
from docling_core.types.doc.utils import get_text_direction

__all__ = [
    "ContentType",
    "DocLangDocSerializer",
    "DocLangParams",
    "DocLangVocabulary",
    "EscapeMode",
    "LabelMode",
    "LayerMode",
    "WrapMode",
]


def _create_page_break_markup(node: _PageBreakNode) -> str:
    """Return the internal page-break placeholder replaced in ``serialize_doc``."""
    return f"#_#_DOCLING_DOC_PAGE_BREAK_{node.prev_page}_{node.next_page}_#_#"


def _suppress_document_page_break(
    doc_serializer: BaseDocSerializer,
    *,
    prev_page: int,
    next_page: int,
) -> None:
    """Skip a duplicate document-level page break already emitted by list/table threading."""
    if isinstance(doc_serializer, DocLangDocSerializer):
        doc_serializer._suppressed_page_breaks.add((prev_page, next_page))


def _allocate_thread_id(doc_serializer: BaseDocSerializer, node: NodeItem) -> str:
    """Allocate a document-scoped positive ``thread_id`` in reading order."""
    if isinstance(doc_serializer, DocLangDocSerializer):
        return doc_serializer.allocate_thread_id(node)
    raise TypeError("DocLang threading requires DocLangDocSerializer")


def _primary_page_no(node: NodeItem) -> Optional[int]:
    """Return the primary page number for a document item, if known."""
    if isinstance(node, DocItem) and node.prov:
        return node.prov[0].page_no
    return None


class EscapeMode(str, Enum):
    """XML escape mode for DocLang output."""

    ALWAYS = "always"  # wrap all text in CDATA
    AUTO = "auto"  # wrap text in CDATA only if it contains special characters


class WrapMode(str, Enum):
    """Explicit content-wrapper mode for DocLang output."""

    ALWAYS = "always"  # wrap all text in an explicit wrapper element
    AUTO = "auto"  # wrap text with leading/trailing whitespace or newlines


class LayerMode(str, Enum):
    """Content-layer element emission mode for DocLang output."""

    ALWAYS = "always"  # always include the layer element
    AUTO = "auto"  # include layer only when it differs from the default


class LabelMode(str, Enum):
    """Element-head label emission mode for DocLang output."""

    ALWAYS = "always"  # always emit label, using ``undefined`` when absent
    AUTO = "auto"  # emit label only when present and not ``undefined``


class ContentType(str, Enum):
    """Content type for DocLang output."""

    REF_CAPTION = "ref_caption"
    REF_FOOTNOTE = "ref_footnote"

    TEXT_CODE = "text_code"
    TEXT_FORMULA = "text_formula"
    TEXT_OTHER = "text_other"
    TABLE = "table"
    CHART = "chart"
    TABLE_CELL = "table_cell"
    PICTURE = "picture"
    CHEMISTRY = "chemistry"


_DEFAULT_CONTENT_TYPES: set[ContentType] = set(ContentType)


def _advanced_field(*, detail: str = "") -> Any:
    """Build a Pydantic ``Field`` for advanced ``DocLangParams`` members."""
    description = "Advanced parameter, meant for internal use."
    if detail:
        description = f"{description} {detail}"
    return Field(description=description)


class DocLangParams(CommonParams):
    """DocLang-specific serialization parameters independent of DocLang."""

    # Override parent's layers to default to all ContentLayers
    layers: set[ContentLayer] = set(ContentLayer)

    # Advanced parameters (meant for internal use):

    # Geometry & content controls (aligned with DocLang defaults)
    xsize: Annotated[int, _advanced_field()] = DOCLANG_DFLT_RESOLUTION
    ysize: Annotated[int, _advanced_field()] = DOCLANG_DFLT_RESOLUTION
    add_location: Annotated[bool, _advanced_field()] = True
    add_table_cell_location: Annotated[bool, _advanced_field()] = False
    add_referenced_caption: Annotated[bool, _advanced_field()] = True
    add_referenced_footnote: Annotated[bool, _advanced_field()] = True
    add_page_break: Annotated[bool, _advanced_field()] = True
    add_content: Annotated[bool, _advanced_field()] = True
    content_types: Annotated[
        set[ContentType],
        _advanced_field(detail="Types of content to serialize (only relevant if add_content is True)."),
    ] = _DEFAULT_CONTENT_TYPES
    layer_mode: Annotated[LayerMode, _advanced_field()] = LayerMode.AUTO
    # DocLang formatting
    pretty_indentation: Annotated[
        Optional[str],
        _advanced_field(detail='None means minimized serialization, "" means no indentation.'),
    ] = 2 * " "
    preserve_empty_non_selfclosing: Annotated[bool, _advanced_field()] = True
    suppress_empty_elements: Annotated[
        bool,
        _advanced_field(
            detail=(
                "When True, text items that produce no content (no text, no location) are "
                "completely omitted rather than emitting an empty open/close tag pair."
            ),
        ),
    ] = False
    escape_mode: Annotated[
        EscapeMode,
        _advanced_field(detail="XML compliance: escape special characters in text content."),
    ] = EscapeMode.AUTO
    content_wrapping_mode: Annotated[WrapMode, _advanced_field()] = WrapMode.AUTO
    image_mode: Annotated[ImageRefMode, _advanced_field()] = ImageRefMode.PLACEHOLDER
    include_namespace: Annotated[bool, _advanced_field()] = False
    include_version: Annotated[bool, _advanced_field()] = False
    use_virtual_text: Annotated[
        bool,
        _advanced_field(detail="When True, the <text> wrapper is omitted whenever allowed."),
    ] = True
    label_mode: Annotated[LabelMode, _advanced_field()] = LabelMode.AUTO
    interpret_code_unknown_as_other: Annotated[
        bool,
        _advanced_field(
            detail="When False, CodeLanguageLabel.UNKNOWN maps to undefined; when True, to other.",
        ),
    ] = False


def _create_layer_token(
    *,
    item: DocItem,
    params: DocLangParams,
) -> str:
    """Create `<layer value="..."/>` in element head."""
    if params.layer_mode == LayerMode.ALWAYS or (
        params.layer_mode == LayerMode.AUTO and item.content_layer != ContentLayer.BODY
    ):
        return DocLangVocabulary._create_selfclosing_token(
            token=DocLangToken.LAYER,
            attrs={DocLangAttributeKey.VALUE: item.content_layer.value},
        )
    return ""


def _create_label_token(*, value: str) -> str:
    """Emit `<label value="..."/>` for element head (e.g. code language)."""
    safe = value.replace("&", "&amp;").replace('"', "&quot;")
    return DocLangVocabulary._create_selfclosing_token(
        token=DocLangToken.LABEL,
        attrs={DocLangAttributeKey.VALUE: safe},
    )


def _create_src_token(*, uri: str) -> str:
    """Emit `<src uri="..."/>` for picture body (v0.5)."""
    safe = uri.replace("&", "&amp;").replace('"', "&quot;")
    return DocLangVocabulary._create_selfclosing_token(
        token=DocLangToken.SRC,
        attrs={DocLangAttributeKey.URI: safe},
    )


def _create_href_token(*, uri: str) -> str:
    """Emit `<href uri="..."/>` in element head."""
    safe = uri.replace("&", "&amp;").replace('"', "&quot;")
    return DocLangVocabulary._create_selfclosing_token(
        token=DocLangToken.HREF,
        attrs={DocLangAttributeKey.URI: safe},
    )


def _text_item_hyperlink_uri(item: DocItem) -> Optional[str]:
    if isinstance(item, TextItem) and item.hyperlink is not None:
        return str(item.hyperlink)
    return None


def _element_head_prefix(
    *,
    item: DocItem,
    doc: DoclingDocument,
    params: DocLangParams,
    label_value: Optional[str] = None,
    caption_text: Optional[str] = None,
    custom_text: Optional[str] = None,
    include_href: bool = True,
    thread_id: Optional[str] = None,
) -> str:
    """Emit element-head property elements in XSD order (label → thread → href → layer → location → caption → custom)."""
    parts: list[str] = []
    if label_value:
        parts.append(_create_label_token(value=label_value))
    if thread_id:
        parts.append(DocLangVocabulary._create_threading_token(thread_id=thread_id))
    if include_href and (href_uri := _text_item_hyperlink_uri(item)):
        parts.append(_create_href_token(uri=href_uri))
    if layer_token := _create_layer_token(item=item, params=params):
        parts.append(layer_token)
    if params.add_location:
        if loc := _create_location_tokens_for_item(item=item, doc=doc, xres=params.xsize, yres=params.ysize):
            parts.append(loc)
    if caption_text:
        parts.append(caption_text)
    if custom_text:
        parts.append(custom_text)
    return "".join(parts)


def _serialize_floating_caption_head(
    *,
    item: FloatingItem,
    doc_serializer: BaseDocSerializer,
    doc: DoclingDocument,
    params: DocLangParams,
    **kwargs: Any,
) -> str:
    """Serialize referenced caption(s) for inclusion in the host element head."""
    if not params.add_referenced_caption or not item.captions:
        return ""
    cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
    return cap_res.text or ""


def _element_label_for_serialization(
    *,
    raw_label: Optional[str],
    params: DocLangParams,
) -> Optional[str]:
    """Resolve element-head ``<label>`` emission per ``params.label_mode``."""
    if params.label_mode == LabelMode.ALWAYS:
        return raw_label if raw_label is not None else _DOCLANG_LABEL_UNDEFINED
    # AUTO: emit only when a label is present and not ``undefined``.
    if raw_label is None or raw_label == _DOCLANG_LABEL_UNDEFINED:
        return None
    return raw_label


def _picture_classification_label_value(item: PictureItem) -> Optional[str]:
    """Picture type label for element head (raw ``class_name`` from the main prediction)."""
    if item.meta and item.meta.classification:
        class_name = item.meta.classification.get_main_prediction().class_name
        return _picture_classification_label_to_doclang(class_name)
    return None


def _serialize_item_custom_head(
    *,
    item: NodeItem,
    doc_serializer: BaseDocSerializer,
    params: DocLangParams,
    **kwargs: Any,
) -> str:
    """Serialize item meta as ``<custom>`` for element head (v0.5)."""
    if not isinstance(item, DocItem) or not item.meta:
        return ""
    meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
    return meta_res.text or ""


def _get_delim(*, params: DocLangParams) -> str:
    """Return record delimiter based on ``pretty_indentation``."""
    return "" if params.pretty_indentation is None else "\n"


def _escape_text(text: str, params: DocLangParams) -> str:
    do_wrap = params.content_wrapping_mode == WrapMode.ALWAYS or (
        params.content_wrapping_mode == WrapMode.AUTO and (text != text.strip() or "\n" in text)
    )
    if params.escape_mode == EscapeMode.ALWAYS or (
        params.escape_mode == EscapeMode.AUTO and any(c in text for c in ['"', "'", "&", "<", ">"])
    ):
        text = f"<![CDATA[{text}]]>"
    if do_wrap:
        # text = f'<{el_str} xml:space="preserve">{text}</{el_str}>'
        text = _wrap(text=text, wrap_tag=DocLangToken.CONTENT.value)
    return text


def _list_item_segment_sibling(child: NodeItem) -> bool:
    """True when ``child`` is serialized as a sibling in the same ``<ldiv>`` segment."""
    return isinstance(child, ListGroup | PictureItem)


def _list_item_has_segment_siblings(*, item: ListItem, doc: DoclingDocument) -> bool:
    """True when markup besides the list item text is emitted in the same ldiv segment."""
    for child_ref in item.children:
        if _list_item_segment_sibling(child_ref.resolve(doc)):
            return True
    parent = item.parent.resolve(doc) if item.parent else None
    if isinstance(parent, ListGroup):
        seen_self = False
        for child_ref in parent.children:
            child = child_ref.resolve(doc)
            if child is item:
                seen_self = True
                continue
            if seen_self and isinstance(child, ListGroup):
                return True
    return False


class DocLangListSerializer(BaseModel, BaseListSerializer):
    """DocLang-specific list serializer."""

    indent: int = 4

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
        """Serialize a ``ListGroup`` into DocLang markup.

        This emits list containers (``<ordered_list>``/``<unordered_list>``) and
        serializes children explicitly. Nested ``ListGroup`` items are emitted as
        siblings, and individual list items are not wrapped here. The text
        serializer is responsible for wrapping list item content (as
        ``<ldiv>``), so this serializer remains agnostic of item types.

        Args:
            item: The list group to serialize.
            doc_serializer: The document-level serializer to delegate nested items.
            doc: The document that provides item resolution.
            list_level: Current nesting depth (0-based).
            is_inline_scope: Whether serialization happens in an inline context.
            visited: Set of already visited item refs to avoid cycles.
            **kwargs: Additional serializer parameters forwarded to ``DocLangParams``.

        Returns:
            A ``SerializationResult`` containing serialized text and metadata.
        """
        my_visited = visited if visited is not None else set()
        params = DocLangParams(**kwargs)

        # Build list children explicitly. Requirements:
        # 1) <list ordered="true|false"></list> can be children of lists.
        # 2) Do NOT wrap nested lists into <ldiv>, even if they are
        #    children of a ListItem in the logical structure.
        # 3) Still ensure structural wrappers are preserved even when
        #    content is suppressed (e.g., add_content=False).
        item_results: list[SerializationResult] = []
        child_segments: list[tuple[str, Optional[int]]] = []

        excluded = doc_serializer.get_excluded_refs(**kwargs)
        for child_ref in item.children:
            child = child_ref.resolve(doc)

            # If a nested list group is present directly under this list group,
            # emit it as a sibling (no <list_item> wrapper).
            if isinstance(child, ListGroup):
                if child.self_ref in my_visited or child.self_ref in excluded:
                    continue
                my_visited.add(child.self_ref)
                sub_res = doc_serializer.serialize(
                    item=child,
                    list_level=list_level + 1,
                    is_inline_scope=is_inline_scope,
                    visited=my_visited,
                    **kwargs,
                )
                if sub_res.text:
                    child_segments.append((sub_res.text, None))
                item_results.append(sub_res)
                continue

            # Normal case: ListItem under ListGroup
            if not isinstance(child, ListItem):
                continue
            if child.self_ref in my_visited or child.self_ref in excluded:
                continue

            my_visited.add(child.self_ref)

            # Serialize the list item content; wrapping is handled by the text
            # serializer (as <ldiv>), not here.
            child_res = doc_serializer.serialize(
                item=child,
                list_level=list_level + 1,
                is_inline_scope=is_inline_scope,
                visited=my_visited,
                **kwargs,
            )
            item_results.append(child_res)
            if child_res.text:
                child_segments.append((child_res.text, _primary_page_no(child)))

            # After the <ldiv>, append nested lists and pictures (children of this
            # ListItem) as siblings at the same level (not wrapped in <ldiv>).
            for subref in child.children:
                sub = subref.resolve(doc)
                if not _list_item_segment_sibling(sub):
                    continue
                if sub.self_ref in my_visited or sub.self_ref in excluded:
                    continue
                my_visited.add(sub.self_ref)
                sub_res = doc_serializer.serialize(
                    item=sub,
                    list_level=list_level + 1,
                    is_inline_scope=is_inline_scope,
                    visited=my_visited,
                    **kwargs,
                )
                if sub_res.text:
                    child_segments.append((sub_res.text, _primary_page_no(sub) if isinstance(sub, DocItem) else None))
                item_results.append(sub_res)

        delim = _get_delim(params=params)
        if not child_segments:
            return create_ser_result(text="", span_source=item_results)

        ordered = item.first_item_is_enumerated(doc)
        list_close = f"</{DocLangToken.LIST.value}>"
        spans_pages = any(
            child_segments[i][1] is not None
            and child_segments[i + 1][1] is not None
            and child_segments[i][1] != child_segments[i + 1][1]
            for i in range(len(child_segments) - 1)
        )

        if not spans_pages:
            child_texts = [text for text, _ in child_segments if text]
            text_res = delim.join(child_texts)
            text_res = f"{text_res}{delim}"
            open_token = (
                DocLangVocabulary._create_list_token(ordered=True)
                if ordered
                else DocLangVocabulary._create_list_token(ordered=False)
            )
            text_res = _wrap_token(text=text_res, open_token=open_token)
            return create_ser_result(text=text_res, span_source=item_results)

        thread_id = _allocate_thread_id(doc_serializer, item)
        out_parts: list[str] = []
        current_block: list[str] = []
        current_page: Optional[int] = None
        for text, page_no in child_segments:
            if current_block and page_no is not None and current_page is not None and page_no != current_page:
                list_open = DocLangVocabulary._create_list_token(
                    ordered=ordered
                ) + DocLangVocabulary._create_threading_token(thread_id=thread_id)
                block_text = delim.join(current_block)
                out_parts.append(f"{list_open}{block_text}{delim}{list_close}")
                pb = _PageBreakNode(
                    self_ref=f"#/pb/{len(out_parts)}",
                    prev_page=current_page,
                    next_page=page_no,
                )
                _suppress_document_page_break(
                    doc_serializer,
                    prev_page=current_page,
                    next_page=page_no,
                )
                out_parts.append(_create_page_break_markup(pb))
                current_block = []
            if text:
                current_block.append(text)
            if page_no is not None:
                current_page = page_no

        if current_block:
            list_open = DocLangVocabulary._create_list_token(
                ordered=ordered
            ) + DocLangVocabulary._create_threading_token(thread_id=thread_id)
            block_text = delim.join(current_block)
            out_parts.append(f"{list_open}{block_text}{delim}{list_close}")

        return create_ser_result(text="".join(out_parts), span_source=item_results)


class DocLangTextSerializer(BaseModel, BaseTextSerializer):
    """DocLang-specific text item serializer using `<location>` tokens."""

    @override
    def serialize(
        self,
        *,
        item: "TextItem",
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a text item to DocLang format.

        Handles multi-provenance items by splitting them into per-provenance items,
        serializing each separately, and merging the results.

        Args:
            item: The text item to serialize.
            doc_serializer: The document serializer instance.
            doc: The DoclingDocument being serialized.
            visited: Set of already visited item references.
            **kwargs: Additional keyword arguments.

        Returns:
            SerializationResult containing the serialized text and span mappings.
        """
        if len(item.prov) > 1 and not isinstance(item, ListItem):
            # Split multi-provenance items into per-provenance fragments linked by
            # a shared thread_id; insert page breaks when page_no changes.
            # List items are not split here; cross-page lists are handled at list-group level.
            thread_id = _allocate_thread_id(doc_serializer, item)
            res: list[SerializationResult] = []
            for idp, prov_ in enumerate(item.prov):
                item_ = copy.deepcopy(item)
                item_.prov = [prov_]
                item_.text = item.orig[prov_.charspan[0] : prov_.charspan[1]]  # it must be `orig`, not `text` here!
                item_.orig = item.orig[prov_.charspan[0] : prov_.charspan[1]]

                item_.prov[0].charspan = (0, len(item_.orig))

                # marker field should be cleared on subsequent split parts
                if idp > 0 and isinstance(item_, ListItem):
                    item_.marker = ""

                tres: SerializationResult = self._serialize_single_item(
                    item=item_,
                    doc_serializer=doc_serializer,
                    doc=doc,
                    visited=visited,
                    is_inline_scope=is_inline_scope,
                    thread_id=thread_id,
                    **kwargs,
                )
                res.append(tres)

            out_parts: list[str] = []
            for idp, tres in enumerate(res):
                if idp > 0 and item.prov[idp - 1].page_no != item.prov[idp].page_no:
                    pb = _PageBreakNode(
                        self_ref=f"#/pb/{idp}",
                        prev_page=item.prov[idp - 1].page_no,
                        next_page=item.prov[idp].page_no,
                    )
                    _suppress_document_page_break(
                        doc_serializer,
                        prev_page=item.prov[idp - 1].page_no,
                        next_page=item.prov[idp].page_no,
                    )
                    out_parts.append(_create_page_break_markup(pb))
                out_parts.append(tres.text)
            return create_ser_result(text="".join(out_parts), span_source=res)

        else:
            return self._serialize_single_item(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                visited=visited,
                is_inline_scope=is_inline_scope,
                **kwargs,
            )

    def _should_skip_location_for_list_item(self, *, item: ListItem, doc: DoclingDocument) -> bool:
        """Check if location tokens should be skipped for a ListItem.

        Returns True if the ListItem has empty text, provenance, and its first
        child is an InlineGroup (which will handle location tokens itself).
        """
        if not item.text and item.prov and item.children:
            first_child_ref = item.children[0]
            first_child_item = first_child_ref.resolve(doc)
            return isinstance(first_child_item, InlineGroup)
        return False

    def _list_item_has_segment_siblings(self, *, item: ListItem, doc: DoclingDocument) -> bool:
        """True when markup besides the list item text is emitted in the same ldiv segment."""
        return _list_item_has_segment_siblings(item=item, doc=doc)

    def _determine_list_item_wrapper(
        self, *, item: ListItem, doc: DoclingDocument, use_virtual_text: bool = True
    ) -> tuple[Optional[str], Optional[DocLangToken]]:
        """Determine the wrapper token for a ListItem.

        Args:
            item: The ListItem to determine wrapper for.
            doc: The document containing the item.
            use_virtual_text: If True, omit ``<text>`` when the ldiv segment contains
                only that text (DocLang v0.4 virtual text mode).

        Returns:
            Tuple of (wrap_open_token, tok) where wrap_open_token is the opening tag
            string or None, and tok is the DocLangToken or None.
        """
        if item.text:
            if use_virtual_text and not self._list_item_has_segment_siblings(item=item, doc=doc):
                return None, None
            tok = DocLangToken.TEXT
            return f"<{tok.value}>", tok
        elif not item.text and item.prov and item.children:
            # Check if first child is InlineGroup (rich text case)
            first_child_ref = item.children[0]
            first_child_item = first_child_ref.resolve(doc)
            if isinstance(first_child_item, InlineGroup):
                # First child is InlineGroup: don't wrap, let InlineGroup handle it
                # InlineSerializer will use parent ListItem's provenance for location tokens
                return None, None
            else:
                # Other children with bbox: wrap in <group>
                tok = DocLangToken.GROUP
                return f"<{tok.value}>", tok
        else:
            return None, None

    def _serialize_single_item(  # noqa: C901
        self,
        *,
        item: "TextItem",
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``TextItem`` into DocLang markup.

        Depending on parameters, emits meta blocks, location tokens, and the
        item's textual content (prefixing code language for ``CodeItem``). For
        floating items, captions may be appended. The result can be wrapped in a
        tag derived from the item's label when applicable.

        Args:
            item: The text-like item to serialize.
            doc_serializer: The document-level serializer for delegating nested items.
            doc: The document used to resolve references and children.
            visited: Set of already visited item refs to avoid cycles.
            **kwargs: Additional serializer parameters forwarded to ``DocLangParams``.

        Returns:
            A ``SerializationResult`` with the serialized text and span source.
        """
        my_visited = visited if visited is not None else set()
        params = DocLangParams(**kwargs)

        # Determine wrapper open-token for this item using DocLang vocabulary.
        # - TitleItem: use <heading level="1"> ... </heading>.
        # - SectionHeaderItem: use <heading level="N+1"> ... </heading> where N is SectionHeaderItem.level.
        # - Other text-like items: map the label to an DocLangToken; for
        #   list items, this maps to <ldiv> and keeps the text serializer
        #   free of type-based special casing.
        wrap_open_token: Optional[str]
        tok: DocLangToken | None = None
        if isinstance(item, TitleItem):
            wrap_open_token = DocLangVocabulary._create_heading_token(level=1)
        elif isinstance(item, SectionHeaderItem):
            wrap_open_token = DocLangVocabulary._create_heading_token(level=item.level + 1)
        elif isinstance(item, ListItem):
            wrap_open_token, tok = self._determine_list_item_wrapper(
                item=item, doc=doc, use_virtual_text=params.use_virtual_text
            )
        elif isinstance(item, CodeItem):
            tok = DocLangToken.CODE
            wrap_open_token = f"<{tok.value}>"
        elif isinstance(item, TextItem) and item.label in [
            DocItemLabel.CHECKBOX_SELECTED,
            DocItemLabel.CHECKBOX_UNSELECTED,
        ]:
            if item.parent and isinstance((parent_item := item.parent.resolve(doc)), TextItem) and not parent_item.text:
                # skip re-wrapping if already in a text item
                wrap_open_token = None
            else:
                tok = DocLangToken.TEXT
                wrap_open_token = f"<{tok.value}>"
        elif isinstance(item, TextItem) and item.label == DocItemLabel.CAPTION:
            # v0.5: <caption> is only valid in a host element head, not top-level.
            tok = DocLangToken.TEXT
            wrap_open_token = f"<{tok.value}>"
        elif isinstance(item, TextItem) and (
            tok := {
                DocItemLabel.FIELD_KEY: DocLangToken.FIELD_KEY,
                DocItemLabel.FIELD_VALUE: DocLangToken.FIELD_VALUE,
                DocItemLabel.FIELD_HEADING: DocLangToken.FIELD_HEADING,
                DocItemLabel.FIELD_HINT: DocLangToken.FIELD_HINT,
                DocItemLabel.MARKER: DocLangToken.MARKER,
            }.get(item.label)
        ):
            wrap_open_token = f"<{tok.value}>"
            if isinstance(item, FieldValueItem) and item.kind != "read_only":
                wrap_open_token = f'<{tok.value} class="{item.kind}">'
            elif isinstance(item, FieldHeadingItem):
                wrap_open_token = DocLangVocabulary._create_field_heading_token(level=item.level)
        elif isinstance(item, TextItem) and (
            item.label
            in [  # FIXME: Catch all ...
                DocItemLabel.EMPTY_VALUE,  # FIXME: this might need to become a FormItem with only a value key!
                DocItemLabel.HANDWRITTEN_TEXT,
                DocItemLabel.PARAGRAPH,
                DocItemLabel.REFERENCE,
                DocItemLabel.GRADING_SCALE,
            ]
        ):
            tok = DocLangToken.TEXT
            wrap_open_token = f"<{tok.value}>"
        else:
            label_value = str(item.label)
            try:
                tok = DocLangToken(label_value)
                wrap_open_token = f"<{tok.value}>"
            except ValueError:
                raise ValueError(f"Unsupported DocLang token for label '{label_value}'")

        parts: list[str] = []

        # For ListItems, emit <ldiv> as a separate delimiter element before content
        ldiv_element = ""
        if isinstance(item, ListItem):
            if item.marker:
                marker_text = _escape_text(item.marker, params)
                marker_element = _wrap(text=marker_text, wrap_tag=DocLangToken.MARKER.value)
                ldiv_element = _wrap(text=marker_element, wrap_tag=DocLangToken.LDIV.value)
            else:
                # Empty ldiv (self-closing)
                ldiv_element = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.LDIV)

        custom_head = _serialize_item_custom_head(item=item, doc_serializer=doc_serializer, params=params, **kwargs)

        # Skip adding location tokens if this is a ListItem with InlineGroup child
        # (InlineSerializer will handle location tokens using parent's provenance)
        skip_location = isinstance(item, ListItem) and self._should_skip_location_for_list_item(item=item, doc=doc)

        code_label: Optional[str] = None
        if isinstance(item, CodeItem):
            code_label = _element_label_for_serialization(
                raw_label=_code_language_label_to_doclang(
                    item.code_language,
                    interpret_unknown_as_other=params.interpret_code_unknown_as_other,
                ),
                params=params,
            )

        include_href = not is_inline_scope
        if not skip_location:
            parts.append(
                _element_head_prefix(
                    item=item,
                    doc=doc,
                    params=params,
                    label_value=code_label,
                    custom_text=custom_head or None,
                    include_href=include_href,
                    thread_id=thread_id,
                )
            )
        else:
            if code_label:
                parts.append(_create_label_token(value=code_label))
            if thread_id:
                parts.append(DocLangVocabulary._create_threading_token(thread_id=thread_id))
            if include_href and (href_uri := _text_item_hyperlink_uri(item)):
                parts.append(_create_href_token(uri=href_uri))
            if layer_token := _create_layer_token(item=item, params=params):
                parts.append(layer_token)
            if custom_head:
                parts.append(custom_head)

        text_part = ""
        if (
            (isinstance(item, CodeItem) and ContentType.TEXT_CODE in params.content_types)
            or (isinstance(item, FormulaItem) and ContentType.TEXT_FORMULA in params.content_types)
            or (not isinstance(item, CodeItem | FormulaItem) and ContentType.TEXT_OTHER in params.content_types)
        ):
            if item.children and not item.text:
                # Check if first child is InlineGroup - if so, only serialize that as text content
                first_child_ref = item.children[0]
                first_child_item = first_child_ref.resolve(doc)

                if isinstance(first_child_item, InlineGroup):
                    # Only serialize the first child (InlineGroup) as the text content
                    # Other children are hierarchical subordinates and will be serialized separately
                    text_part = doc_serializer.serialize(item=first_child_item, visited=my_visited, **kwargs).text
                else:
                    # Serialize all children as text content
                    sub_parts: list[str] = []
                    for child_ref in item.children:
                        child_item = child_ref.resolve(doc)
                        if isinstance(item, ListItem) and _list_item_segment_sibling(child_item):
                            continue
                        sub_parts.append(doc_serializer.serialize(item=child_item, visited=my_visited, **kwargs).text)
                    text_part = _get_delim(params=params).join(sub_parts)
            else:
                text_part = _escape_text(item.text, params)
                text_part = doc_serializer.post_process(
                    text=text_part,
                    formatting=item.formatting,
                    hyperlink=None,
                )
                if item.label == DocItemLabel.HANDWRITTEN_TEXT:
                    text_part = _wrap(text=text_part, wrap_tag=DocLangToken.HANDWRITING.value)
                elif item.label in [
                    DocItemLabel.CHECKBOX_SELECTED,
                    DocItemLabel.CHECKBOX_UNSELECTED,
                ]:
                    # Add checkbox token before the text
                    checkbox_token = DocLangVocabulary._create_checkbox_token(
                        selected=(item.label == DocItemLabel.CHECKBOX_SELECTED)
                    )
                    text_part = checkbox_token + text_part

            if text_part:
                parts.append(text_part)

        if params.add_referenced_caption and isinstance(item, FloatingItem):
            cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
            if cap_text:
                cap_text = _escape_text(cap_text, params)
                parts.append(cap_text)

        if params.add_referenced_footnote and isinstance(item, FloatingItem):
            ftn_text = doc_serializer.serialize_footnotes(item=item, **kwargs).text
            if ftn_text:
                ftn_text = _escape_text(ftn_text, params)
                parts.append(ftn_text)

        text_res = "".join(parts)

        # Special handling for ListItems with suppress_empty_elements
        if isinstance(item, ListItem) and params.suppress_empty_elements and not text_res:
            # Empty ListItem with suppression: emit nothing (not even ldiv)
            # This allows the list serializer to detect all-empty lists and suppress them
            return create_ser_result(text="", span_source=item)

        if wrap_open_token is not None and not (
            is_inline_scope
            and item.label
            in {
                DocItemLabel.TEXT,
                DocItemLabel.HANDWRITTEN_TEXT,
                DocItemLabel.CHECKBOX_SELECTED,
                DocItemLabel.CHECKBOX_UNSELECTED,
            }
        ):
            if text_res or not params.suppress_empty_elements:
                text_res = _wrap_token(text=text_res, open_token=wrap_open_token)
                if isinstance(item, FieldHeadingItem):
                    text_res = _wrap_in_field_region_if_needed(text=text_res, item=item, doc=doc)
                elif item.label in (DocItemLabel.FIELD_KEY, DocItemLabel.FIELD_VALUE):
                    text_res = _wrap_field_kv_markup_if_needed(text=text_res, item=item, doc=doc)

        # Prepend ldiv element for ListItems (it's a delimiter, not a wrapper)
        if ldiv_element:
            text_res = ldiv_element + text_res

        return create_ser_result(text=text_res, span_source=item)


class DocLangMetaSerializer(BaseModel, BaseMetaSerializer):
    """DocLang-specific meta serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """DocLang-specific meta serializer."""
        params = DocLangParams(**kwargs)

        elem_delim = ""
        texts = (
            [
                tmp
                for key in (list(item.meta.__class__.model_fields) + list(item.meta.get_custom_part()))
                if (
                    (params.allowed_meta_names is None or key in params.allowed_meta_names)
                    and (key not in params.blocked_meta_names)
                    and (tmp := self._serialize_meta_field(item.meta, key, params))
                )
            ]
            if item.meta
            else []
        )
        if texts:
            texts.insert(0, f"<{DocLangToken.CUSTOM.value}>")
            texts.append(f"</{DocLangToken.CUSTOM.value}>")
        return create_ser_result(
            text=elem_delim.join(texts),
            span_source=item if isinstance(item, DocItem) else [],
        )

    def _serialize_meta_field(self, meta: BaseMeta, name: str, params: DocLangParams) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None:
            if name == MetaFieldName.SUMMARY and isinstance(field_val, SummaryMetaField):
                escaped_text = _escape_text(field_val.text, params)
                txt = f"<docling__summary>{escaped_text}</docling__summary>"
            elif name == MetaFieldName.DESCRIPTION and isinstance(field_val, DescriptionMetaField):
                escaped_text = _escape_text(field_val.text, params)
                txt = f"<docling__description>{escaped_text}</docling__description>"
            elif name == MetaFieldName.CLASSIFICATION:
                # Picture classification is emitted as <label value="..."/> in element head.
                return None
            elif name == MetaFieldName.MOLECULE and isinstance(field_val, MoleculeMetaField):
                escaped_smi = _escape_text(field_val.smi, params)
                txt = f"<docling__smiles>{escaped_smi}</docling__smiles>"
            elif name == MetaFieldName.TABULAR_CHART and isinstance(field_val, TabularChartMetaField):
                # suppressing tabular chart serialization
                return None
            # elif tmp := str(field_val or ""):
            #     txt = tmp
            elif name not in {v.value for v in MetaFieldName}:
                escaped_text = _escape_text(str(field_val or ""), params)
                txt = _wrap(text=escaped_text, wrap_tag=name)
            return txt
        return None


class DocLangPictureSerializer(BasePictureSerializer):
    """DocLang-specific picture item serializer."""

    def _picture_is_chart(self, item: PictureItem) -> bool:
        """Check if predicted class indicates a chart."""
        if item.meta and item.meta.classification:
            return item.meta.classification.get_main_prediction().class_name in {
                PictureClassificationLabel.PIE_CHART.value,
                PictureClassificationLabel.BAR_CHART.value,
                PictureClassificationLabel.STACKED_BAR_CHART.value,
                PictureClassificationLabel.LINE_CHART.value,
                PictureClassificationLabel.FLOW_CHART.value,
                PictureClassificationLabel.SCATTER_CHART.value,
                PictureClassificationLabel.HEATMAP.value,
            }
        return False

    def _picture_is_chem(self, item: PictureItem) -> bool:
        """Check if predicted class indicates a chemistry structure."""
        if item.meta and item.meta.molecule:
            return True
        return False

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
        params = DocLangParams(**kwargs)
        res_parts: list[SerializationResult] = []

        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result()

        caption_head = _serialize_floating_caption_head(
            item=item, doc_serializer=doc_serializer, doc=doc, params=params, **kwargs
        )
        if caption_head:
            res_parts.append(create_ser_result(text=caption_head))

        body_parts: list[str] = []
        is_chart = self._picture_is_chart(item)
        is_chem = self._picture_is_chem(item)
        has_picture_ct = ContentType.PICTURE in params.content_types
        specific_match = (is_chart and ContentType.CHART in params.content_types) or (
            is_chem and ContentType.CHEMISTRY in params.content_types
        )
        any_match = has_picture_ct or specific_match

        picture_label = _element_label_for_serialization(
            raw_label=_picture_classification_label_value(item),
            params=params,
        )
        custom_head = ""
        if any_match and item.meta:
            meta_kwargs = dict(**kwargs)
            blocked = set(params.blocked_meta_names) | {MetaFieldName.CLASSIFICATION}
            if not specific_match:
                blocked |= {MetaFieldName.MOLECULE, MetaFieldName.TABULAR_CHART}
            meta_kwargs["blocked_meta_names"] = blocked
            meta_res = doc_serializer.serialize_meta(item=item, **meta_kwargs)
            if meta_res.text:
                custom_head = meta_res.text
                res_parts.append(meta_res)

            if specific_match and item.meta and item.meta.tabular_chart:
                chart_data = item.meta.tabular_chart.chart_data
                if chart_data and chart_data.table_cells:
                    temp_doc = DoclingDocument(name="temp")
                    temp_table = temp_doc.add_table(data=chart_data)
                    params_chart = DocLangParams(**{**params.model_dump(), "add_table_cell_location": False})
                    otsl_content = DocLangTableSerializer()._emit_otsl(
                        item=temp_table,  # type: ignore[arg-type]
                        doc_serializer=doc_serializer,
                        doc=temp_doc,
                        params=params_chart,
                        **kwargs,
                    )
                    body_parts.append(_wrap(text=otsl_content, wrap_tag=DocLangToken.TABLE.value))

        uri: Optional[str] = None
        if params.image_mode in [ImageRefMode.REFERENCED, ImageRefMode.EMBEDDED] and item.image and item.image.uri:
            uri = str(item.image.uri)
        elif params.image_mode == ImageRefMode.EMBEDDED and (img := item.get_image(doc)):
            imgb64 = item._image_to_base64(img)
            uri = f"data:image/png;base64,{imgb64}"
        if uri:
            body_parts.append(_create_src_token(uri=uri))

        head = _element_head_prefix(
            item=item,
            doc=doc,
            params=params,
            label_value=picture_label,
            caption_text=caption_head or None,
            custom_text=custom_head or None,
        )
        inner = head + "".join(body_parts)
        picture_open = f"<{DocLangToken.PICTURE.value}"
        if body_parts and any(p.startswith(f"<{DocLangToken.TABLE.value}") for p in body_parts):
            picture_open += f' {DocLangAttributeKey.CLASS.value}="chart"'
        picture_open += ">"
        picture_text = f"{picture_open}{inner}</{DocLangToken.PICTURE.value}>"

        footnote_text = ""
        if params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        if not inner and not footnote_text:
            if params.suppress_empty_elements:
                return create_ser_result()
            text_res = f"<{DocLangToken.PICTURE.value}></{DocLangToken.PICTURE.value}>"
        elif footnote_text:
            text_res = _wrap(text=picture_text + footnote_text, wrap_tag=DocLangToken.GROUP.value)
        else:
            text_res = picture_text

        return create_ser_result(text=text_res, span_source=res_parts)


def _table_fragment_bounds(item: TableItem, prov_index: int) -> tuple[int, int, int, int]:
    """Return half-open ``(row_start, row_end, col_start, col_end)`` for a prov fragment."""
    nprov = len(item.prov)
    if not item.data:
        return 0, 0, 0, 0
    nrows, ncols = item.data.num_rows, item.data.num_cols
    if nprov <= 1:
        return 0, nrows, 0, ncols
    page_nos = [p.page_no for p in item.prov]
    if len(set(page_nos)) == 1:
        c0 = prov_index * ncols // nprov
        c1 = (prov_index + 1) * ncols // nprov
        return 0, nrows, c0, c1
    r0 = prov_index * nrows // nprov
    r1 = (prov_index + 1) * nrows // nprov
    return r0, r1, 0, ncols


class DocLangTableSerializer(BaseTableSerializer):
    """DocLang-specific table item serializer."""

    # _get_table_token no longer needed; OTSL tokens are emitted via vocabulary

    @staticmethod
    def _otsl_origin_token_for_slice(
        *,
        cell: TableCell,
        row_idx: int,
        col_idx: int,
        rowstart: int,
        colstart: int,
        row_start: int,
        col_start: int,
        has_content: bool,
    ) -> DocLangToken:
        """Pick OTSL origin token for a cell origin inside a table fragment slice."""
        cont_left = col_idx == col_start and col_start > 0
        cont_up = rowstart < row_start and row_idx == row_start
        if cont_left and cont_up:
            return DocLangToken.XCEL
        if cont_up:
            return DocLangToken.UCEL
        if cont_left:
            return DocLangToken.LCEL
        if has_content:
            if cell.column_header and cell.row_header:
                return DocLangToken.CORN
            if cell.column_header:
                return DocLangToken.CHED
            if cell.row_header:
                return DocLangToken.RHED
            if cell.row_section:
                return DocLangToken.SROW
            return DocLangToken.FCEL
        if cell.column_header and cell.row_header:
            return DocLangToken.CORN
        return DocLangToken.ECEL

    def _emit_otsl(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        params: "DocLangParams",
        row_start: int = 0,
        row_end: Optional[int] = None,
        col_start: int = 0,
        col_end: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Emit OTSL payload using DocLang tokens and location semantics.

        Location tokens are included only when all required information is available
        (cell bboxes, provenance, page info, valid page size). Otherwise, location
        tokens are omitted without raising errors.

        Optional ``row_*`` / ``col_*`` bounds restrict output to a table fragment slice
        (for multi-prov threading); continuation tokens (``lcel``/``ucel``/``xcel``)
        are used at slice edges per the DocLang OTSL rules.
        """
        if not item.data or not item.data.table_cells:
            return ""

        nrows, ncols = item.data.num_rows, item.data.num_cols
        row_end = nrows if row_end is None else row_end
        col_end = ncols if col_end is None else col_end

        # Determine if we need page context for location serialization
        # Only proceed if all required information is available
        need_cell_loc = False
        page_no = 0
        page_w, page_h = (1.0, 1.0)

        if params.add_table_cell_location:
            # Check if we have all required information for location serialization
            if item.prov and len(item.prov) > 0:
                page_no = item.prov[0].page_no
                if doc.pages and page_no in doc.pages:
                    page_w, page_h = doc.pages[page_no].size.as_tuple()
                    if page_w > 0 and page_h > 0:
                        # All prerequisites met, enable location serialization
                        # Individual cells will still be checked for bbox availability
                        need_cell_loc = True

        parts: list[str] = []
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                cell = item.data.grid[i][j]
                content = cell._get_text(doc=doc, doc_serializer=doc_serializer, **kwargs).strip()

                rowstart = cell.start_row_offset_idx
                colstart = cell.start_col_offset_idx

                # Optional per-cell location
                cell_loc = ""
                if need_cell_loc and cell.bbox is not None:
                    bbox = cell.bbox.to_top_left_origin(page_h).as_tuple()
                    cell_loc = _create_location_tokens_for_bbox(
                        bbox=bbox,
                        page_w=page_w,
                        page_h=page_h,
                        xres=params.xsize,
                        yres=params.ysize,
                    )

                if rowstart == i and colstart == j:
                    origin = self._otsl_origin_token_for_slice(
                        cell=cell,
                        row_idx=i,
                        col_idx=j,
                        rowstart=rowstart,
                        colstart=colstart,
                        row_start=row_start,
                        col_start=col_start,
                        has_content=bool(content),
                    )
                    parts.append(DocLangVocabulary._create_selfclosing_token(token=origin))
                    if content and origin != DocLangToken.ECEL:
                        if cell_loc:
                            parts.append(cell_loc)
                        if ContentType.TABLE_CELL in params.content_types:
                            if not isinstance(cell, RichTableCell):
                                content = _escape_text(content, params)
                                if not params.use_virtual_text:
                                    content = _wrap(text=content, wrap_tag=DocLangToken.TEXT.value)
                            parts.append(content)
                elif rowstart != i and colstart != j:
                    parts.append(DocLangVocabulary._create_selfclosing_token(token=DocLangToken.XCEL))
                elif rowstart != i:
                    parts.append(DocLangVocabulary._create_selfclosing_token(token=DocLangToken.UCEL))
                elif colstart != j:
                    parts.append(DocLangVocabulary._create_selfclosing_token(token=DocLangToken.LCEL))

            parts.append(DocLangVocabulary._create_selfclosing_token(token=DocLangToken.NL))

        return "".join(parts)

    def _serialize_single_table(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        params: DocLangParams,
        visited: Optional[set[str]] = None,
        thread_id: Optional[str] = None,
        include_caption_head: bool = True,
        row_start: int = 0,
        row_end: Optional[int] = None,
        col_start: int = 0,
        col_end: Optional[int] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize one table fragment (single provenance span)."""
        from docling_core.types.doc.labels import DocItemLabel

        res_parts: list[SerializationResult] = []

        caption_head = ""
        if include_caption_head:
            caption_head = _serialize_floating_caption_head(
                item=item, doc_serializer=doc_serializer, doc=doc, params=params, **kwargs
            )
            if caption_head:
                res_parts.append(create_ser_result(text=caption_head))

        host_token = DocLangToken.INDEX if item.label == DocItemLabel.DOCUMENT_INDEX else DocLangToken.TABLE
        inner_parts: list[str] = []
        if ContentType.TABLE in params.content_types:
            otsl_text = self._emit_otsl(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                params=params,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                visited=visited,
                **kwargs,
            )
            if otsl_text:
                inner_parts.append(otsl_text)

        head = _element_head_prefix(
            item=item,
            doc=doc,
            params=params,
            caption_text=caption_head or None,
            thread_id=thread_id,
        )
        table_text = _wrap(text=head + "".join(inner_parts), wrap_tag=host_token.value)
        res_parts.append(create_ser_result(text=head + "".join(inner_parts), span_source=item))

        footnote_text = ""
        if include_caption_head and params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        if not (head or inner_parts) and not footnote_text:
            if params.suppress_empty_elements:
                return create_ser_result()
            text_res = f"<{host_token.value}></{host_token.value}>"
        elif footnote_text:
            text_res = _wrap(text=table_text + footnote_text, wrap_tag=DocLangToken.GROUP.value)
        else:
            text_res = table_text

        return create_ser_result(text=text_res, span_source=res_parts)

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocLangParams(**kwargs)

        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result()

        if len(item.prov) > 1:
            thread_id = _allocate_thread_id(doc_serializer, item)
            res: list[SerializationResult] = []
            for idp, prov_ in enumerate(item.prov):
                row_start, row_end, col_start, col_end = _table_fragment_bounds(item, idp)
                item_ = copy.deepcopy(item)
                item_.prov = [prov_]
                tres = self._serialize_single_table(
                    item=item_,
                    doc_serializer=doc_serializer,
                    doc=doc,
                    params=params,
                    visited=visited,
                    thread_id=thread_id,
                    include_caption_head=idp == 0,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    **kwargs,
                )
                res.append(tres)

            out_parts: list[str] = []
            for idp, tres in enumerate(res):
                if idp > 0 and item.prov[idp - 1].page_no != item.prov[idp].page_no:
                    pb = _PageBreakNode(
                        self_ref=f"#/pb/{idp}",
                        prev_page=item.prov[idp - 1].page_no,
                        next_page=item.prov[idp].page_no,
                    )
                    _suppress_document_page_break(
                        doc_serializer,
                        prev_page=item.prov[idp - 1].page_no,
                        next_page=item.prov[idp].page_no,
                    )
                    out_parts.append(_create_page_break_markup(pb))
                out_parts.append(tres.text)
            return create_ser_result(text="".join(out_parts), span_source=res)

        return self._serialize_single_table(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            params=params,
            visited=visited,
            **kwargs,
        )


class DocLangInlineSerializer(BaseInlineSerializer):
    """Inline serializer emitting DocLang `<inline>` and `<location>` tokens."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize inline content with optional location into DocLang."""
        my_visited = visited if visited is not None else set()
        params = DocLangParams(**kwargs)
        parts: list[SerializationResult] = []
        if params.add_location:
            # Check if parent is ListItem with provenance - use that instead of children
            parent_item = item.parent.resolve(doc) if item.parent else None
            if isinstance(parent_item, ListItem) and parent_item.prov:
                # Use parent ListItem's provenance
                for prov in parent_item.prov:
                    page_w, page_h = doc.pages[prov.page_no].size.as_tuple()
                    bbox = prov.bbox.to_top_left_origin(page_h).as_tuple()
                    loc_str = _create_location_tokens_for_bbox(
                        bbox=bbox,
                        page_w=page_w,
                        page_h=page_h,
                        xres=params.xsize,
                        yres=params.ysize,
                    )
                    parts.append(create_ser_result(text=loc_str))
            else:
                # Create a single enclosing bbox over inline children
                boxes: list[tuple[float, float, float, float]] = []
                prov_page_w_h: Optional[tuple[float, float, int]] = None
                for it, _ in doc.iterate_items(root=item):
                    if isinstance(it, DocItem) and it.prov:
                        for prov in it.prov:
                            page_w, page_h = doc.pages[prov.page_no].size.as_tuple()
                            boxes.append(prov.bbox.to_top_left_origin(page_h).as_tuple())
                            prov_page_w_h = (page_w, page_h, prov.page_no)
                if boxes and prov_page_w_h is not None:
                    x0 = min(b[0] for b in boxes)
                    y0 = min(b[1] for b in boxes)
                    x1 = max(b[2] for b in boxes)
                    y1 = max(b[3] for b in boxes)
                    page_w, page_h, _ = prov_page_w_h
                    loc_str = _create_location_tokens_for_bbox(
                        bbox=(x0, y0, x1, y1),
                        page_w=page_w,
                        page_h=page_h,
                        xres=params.xsize,
                        yres=params.ysize,
                    )
                    parts.append(create_ser_result(text=loc_str))
            params.add_location = False
        parts.extend(
            doc_serializer.get_parts(
                item=item,
                list_level=list_level,
                is_inline_scope=True,
                visited=my_visited,
                **{**kwargs, **params.model_dump()},
            )
        )
        delim = _get_delim(params=params)
        text_res = delim.join([p.text for p in parts if p.text])
        if text_res:
            text_res = f"{text_res}{delim}"

        parent_item = item.parent.resolve(doc) if item.parent else None
        if parent_item is None:
            should_wrap = True
        elif isinstance(parent_item, ListItem):
            should_wrap = not params.use_virtual_text or _list_item_has_segment_siblings(item=parent_item, doc=doc)
        elif isinstance(parent_item, TextItem):
            should_wrap = False
        else:
            should_wrap = True
        if should_wrap:
            # if "unwrapped", wrap in <text>...</text>
            if text_res or not params.suppress_empty_elements:
                text_res = _wrap(text=text_res, wrap_tag=DocLangToken.TEXT.value)
        return create_ser_result(text=text_res, span_source=parts)


class DocLangFallbackSerializer(BaseFallbackSerializer):
    """Fallback serializer concatenating text for list/inline groups."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize unsupported nodes by concatenating their textual parts."""
        params = DocLangParams(**kwargs)
        delim = _get_delim(params=DocLangParams(**kwargs))
        if isinstance(item, GroupItem):
            parts = doc_serializer.get_parts(item=item, **kwargs)
            text_res = delim.join([p.text for p in parts if p.text])
            return create_ser_result(text=text_res, span_source=parts)
        elif isinstance(item, FieldRegionItem | FieldItem):
            parts = []
            is_fri = isinstance(item, FieldRegionItem)
            # Element head (layer, location) for field regions only
            if is_fri and (head := _element_head_prefix(item=item, doc=doc, params=params)):
                parts.append(create_ser_result(text=head, span_source=item))
            parts.extend(doc_serializer.get_parts(item=item, **kwargs))
            text_res = delim.join([p.text for p in parts if p.text])
            tok = DocLangToken.FIELD_REGION if is_fri else DocLangToken.FIELD_ITEM
            text_res = _wrap(text=text_res, wrap_tag=tok.value)
            if isinstance(item, FieldItem):
                text_res = _wrap_in_field_region_if_needed(text=text_res, item=item, doc=doc)
            return create_ser_result(text=text_res, span_source=parts)
        return create_ser_result()


class DocLangKeyValueSerializer(BaseKeyValueSerializer):
    """No-op serializer for key/value items in DocLang."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Return an empty result for key/value items."""
        return create_ser_result()


class DocLangFormSerializer(BaseFormSerializer):
    """No-op serializer for form items in DocLang."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Return an empty result for form items."""
        return create_ser_result()


class DocLangAnnotationSerializer(BaseAnnotationSerializer):
    """No-op annotation serializer; DocLang relies on meta instead."""

    @override
    def serialize(
        self,
        *,
        item: DocItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Return an empty result; annotations are handled via meta."""
        return create_ser_result()


class DocLangDocSerializer(DocSerializer):
    """DocLang document serializer."""

    _suppressed_page_breaks: set[tuple[int, int]] = PrivateAttr(default_factory=set)
    _next_thread_id: int = PrivateAttr(default=1)
    _thread_id_by_ref: dict[str, str] = PrivateAttr(default_factory=dict)

    def allocate_thread_id(self, node: NodeItem) -> str:
        """Return a spec-unique positive ``thread_id`` for a fragmented component."""
        if node.self_ref in self._thread_id_by_ref:
            return self._thread_id_by_ref[node.self_ref]
        thread_id = str(self._next_thread_id)
        self._next_thread_id += 1
        self._thread_id_by_ref[node.self_ref] = thread_id
        return thread_id

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
        """Serialize a node, suppressing redundant page breaks already emitted by list/table threading."""
        if isinstance(item, _PageBreakNode):
            key = (item.prev_page, item.next_page)
            if key in self._suppressed_page_breaks:
                self._suppressed_page_breaks.discard(key)
                return create_ser_result()
        return super().serialize(
            item=item,
            list_level=list_level,
            is_inline_scope=is_inline_scope,
            visited=visited,
            **kwargs,
        )

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ) -> str:
        """Hyperlinks are emitted as ``<href uri=\"...\"/>`` in element head, not inline."""
        return text

    text_serializer: BaseTextSerializer = DocLangTextSerializer()
    table_serializer: BaseTableSerializer = DocLangTableSerializer()
    picture_serializer: BasePictureSerializer = DocLangPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = DocLangKeyValueSerializer()
    form_serializer: BaseFormSerializer = DocLangFormSerializer()
    fallback_serializer: BaseFallbackSerializer = DocLangFallbackSerializer()

    list_serializer: BaseListSerializer = DocLangListSerializer()
    inline_serializer: BaseInlineSerializer = DocLangInlineSerializer()

    meta_serializer: BaseMetaSerializer = DocLangMetaSerializer()
    annotation_serializer: BaseAnnotationSerializer = DocLangAnnotationSerializer()

    params: DocLangParams = DocLangParams()

    @override
    def _meta_is_wrapped(self) -> bool:
        return True

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's captions with DocLang location tokens."""
        params = DocLangParams(**kwargs)
        results: list[SerializationResult] = []
        if item.captions:
            cap_res = super().serialize_captions(item, **kwargs)
            if cap_res.text:
                for caption in item.captions:
                    if caption.cref not in self.get_excluded_refs(**kwargs):
                        if isinstance(cap := caption.resolve(self.doc), DocItem):
                            if head_txt := _element_head_prefix(item=cap, doc=self.doc, params=params):
                                results.append(create_ser_result(text=head_txt))
            if cap_res.text and ContentType.REF_CAPTION in params.content_types:
                cap_res.text = _escape_text(cap_res.text, params)
                results.append(cap_res)
        text_res = "".join([r.text for r in results])
        if text_res:
            text_res = _wrap(text=text_res, wrap_tag=DocLangToken.CAPTION.value)
        return create_ser_result(text=text_res, span_source=results)

    @override
    def serialize_footnotes(
        self,
        item: FloatingItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's footnotes with DocLang location tokens."""
        params = DocLangParams(**kwargs)
        results: list[SerializationResult] = []
        for footnote in item.footnotes:
            if footnote.cref not in self.get_excluded_refs(**kwargs):
                if isinstance(ftn := footnote.resolve(self.doc), TextItem):
                    head = _element_head_prefix(item=ftn, doc=self.doc, params=params)

                    content = ""
                    if ftn.text and ContentType.REF_FOOTNOTE in params.content_types:
                        content = _escape_text(ftn.text, params)

                    text_res = f"{head}{content}"
                    if text_res:
                        text_res = _wrap(text_res, wrap_tag=DocLangToken.FOOTNOTE.value)
                        results.append(create_ser_result(text=text_res))

        text_res = "".join([r.text for r in results])

        return create_ser_result(text=text_res, span_source=results)

    def _create_head(self) -> str:
        """Create the head section of the DocLang document."""
        parts = []
        if self.params.xsize != DOCLANG_DFLT_RESOLUTION or self.params.ysize != DOCLANG_DFLT_RESOLUTION:
            parts.append(f'<default_resolution width="{self.params.xsize}" height="{self.params.ysize}"/>')
        return _wrap(text="".join(parts), wrap_tag=DocLangToken.HEAD.value) if parts else ""

    def _is_content_tag(self, tag: str) -> bool:
        return tag in {DocLangToken.CONTENT.value}

    def _remove_content_subtrees(self, element: ET.Element) -> None:
        """Remove any child that is a regarded as content element and its entire subtree."""
        to_remove = []
        for child in element:
            if self._is_content_tag(child.tag):
                to_remove.append(child)
            else:
                self._remove_content_subtrees(child)
        for child in to_remove:
            element.remove(child)

    def _strip_text_from_element(self, element: ET.Element) -> None:
        """Remove all text content and whitespace from element."""
        element.text = None
        for child in element:
            self._strip_text_from_element(child)
            child.tail = None

    def _filter_out_all_content(self, text: str) -> str:
        root = fromstring(text)
        self._remove_content_subtrees(root)
        self._strip_text_from_element(root)
        out = ET.tostring(root, encoding="unicode", method="xml", short_empty_elements=True)
        return out

    def _serialize_body(self, **kwargs) -> SerializationResult:
        """Serialize the document body."""

        self._suppressed_page_breaks = set()
        self._next_thread_id = 1
        self._thread_id_by_ref = {}

        # intercept from DocLangDocSerializer to update param kwargs:
        my_delta = {}
        if not self.params.add_content:
            # in this case filtering is done by XML-based post-processing
            params = self.params.model_copy(deep=True)
            params.content_types = set(ContentType)
            my_delta = params.model_dump()
        my_kwargs = {**kwargs, **my_delta}
        subparts = self.get_parts(**my_kwargs)
        res = self.serialize_doc(parts=subparts, **my_kwargs)
        return res

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Doc-level serialization with DocLang root wrapper."""
        # Note: removed internal thread counting; not used.

        delim = _get_delim(params=self.params)

        open_token: str = DocLangVocabulary._create_doclang_root(
            namespace=DOCLANG_NAMESPACE if self.params.include_namespace else None,
            version=_DOCLANG_VERSION if self.params.include_version else None,
        )
        head = self._create_head()
        close_token: str = DocLangVocabulary._create_doclang_root(closing=True)

        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            # Always emit well-formed page breaks using the vocabulary
            page_sep = DocLangVocabulary._create_selfclosing_token(token=DocLangToken.PAGE_BREAK)
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        text_res = f"{open_token}{head}{text_res}{close_token}"

        if not self.params.add_content:
            # do XML-based post-filtering
            text_res = self._filter_out_all_content(text_res)

        if self.params.pretty_indentation is not None:
            try:
                my_root = parseString(text_res).documentElement
            except Exception as e:
                # print(text_res)

                ctx = _xml_error_context(text_res, e)
                raise ValueError(f"XML pretty-print failed: {e}\n--- XML context ---\n{ctx}") from e
            if my_root is None:
                raise ValueError("XML pretty-print failed: documentElement is None")
            text_res = my_root.toprettyxml(indent=self.params.pretty_indentation)

            # Filter out empty lines, but preserve them inside <content> tags
            lines = text_res.split("\n")
            filtered_lines = []
            inside_content = False
            for line in lines:
                # Check if we're entering or exiting a content tag
                if "<content>" in line or "<content " in line:
                    inside_content = True
                if "</content>" in line:
                    # Add the line first, then mark as outside content
                    filtered_lines.append(line)
                    inside_content = False
                    continue

                # Keep all lines inside content tags, filter empty lines outside
                if inside_content or line.strip():
                    filtered_lines.append(line)

            text_res = "\n".join(filtered_lines)

            if self.params.preserve_empty_non_selfclosing:
                # Expand self-closing forms for tokens that are not allowed
                # to be self-closing according to the vocabulary.
                # Example: <ldiv/> -> <ldiv></ldiv>
                non_selfclosing = [tok for tok in DocLangToken if tok not in DocLangVocabulary.IS_SELFCLOSING]

                def _expand_tag(text: str, name: str) -> str:
                    # Match <name/> or <name .../>
                    pattern = rf"<\s*{name}(\s[^>]*)?/\s*>"
                    return re.sub(pattern, rf"<{name}\1></{name}>", text)

                for tok in non_selfclosing:
                    text_res = _expand_tag(text_res, tok.value)

        return create_ser_result(text=text_res, span_source=parts)

    @override
    def requires_page_break(self):
        """Return whether page breaks should be emitted for the document."""
        return self.params.add_page_break

    @override
    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific bold serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.BOLD.value)

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific italic serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.ITALIC.value)

    @override
    def serialize_underline(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific underline serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.UNDERLINE.value)

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific strikethrough serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.STRIKETHROUGH.value)

    @override
    def serialize_subscript(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific subscript serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.SUBSCRIPT.value)

    @override
    def serialize_superscript(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific superscript serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.SUPERSCRIPT.value)

    def serialize_rtl(self, text: str, **kwargs: Any) -> str:
        """Apply DocLang-specific right-to-left text serialization."""
        return _wrap(text=text, wrap_tag=DocLangToken.RTL.value)

    @override
    def post_process(
        self,
        text: str,
        *,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """Apply DocLang text post-processing including RTL direction."""
        res = super().post_process(
            text=text,
            formatting=formatting,
            hyperlink=hyperlink,
            **kwargs,
        )
        params = self.params.merge_with_patch(patch=kwargs)
        if params.include_formatting and get_text_direction(text) == "rtl":
            res = self.serialize_rtl(text=res, **kwargs)
        return res
