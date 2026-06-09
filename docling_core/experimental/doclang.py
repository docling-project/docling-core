"""Define classes for Doclang serialization."""

import copy
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar, Final, Optional, Union, cast
from xml.dom.minidom import Element, Node, Text

from defusedxml.ElementTree import fromstring
from defusedxml.minidom import parseString
from pydantic import BaseModel, PrivateAttr
from pydantic.networks import AnyUrl
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

# Note: Intentionally avoid importing DocumentToken here to ensure
# Doclang uses only its own token vocabulary.

DOCLANG_NAMESPACE: Final = "https://www.doclang.ai/ns/v0"
_DOCLANG_VERSION: Final = "0.5"
DOCLANG_DFLT_RESOLUTION: int = 512

ET.register_namespace("", DOCLANG_NAMESPACE)  # prevent prefix from ET.tostring()


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


def _has_field_region_ancestor(*, item: DocItem, doc: DoclingDocument) -> bool:
    """True when ``item`` sits under a :class:`FieldRegionItem` in the document tree."""
    parent_ref = item.parent
    while parent_ref is not None:
        parent = parent_ref.resolve(doc)
        if isinstance(parent, FieldRegionItem):
            return True
        if parent.self_ref == doc.body.self_ref:
            return False
        parent_ref = parent.parent if isinstance(parent, NodeItem) else None
    return False


def _has_field_item_ancestor(*, item: DocItem, doc: DoclingDocument) -> bool:
    """True when ``item`` sits under a :class:`FieldItem` in the document tree."""
    parent_ref = item.parent
    while parent_ref is not None:
        parent = parent_ref.resolve(doc)
        if isinstance(parent, FieldItem):
            return True
        if parent.self_ref == doc.body.self_ref:
            return False
        parent_ref = parent.parent if isinstance(parent, NodeItem) else None
    return False


def _wrap_in_field_region_if_needed(*, text: str, item: DocItem, doc: DoclingDocument) -> str:
    """Wrap serialized field markup in ``<field_region>`` when not already under one."""
    if _has_field_region_ancestor(item=item, doc=doc):
        return text
    return _wrap(text=text, wrap_tag=DoclangToken.FIELD_REGION.value)


def _wrap_in_field_item_if_needed(*, text: str, item: DocItem, doc: DoclingDocument) -> str:
    """Wrap serialized key/value markup in ``<field_item>`` when not already under one."""
    if _has_field_item_ancestor(item=item, doc=doc):
        return text
    return _wrap(text=text, wrap_tag=DoclangToken.FIELD_ITEM.value)


def _wrap_field_kv_markup_if_needed(*, text: str, item: DocItem, doc: DoclingDocument) -> str:
    """Ensure key/value XML is nested under ``field_item`` (and ``field_region`` when orphan)."""
    text = _wrap_in_field_item_if_needed(text=text, item=item, doc=doc)
    # Keys/values under an existing field_item rely on that item's field_region wrapper.
    if not _has_field_item_ancestor(item=item, doc=doc):
        text = _wrap_in_field_region_if_needed(text=text, item=item, doc=doc)
    return text


def _wrap_token(*, text: str, open_token: str) -> str:
    close_token = DoclangVocabulary._create_closing_token(token=open_token)
    return f"{open_token}{text}{close_token}"


def _xml_error_context(
    text: str,
    err: Exception,
    *,
    radius_lines: int = 2,
    max_line_chars: int = 500,
    max_total_chars: int = 2000,
) -> str:
    lineno = getattr(err, "lineno", None)
    offset = getattr(err, "offset", None)
    if not lineno or lineno <= 0:
        m = re.search(r"line\s+(\d+)\s*,\s*column\s+(\d+)", str(err))
        if m:
            try:
                lineno = int(m.group(1))
                offset = int(m.group(2))
            except Exception:
                lineno = None
                offset = None
    if not lineno or lineno <= 0:
        snippet = text[:max_total_chars]
        if len(text) > max_total_chars:
            snippet += " …"
        return snippet
    lines = text.splitlines()
    lineno = min(max(1, lineno), len(lines))
    start = max(1, lineno - radius_lines)
    end = min(len(lines), lineno + radius_lines)
    out: list[str] = []
    for i in range(start, end + 1):
        line = lines[i - 1]
        line_display = line[:max_line_chars]
        if len(line) > max_line_chars:
            line_display += " …"
        out.append(f"{i:>6}: {line_display}")
        if i == lineno and offset and offset > 0:
            caret_pos = min(offset - 1, len(line_display))
            prefix_len = len(f"{i:>6}: ")
            out.append(" " * (prefix_len + caret_pos) + "^")
    return "\n".join(out)


def _quantize_to_resolution(value: float, resolution: int) -> int:
    """Quantize normalized value in [0,1) to [0,resolution)."""
    n = round(resolution * value)
    if n < 0:
        warnings.warn(f"Normalized {value=} less than 0; returning 0", stacklevel=2)
        return 0
    elif n >= resolution:
        warnings.warn(
            f"Normalized {value=} greater or equal to 1; returning {resolution-1=}",
            stacklevel=2,
        )
        return resolution - 1
    else:
        return n


def _create_location_tokens_for_bbox(
    *,
    bbox: tuple[float, float, float, float],
    page_w: float,
    page_h: float,
    xres: int,
    yres: int,
) -> str:
    """Create four `<location .../>` tokens for x0,y0,x1,y1 given a bbox."""
    x0 = bbox[0] / page_w
    y0 = bbox[1] / page_h
    x1 = bbox[2] / page_w
    y1 = bbox[3] / page_h

    x0v = _quantize_to_resolution(min(x0, x1), xres)
    y0v = _quantize_to_resolution(min(y0, y1), yres)
    x1v = _quantize_to_resolution(max(x0, x1), xres)
    y1v = _quantize_to_resolution(max(y0, y1), yres)

    return (
        DoclangVocabulary._create_location_token(value=x0v, resolution=xres)
        + DoclangVocabulary._create_location_token(value=y0v, resolution=yres)
        + DoclangVocabulary._create_location_token(value=x1v, resolution=xres)
        + DoclangVocabulary._create_location_token(value=y1v, resolution=yres)
    )


def _create_location_tokens_for_item(
    *,
    item: "DocItem",
    doc: "DoclingDocument",
    xres: int,
    yres: int,
) -> str:
    """Create concatenated `<location .../>` tokens for an item's provenance."""
    if not getattr(item, "prov", None):
        return ""
    out: list[str] = []
    for prov in item.prov:
        page_w, page_h = doc.pages[prov.page_no].size.as_tuple()
        bbox = prov.bbox.to_top_left_origin(page_h).as_tuple()
        out.append(_create_location_tokens_for_bbox(bbox=bbox, page_w=page_w, page_h=page_h, xres=xres, yres=yres))

    # Multi-provenance items emit one location set per fragment
    if len(out) > 1:
        res = []
        for i, _ in enumerate(item.prov):
            res.append(f"{i} {_}")
        err = "\n".join(res)

        raise ValueError(f"We have more than 1 location for this item [{item.label}]:\n\n{err}\n\n{out}")

    return "".join(out)


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
    if isinstance(doc_serializer, DoclangDocSerializer):
        doc_serializer._suppressed_page_breaks.add((prev_page, next_page))


def _allocate_thread_id(doc_serializer: BaseDocSerializer, node: NodeItem) -> str:
    """Allocate a document-scoped positive ``thread_id`` in reading order."""
    if isinstance(doc_serializer, DoclangDocSerializer):
        return doc_serializer.allocate_thread_id(node)
    raise TypeError("Doclang threading requires DoclangDocSerializer")


def _primary_page_no(node: NodeItem) -> Optional[int]:
    """Return the primary page number for a document item, if known."""
    if isinstance(node, DocItem) and node.prov:
        return node.prov[0].page_no
    return None


def _provenance_with_charspan(prov_list: list[ProvenanceItem], charspan: tuple[int, int]) -> list[ProvenanceItem]:
    """Return provenance copies with the given ``charspan``."""
    return [ProvenanceItem(page_no=prov.page_no, bbox=prov.bbox, charspan=charspan) for prov in prov_list]


def _append_textual_fragment(
    item: TextItem,
    *,
    text: str,
    prov_list: list[ProvenanceItem],
) -> None:
    """Append a threaded fragment to a text-like item, updating ``charspan`` offsets."""
    offset = len(item.orig)
    if text:
        item.text += text
        item.orig += text
    span = (offset, offset + len(text))
    for prov in prov_list:
        item.prov.append(
            ProvenanceItem(page_no=prov.page_no, bbox=prov.bbox, charspan=span),
        )


class DoclangCategory(str, Enum):
    """DoclangCategory."""

    ROOT = "root"
    SPECIAL = "special"
    METADATA = "metadata"
    GEOMETRIC = "geometric"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    FORMATTING = "formatting"
    GROUPING = "grouping"
    STRUCTURAL = "structural"
    CONTENT = "content"
    CONTINUATION = "continuation"


class DoclangToken(str, Enum):
    """DoclangToken."""

    # Root and metadata
    DOCUMENT = "doclang"
    HEAD = "head"
    LABEL = "label"

    # Special
    PAGE_BREAK = "page_break"

    # Geometric and temporal
    LOCATION = "location"
    LAYER = "layer"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    CENTISECOND = "centisecond"

    # Semantic
    HEADING = "heading"
    TEXT = "text"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PICTURE = "picture"
    FORMULA = "formula"
    CODE = "code"
    LDIV = "ldiv"
    CHECKBOX = "checkbox"
    TABLE = "table"
    FIELD_REGION = "field_region"
    FIELD_ITEM = "field_item"
    FIELD_KEY = "key"
    FIELD_VALUE = "value"
    FIELD_HEADING = "field_heading"
    FIELD_HINT = "hint"

    # Grouping
    LIST = "list"
    GROUP = "group"

    # Formatting
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"
    HANDWRITING = "handwriting"

    # Formatting self-closing
    RTL = "rtl"
    BR = "br"

    # Structural
    # -- Tables
    FCEL = "fcel"
    ECEL = "ecel"
    CHED = "ched"
    RHED = "rhed"
    CORN = "corn"
    SROW = "srow"
    LCEL = "lcel"
    UCEL = "ucel"
    XCEL = "xcel"
    NL = "nl"

    # Continuation
    THREAD = "thread"
    HREF = "href"
    XREF = "xref"

    # Binary data / content helpers
    SRC = "src"
    CUSTOM = "custom"
    INDEX = "index"
    MARKER = "marker"
    CONTENT = "content"  # TODO: review element name


class DoclangAttributeKey(str, Enum):
    """Attribute keys allowed on Doclang tokens."""

    XMLNS = "xmlns"
    VERSION = "version"
    VALUE = "value"
    RESOLUTION = "resolution"
    LEVEL = "level"
    ORDERED = "ordered"
    TYPE = "type"
    CLASS = "class"
    THREAD_ID = "thread_id"
    URI = "uri"


class DoclangAttributeValue(str, Enum):
    """Enumerated values for specific Doclang attributes."""

    # Generic boolean-like values
    TRUE = "true"
    FALSE = "false"

    # List classes
    ORDERED = "ordered"
    UNORDERED = "unordered"

    # Checkbox classes
    SELECTED = "selected"
    UNSELECTED = "unselected"

    # Inline class values
    FORMULA = "formula"
    CODE = "code"
    PICTURE = "picture"


class DoclangVocabulary(BaseModel):
    """DoclangVocabulary."""

    # Allowed attributes per token (defined outside the Enum to satisfy mypy)
    ALLOWED_ATTRIBUTES: ClassVar[dict[DoclangToken, set["DoclangAttributeKey"]]] = {
        DoclangToken.DOCUMENT: {
            DoclangAttributeKey.XMLNS,
            DoclangAttributeKey.VERSION,
        },
        DoclangToken.LOCATION: {
            DoclangAttributeKey.VALUE,
            DoclangAttributeKey.RESOLUTION,
        },
        DoclangToken.LAYER: {DoclangAttributeKey.VALUE},
        DoclangToken.LABEL: {DoclangAttributeKey.VALUE},
        DoclangToken.SRC: {DoclangAttributeKey.URI},
        DoclangToken.HREF: {DoclangAttributeKey.URI},
        DoclangToken.HOUR: {DoclangAttributeKey.VALUE},
        DoclangToken.MINUTE: {DoclangAttributeKey.VALUE},
        DoclangToken.SECOND: {DoclangAttributeKey.VALUE},
        DoclangToken.CENTISECOND: {DoclangAttributeKey.VALUE},
        DoclangToken.HEADING: {DoclangAttributeKey.LEVEL},
        DoclangToken.FIELD_HEADING: {DoclangAttributeKey.LEVEL},
        DoclangToken.CHECKBOX: {DoclangAttributeKey.CLASS},
        DoclangToken.LIST: {DoclangAttributeKey.CLASS},
        DoclangToken.THREAD: {DoclangAttributeKey.THREAD_ID},
        DoclangToken.XREF: {DoclangAttributeKey.THREAD_ID},
    }

    # Allowed values for specific attributes (enumerations)
    # Structure: token -> attribute name -> set of allowed string values
    ALLOWED_ATTRIBUTE_VALUES: ClassVar[
        dict[
            DoclangToken,
            dict["DoclangAttributeKey", set["DoclangAttributeValue"]],
        ]
    ] = {
        # Grouping and inline enumerations
        DoclangToken.LIST: {
            DoclangAttributeKey.CLASS: {
                DoclangAttributeValue.ORDERED,
                DoclangAttributeValue.UNORDERED,
            }
        },
        DoclangToken.CHECKBOX: {
            DoclangAttributeKey.CLASS: {
                DoclangAttributeValue.SELECTED,
                DoclangAttributeValue.UNSELECTED,
            }
        },
        # Other attributes (e.g., level, type, thread_id) are not enumerated here
    }

    ALLOWED_ATTRIBUTE_RANGE: ClassVar[dict[DoclangToken, dict["DoclangAttributeKey", tuple[int, int]]]] = {
        # Geometric: value in [0, res]; resolution optional.
        # Keep conservative defaults aligned with existing usage.
        DoclangToken.LOCATION: {
            DoclangAttributeKey.VALUE: (0, DOCLANG_DFLT_RESOLUTION),  # TODO: review
            DoclangAttributeKey.RESOLUTION: (
                DOCLANG_DFLT_RESOLUTION,
                DOCLANG_DFLT_RESOLUTION,
            ),  # TODO: review
        },
        # Temporal components
        DoclangToken.HOUR: {DoclangAttributeKey.VALUE: (0, 99)},
        DoclangToken.MINUTE: {DoclangAttributeKey.VALUE: (0, 59)},
        DoclangToken.SECOND: {DoclangAttributeKey.VALUE: (0, 59)},
        DoclangToken.CENTISECOND: {DoclangAttributeKey.VALUE: (0, 99)},
        # Levels (N ≥ 1)
        DoclangToken.HEADING: {DoclangAttributeKey.LEVEL: (1, 6)},
        DoclangToken.FIELD_HEADING: {DoclangAttributeKey.LEVEL: (1, 6)},
        # Continuation markers (thread_id length constraints)
        DoclangToken.THREAD: {DoclangAttributeKey.THREAD_ID: (1, 10)},
        DoclangToken.XREF: {DoclangAttributeKey.THREAD_ID: (1, 10)},
    }

    # Self-closing tokens set
    IS_SELFCLOSING: ClassVar[set[DoclangToken]] = {
        DoclangToken.PAGE_BREAK,
        DoclangToken.LOCATION,
        DoclangToken.LAYER,
        DoclangToken.LABEL,
        DoclangToken.SRC,
        DoclangToken.HREF,
        DoclangToken.HOUR,
        DoclangToken.MINUTE,
        DoclangToken.SECOND,
        DoclangToken.CENTISECOND,
        DoclangToken.BR,
        DoclangToken.CHECKBOX,
        DoclangToken.LDIV,
        # OTSL structural tokens are emitted as self-closing markers
        DoclangToken.FCEL,
        DoclangToken.ECEL,
        DoclangToken.CHED,
        DoclangToken.RHED,
        DoclangToken.CORN,
        DoclangToken.SROW,
        DoclangToken.LCEL,
        DoclangToken.UCEL,
        DoclangToken.XCEL,
        DoclangToken.NL,
        # Continuation markers
        DoclangToken.THREAD,
    }

    # Token to category mapping
    TOKEN_CATEGORIES: ClassVar[dict[DoclangToken, DoclangCategory]] = {
        # Root
        DoclangToken.DOCUMENT: DoclangCategory.ROOT,
        # Metadata
        DoclangToken.HEAD: DoclangCategory.METADATA,
        # Special
        DoclangToken.PAGE_BREAK: DoclangCategory.SPECIAL,
        # Geometric
        DoclangToken.LOCATION: DoclangCategory.GEOMETRIC,
        DoclangToken.LAYER: DoclangCategory.GEOMETRIC,
        # Temporal
        DoclangToken.HOUR: DoclangCategory.TEMPORAL,
        DoclangToken.MINUTE: DoclangCategory.TEMPORAL,
        DoclangToken.SECOND: DoclangCategory.TEMPORAL,
        DoclangToken.CENTISECOND: DoclangCategory.TEMPORAL,
        # Semantic
        DoclangToken.HEADING: DoclangCategory.SEMANTIC,
        DoclangToken.TEXT: DoclangCategory.SEMANTIC,
        DoclangToken.CAPTION: DoclangCategory.SEMANTIC,
        DoclangToken.FOOTNOTE: DoclangCategory.SEMANTIC,
        DoclangToken.PAGE_HEADER: DoclangCategory.SEMANTIC,
        DoclangToken.PAGE_FOOTER: DoclangCategory.SEMANTIC,
        DoclangToken.PICTURE: DoclangCategory.SEMANTIC,
        DoclangToken.FIELD_REGION: DoclangCategory.SEMANTIC,
        DoclangToken.FIELD_ITEM: DoclangCategory.SEMANTIC,
        DoclangToken.FIELD_HEADING: DoclangCategory.SEMANTIC,
        DoclangToken.FIELD_HINT: DoclangCategory.SEMANTIC,
        DoclangToken.FORMULA: DoclangCategory.SEMANTIC,
        DoclangToken.CODE: DoclangCategory.SEMANTIC,
        DoclangToken.CHECKBOX: DoclangCategory.SEMANTIC,
        DoclangToken.LIST: DoclangCategory.SEMANTIC,
        DoclangToken.TABLE: DoclangCategory.SEMANTIC,
        # Grouping
        DoclangToken.GROUP: DoclangCategory.GROUPING,
        # Formatting
        DoclangToken.BOLD: DoclangCategory.FORMATTING,
        DoclangToken.ITALIC: DoclangCategory.FORMATTING,
        DoclangToken.STRIKETHROUGH: DoclangCategory.FORMATTING,
        DoclangToken.SUPERSCRIPT: DoclangCategory.FORMATTING,
        DoclangToken.SUBSCRIPT: DoclangCategory.FORMATTING,
        DoclangToken.HANDWRITING: DoclangCategory.FORMATTING,
        DoclangToken.RTL: DoclangCategory.FORMATTING,
        DoclangToken.BR: DoclangCategory.FORMATTING,
        # Structural
        DoclangToken.LDIV: DoclangCategory.STRUCTURAL,
        DoclangToken.FCEL: DoclangCategory.STRUCTURAL,
        DoclangToken.ECEL: DoclangCategory.STRUCTURAL,
        DoclangToken.CHED: DoclangCategory.STRUCTURAL,
        DoclangToken.RHED: DoclangCategory.STRUCTURAL,
        DoclangToken.CORN: DoclangCategory.STRUCTURAL,
        DoclangToken.SROW: DoclangCategory.STRUCTURAL,
        DoclangToken.LCEL: DoclangCategory.STRUCTURAL,
        DoclangToken.UCEL: DoclangCategory.STRUCTURAL,
        DoclangToken.XCEL: DoclangCategory.STRUCTURAL,
        DoclangToken.NL: DoclangCategory.STRUCTURAL,
        DoclangToken.FIELD_KEY: DoclangCategory.STRUCTURAL,
        DoclangToken.FIELD_VALUE: DoclangCategory.STRUCTURAL,
        # Continuation
        DoclangToken.THREAD: DoclangCategory.CONTINUATION,
        # Content/Binary data
        DoclangToken.HREF: DoclangCategory.CONTENT,
        DoclangToken.XREF: DoclangCategory.CONTENT,
        DoclangToken.MARKER: DoclangCategory.CONTENT,
        DoclangToken.CONTENT: DoclangCategory.CONTENT,
    }

    @classmethod
    def _get_category(cls, token: DoclangToken) -> DoclangCategory:
        """Get the category for a given Doclang token.

        Args:
            token: The Doclang token to look up.

        Returns:
            The corresponding DoclangCategory for the token.

        Raises:
            ValueError: If the token is not found in the mapping.
        """
        if token not in cls.TOKEN_CATEGORIES:
            raise ValueError(f"Token '{token}' has no defined category")
        return cls.TOKEN_CATEGORIES[token]

    @classmethod
    def _create_closing_token(cls, *, token: str) -> str:
        r"""Create a closing tag from an opening tag string.

        Example: "<heading level=\"2\">" -> "</heading>"
        Validates the tag and ensures it is not self-closing.
        If `token` is already a valid closing tag, it is returned unchanged.
        """
        if not isinstance(token, str) or not token.strip():
            raise ValueError("token must be a non-empty string")

        s = token.strip()

        # Already a closing tag: validate and return as-is
        if s.startswith("</"):
            m_close = re.match(r"^</\s*([a-zA-Z_][\w\-]*)\s*>$", s)
            if not m_close:
                raise ValueError("invalid closing tag format")
            name = m_close.group(1)
            try:
                DoclangToken(name)
            except ValueError:
                raise ValueError(f"unknown token '{name}'")
            return s

        # Extract tag name from an opening tag while dropping attributes
        m = re.match(r"^<\s*([a-zA-Z_][\w\-]*)\b[^>]*?(/?)\s*>$", s)
        if not m:
            raise ValueError("invalid opening tag format")

        name, trailing_slash = m.group(1), m.group(2)

        # Validate the tag name against known tokens
        try:
            tok_enum = DoclangToken(name)
        except ValueError:
            raise ValueError(f"unknown token '{name}'")

        # Disallow explicit self-closing markup or inherently self-closing tokens
        if trailing_slash == "/":
            raise ValueError(f"token '{name}' is self-closing; no closing tag")
        if tok_enum in cls.IS_SELFCLOSING:
            raise ValueError(f"token '{name}' is self-closing; no closing tag")

        return f"</{name}>"

    @classmethod
    def _create_doclang_root(
        cls,
        *,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
        closing: bool = False,
    ) -> str:
        """Create the document root tag.

        - When `closing` is True, returns the closing root tag.
        """
        if closing:
            return f"</{DoclangToken.DOCUMENT.value}>"
        else:
            parts = [DoclangToken.DOCUMENT.value]
            if namespace is not None:
                parts.append(f'{DoclangAttributeKey.XMLNS.value}="{namespace}"')
            if version is not None:
                parts.append(f'{DoclangAttributeKey.VERSION.value}="{version}"')
            return f"<{' '.join(parts)}>"

    @classmethod
    def _create_threading_token(cls, *, thread_id: str) -> str:
        """Create a vertical continuation threading token.

        Emits `<thread thread_id="..."/>`. Validates required attributes
        against the class schema and basic value sanity.
        """
        token = DoclangToken.THREAD
        assert DoclangAttributeKey.THREAD_ID in cls.ALLOWED_ATTRIBUTES.get(token, set())

        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][DoclangAttributeKey.THREAD_ID]
        length = len(thread_id)
        if not (lo <= length <= hi):
            raise ValueError(f"thread_id length must be in [{lo}, {hi}]")

        return f'<{token.value} {DoclangAttributeKey.THREAD_ID.value}="{thread_id}"/>'

    @classmethod
    def _create_group_token(cls, *, closing: bool = False) -> str:
        """Create a group tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag without attributes.
        """
        if closing:
            return f"</{DoclangToken.GROUP.value}>"
        else:
            return f"<{DoclangToken.GROUP.value}>"

    @classmethod
    def _create_list_token(cls, *, ordered: bool, closing: bool = False) -> str:
        """Create a list tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag with an `ordered` boolean attribute.
        """
        if closing:
            return f"</{DoclangToken.LIST.value}>"
        elif ordered:
            return (
                f'<{DoclangToken.LIST.value} {DoclangAttributeKey.CLASS.value}="{DoclangAttributeValue.ORDERED.value}">'
            )
        else:
            return f"<{DoclangToken.LIST.value}>"

    @classmethod
    def _create_level_open_token(cls, *, token: DoclangToken, level: int) -> str:
        """Create an opening tag; level 1 omits the ``level`` attribute."""
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][DoclangAttributeKey.LEVEL]
        if not (lo <= level <= hi):
            raise ValueError(f"level must be in [{lo}, {hi}]")
        if level == 1:
            return f"<{token.value}>"
        return f'<{token.value} {DoclangAttributeKey.LEVEL.value}="{level}">'

    @classmethod
    def _create_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a heading tag with validated level.

        Level 1 is emitted as bare ``<heading>``; levels 2-6 use ``level="N"``.
        """
        if closing:
            return f"</{DoclangToken.HEADING.value}>"
        return cls._create_level_open_token(token=DoclangToken.HEADING, level=level)

    @classmethod
    def _create_field_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a field-heading tag with validated level.

        Level 1 is emitted as bare ``<field_heading>``; levels 2-6 use ``level="N"``.
        """
        if closing:
            return f"</{DoclangToken.FIELD_HEADING.value}>"
        return cls._create_level_open_token(token=DoclangToken.FIELD_HEADING, level=level)

    @classmethod
    def _create_location_token(cls, *, value: int, resolution: int) -> str:
        """Create a location token with value and resolution.

        Validates both attributes using the configured ranges and ensures
        `value` lies within [0, resolution]. Always emits the resolution
        attribute for explicitness.
        """
        if not (0 <= value < resolution):
            raise ValueError(f"value ({value}) must be in [0, {resolution})")

        return f'<{DoclangToken.LOCATION.value} {DoclangAttributeKey.VALUE.value}="{value}"/>'

    @classmethod
    def get_special_tokens(
        cls,
        *,
        include_location_tokens: bool = True,
        include_temporal_tokens: bool = True,
    ) -> list[str]:
        """Return all Doclang special tokens.

        Rules:
        - If a token has attributes, do not emit a bare opening tag without attributes.
        - Respect `include_location_tokens` and `include_temporal_tokens` to limit
          generation of location and time-related tokens.
        - Emit self-closing tokens as `<name/>` when they have no attributes.
        - Emit non-self-closing tokens as paired `<name>` and `</name>` when they
          have no attributes.
        """
        special_tokens: list[str] = []

        temporal_tokens = {
            DoclangToken.HOUR,
            DoclangToken.MINUTE,
            DoclangToken.SECOND,
            DoclangToken.CENTISECOND,
        }

        for token in DoclangToken:
            # Optional gating for location/temporal tokens
            if not include_location_tokens and token is DoclangToken.LOCATION:
                continue
            if not include_temporal_tokens and token in temporal_tokens:
                continue

            name = token.value
            is_selfclosing = token in cls.IS_SELFCLOSING

            # Attribute-aware emission
            attrs = cls.ALLOWED_ATTRIBUTES.get(token, set())
            if attrs:
                if token is DoclangToken.LIST:
                    special_tokens.append(f"<{name}>")
                    special_tokens.append(f"</{name}>")
                    special_tokens.append(
                        f'<{name} {DoclangAttributeKey.CLASS.value}="{DoclangAttributeValue.ORDERED.value}">'
                    )
                    special_tokens.append(f"</{name}>")
                    continue
                if token in {DoclangToken.HEADING, DoclangToken.FIELD_HEADING}:
                    level_attr = DoclangAttributeKey.LEVEL
                    lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][level_attr]
                    special_tokens.append(f"<{name}>")
                    special_tokens.append(f"</{name}>")
                    for n in range(max(lo + 1, 2), hi + 1):
                        special_tokens.append(f'<{name} {level_attr.value}="{n}">')
                        special_tokens.append(f"</{name}>")
                    continue
                # Enumerated attribute values
                enum_map = cls.ALLOWED_ATTRIBUTE_VALUES.get(token, {})
                for attr_name, allowed_vals in enum_map.items():
                    for v in sorted(allowed_vals, key=lambda x: x.value):
                        if is_selfclosing:
                            special_tokens.append(f'<{name} {attr_name.value}="{v.value}"/>')
                        else:
                            special_tokens.append(f'<{name} {attr_name.value}="{v.value}">')
                            special_tokens.append(f"</{name}>")

                # Ranged attribute values (emit a conservative, complete range)
                range_map = cls.ALLOWED_ATTRIBUTE_RANGE.get(token, {})
                for attr_name, (lo, hi) in range_map.items():
                    # Keep the list size reasonable by skipping optional resolution enumeration
                    if token is DoclangToken.LOCATION and attr_name is DoclangAttributeKey.RESOLUTION:
                        continue
                    for n in range(lo, hi + 1):
                        if is_selfclosing:
                            special_tokens.append(f'<{name} {attr_name.value}="{n}"/>')
                        else:
                            special_tokens.append(f'<{name} {attr_name.value}="{n}">')
                            special_tokens.append(f"</{name}>")
                # Do not emit a bare tag for attribute-bearing tokens
                continue

            # Tokens without attributes
            if is_selfclosing:
                special_tokens.append(f"<{name}/>")
            else:
                special_tokens.append(f"<{name}>")
                special_tokens.append(f"</{name}>")

        return special_tokens

    @classmethod
    def _create_selfclosing_token(
        cls,
        *,
        token: DoclangToken,
        attrs: Optional[dict["DoclangAttributeKey", Any]] = None,
    ) -> str:
        """Create a self-closing token with optional attributes (default None).

        - Validates the token is declared self-closing.
        - Validates provided attributes against ``ALLOWED_ATTRIBUTES`` and
          ``ALLOWED_ATTRIBUTE_VALUES`` or ``ALLOWED_ATTRIBUTE_RANGE`` when present.
        """
        if token not in cls.IS_SELFCLOSING:
            raise ValueError(f"token '{token.value}' is not self-closing")

        # No attributes requested
        if not attrs:
            return f"<{token.value}/>"

        # Validate attribute keys
        allowed_keys = cls.ALLOWED_ATTRIBUTES.get(token, set())
        for k in attrs.keys():
            if k not in allowed_keys:
                raise ValueError(f"attribute '{getattr(k, 'value', str(k))}' not allowed on '{token.value}'")

        # Validate values either via enumerations or numeric ranges
        enum_map = cls.ALLOWED_ATTRIBUTE_VALUES.get(token, {})
        range_map = cls.ALLOWED_ATTRIBUTE_RANGE.get(token, {})

        def _coerce_value(val: Any) -> str:
            # Accept enums or native scalars; stringify for emission
            if isinstance(val, Enum):
                return val.value  # type: ignore[attr-defined]
            return str(val)

        parts: list[str] = []
        for k, v in attrs.items():
            # Enumerated allowed values
            if k in enum_map:
                allowed = enum_map[k]
                # Accept either the enum or its string representation
                v_norm = v.value if isinstance(v, Enum) else str(v)
                allowed_strs = {a.value for a in allowed}
                if v_norm not in allowed_strs:
                    raise ValueError(f"invalid value '{v_norm}' for '{k.value}' on '{token.value}'")
                parts.append(f'{k.value}="{v_norm}"')
                continue

            # Ranged numeric values
            if k in range_map:
                lo, hi = range_map[k]
                try:
                    v_num = int(v)
                except Exception:
                    raise ValueError(f"attribute '{k.value}' on '{token.value}' must be an integer")
                if not (lo <= v_num <= hi):
                    raise ValueError(f"attribute '{k.value}' must be in [{lo}, {hi}] for '{token.value}'")
                parts.append(f'{k.value}="{v_num}"')
                continue

            # Free-form attribute without specific constraints
            parts.append(f'{k.value}="{_coerce_value(v)}"')

        # Assemble tag
        attrs_text = " ".join(parts)
        return f"<{token.value} {attrs_text}/>"

    @classmethod
    def _create_checkbox_token(cls, selected: bool) -> str:
        """Create a checkbox token."""
        return cls._create_selfclosing_token(
            token=DoclangToken.CHECKBOX,
            attrs={
                DoclangAttributeKey.CLASS: (
                    DoclangAttributeValue.SELECTED if selected else DoclangAttributeValue.UNSELECTED
                ),
            },
        )


class DoclangSerializationMode(str, Enum):
    """Serialization mode for Doclang output."""

    HUMAN_FRIENDLY = "human_friendly"
    LLM_FRIENDLY = "llm_friendly"


class EscapeMode(str, Enum):
    """XML escape mode for Doclang output."""

    CDATA_ALWAYS = "cdata_always"  # wrap all text in CDATA
    CDATA_WHEN_NEEDED = "cdata_when_needed"  # wrap text in CDATA only if it contains special characters


class WrapMode(str, Enum):
    """Wrap mode for Doclang output."""

    WRAP_ALWAYS = "wrap_always"  # wrap all text in explicit wrapper element
    WRAP_WHEN_NEEDED = "wrap_when_needed"  # wrap text if it has leading/trailing whitespace or contains newlines


class LayerMode(str, Enum):
    """Layer mode for Doclang output."""

    ALWAYS = "always"  # always include layer element
    MINIMAL = "minimal"  # include layer element only when it differs from default


class LabelMode(str, Enum):
    """Label mode for DocLang output."""

    WHEN_DEFINED = "when_defined"  # emit label when present and not ``undefined``
    ALWAYS = "always"  # always emit label
    NEVER = "never"  # never emit label


class ContentType(str, Enum):
    """Content type for Doclang output."""

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


class DoclangParams(CommonParams):
    """Doclang-specific serialization parameters independent of Doclang."""

    # Override parent's layers to default to all ContentLayers
    layers: set[ContentLayer] = set(ContentLayer)

    # Geometry & content controls (aligned with Doclang defaults)
    xsize: int = DOCLANG_DFLT_RESOLUTION
    ysize: int = DOCLANG_DFLT_RESOLUTION
    add_location: bool = True
    add_table_cell_location: bool = False

    add_referenced_caption: bool = True
    add_referenced_footnote: bool = True

    add_page_break: bool = True

    add_content: bool = True

    # types of content to serialize (only relevant if show_content is True):
    content_types: set[ContentType] = _DEFAULT_CONTENT_TYPES

    # Layer mode
    layer_mode: LayerMode = LayerMode.MINIMAL

    # Doclang formatting
    do_self_closing: bool = True
    pretty_indentation: Optional[str] = 2 * " "  # None means minimized serialization, "" means no indentation

    preserve_empty_non_selfclosing: bool = True
    # When True, text items that produce no content (no text, no location) are
    # completely omitted rather than emitting an empty open/close tag pair.
    suppress_empty_elements: bool = False
    # XML compliance: escape special characters in text content
    escape_mode: EscapeMode = EscapeMode.CDATA_WHEN_NEEDED
    content_wrapping_mode: WrapMode = WrapMode.WRAP_WHEN_NEEDED
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
    include_namespace: bool = False
    include_version: bool = False
    # when True, the <text> wrapper is omitted whenever allowed
    use_virtual_text: bool = True
    label_mode: LabelMode = LabelMode.WHEN_DEFINED
    # When False, ``CodeLanguageLabel.UNKNOWN`` maps to ``undefined``; when True, to ``other``.
    interpret_code_unknown_as_other: bool = False


def _create_layer_token(
    *,
    item: DocItem,
    params: DoclangParams,
) -> str:
    """Create `<layer value="..."/>` in element head."""
    if params.layer_mode == LayerMode.ALWAYS or (
        params.layer_mode == LayerMode.MINIMAL and item.content_layer != ContentLayer.BODY
    ):
        return DoclangVocabulary._create_selfclosing_token(
            token=DoclangToken.LAYER,
            attrs={DoclangAttributeKey.VALUE: item.content_layer.value},
        )
    return ""


def _create_label_token(*, value: str) -> str:
    """Emit `<label value="..."/>` for element head (e.g. code language)."""
    safe = value.replace("&", "&amp;").replace('"', "&quot;")
    return DoclangVocabulary._create_selfclosing_token(
        token=DoclangToken.LABEL,
        attrs={DoclangAttributeKey.VALUE: safe},
    )


def _create_src_token(*, uri: str) -> str:
    """Emit `<src uri="..."/>` for picture body (v0.5)."""
    safe = uri.replace("&", "&amp;").replace('"', "&quot;")
    return DoclangVocabulary._create_selfclosing_token(
        token=DoclangToken.SRC,
        attrs={DoclangAttributeKey.URI: safe},
    )


def _create_href_token(*, uri: str) -> str:
    """Emit `<href uri="..."/>` in element head."""
    safe = uri.replace("&", "&amp;").replace('"', "&quot;")
    return DoclangVocabulary._create_selfclosing_token(
        token=DoclangToken.HREF,
        attrs={DoclangAttributeKey.URI: safe},
    )


def _text_item_hyperlink_uri(item: DocItem) -> Optional[str]:
    if isinstance(item, TextItem) and item.hyperlink is not None:
        return str(item.hyperlink)
    return None


def _element_head_prefix(
    *,
    item: DocItem,
    doc: DoclingDocument,
    params: DoclangParams,
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
        parts.append(DoclangVocabulary._create_threading_token(thread_id=thread_id))
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
    params: DoclangParams,
    **kwargs: Any,
) -> str:
    """Serialize referenced caption(s) for inclusion in the host element head."""
    if not params.add_referenced_caption or not item.captions:
        return ""
    cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
    return cap_res.text or ""


_DOCLANG_LABEL_UNDEFINED = "undefined"
_DOCLANG_LABEL_OTHER = "other"


def _element_label_for_serialization(
    *,
    raw_label: Optional[str],
    params: DoclangParams,
) -> Optional[str]:
    """Resolve element-head ``<label>`` emission per ``params.label_mode``."""
    if params.label_mode == LabelMode.NEVER:
        return None
    if params.label_mode == LabelMode.ALWAYS:
        return raw_label if raw_label is not None else _DOCLANG_LABEL_UNDEFINED
    # WHEN_DEFINED: emit only when a label is present and not ``undefined``.
    if raw_label is None or raw_label == _DOCLANG_LABEL_UNDEFINED:
        return None
    return raw_label


def _picture_classification_label_to_doclang(class_name: str) -> str:
    """Map Docling picture classification label to DocLang recommended label."""
    if class_name == PictureClassificationLabel.OTHER.value:
        return _DOCLANG_LABEL_OTHER
    return class_name


def _picture_classification_label_from_doclang(label_val: str) -> Optional[str]:
    """Map DocLang picture label to Docling picture classification label."""
    if label_val == _DOCLANG_LABEL_UNDEFINED:
        return None
    if label_val == _DOCLANG_LABEL_OTHER:
        return PictureClassificationLabel.OTHER.value
    return label_val


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
    params: DoclangParams,
    **kwargs: Any,
) -> str:
    """Serialize item meta as ``<custom>`` for element head (v0.5)."""
    if not isinstance(item, DocItem) or not item.meta:
        return ""
    meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
    return meta_res.text or ""


def _get_delim(*, params: DoclangParams) -> str:
    """Return record delimiter based on DoclangSerializationMode."""
    return "" if params.pretty_indentation is None else "\n"


def _escape_text(text: str, params: DoclangParams) -> str:
    do_wrap = params.content_wrapping_mode == WrapMode.WRAP_ALWAYS or (
        params.content_wrapping_mode == WrapMode.WRAP_WHEN_NEEDED and (text != text.strip() or "\n" in text)
    )
    if params.escape_mode == EscapeMode.CDATA_ALWAYS or (
        params.escape_mode == EscapeMode.CDATA_WHEN_NEEDED and any(c in text for c in ['"', "'", "&", "<", ">"])
    ):
        text = f"<![CDATA[{text}]]>"
    if do_wrap:
        # text = f'<{el_str} xml:space="preserve">{text}</{el_str}>'
        text = _wrap(text=text, wrap_tag=DoclangToken.CONTENT.value)
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


# Tokens allowed in an element head (before body content). Mirrors
# ``_element_head_prefix`` serialization order plus continuation/temporal tokens.
_ELEMENT_HEAD_TAGS: Final[frozenset[str]] = frozenset(
    {
        DoclangToken.LABEL.value,
        DoclangToken.LAYER.value,
        DoclangToken.HREF.value,
        DoclangToken.LOCATION.value,
        DoclangToken.CAPTION.value,
        DoclangToken.CUSTOM.value,
        DoclangToken.THREAD.value,
        DoclangToken.XREF.value,
        DoclangToken.HOUR.value,
        DoclangToken.MINUTE.value,
        DoclangToken.SECOND.value,
        DoclangToken.CENTISECOND.value,
    }
)


class DoclangListSerializer(BaseModel, BaseListSerializer):
    """Doclang-specific list serializer."""

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
        """Serialize a ``ListGroup`` into Doclang markup.

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
            **kwargs: Additional serializer parameters forwarded to ``DoclangParams``.

        Returns:
            A ``SerializationResult`` containing serialized text and metadata.
        """
        my_visited = visited if visited is not None else set()
        params = DoclangParams(**kwargs)

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
        list_close = f"</{DoclangToken.LIST.value}>"
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
                DoclangVocabulary._create_list_token(ordered=True)
                if ordered
                else DoclangVocabulary._create_list_token(ordered=False)
            )
            text_res = _wrap_token(text=text_res, open_token=open_token)
            return create_ser_result(text=text_res, span_source=item_results)

        thread_id = _allocate_thread_id(doc_serializer, item)
        out_parts: list[str] = []
        current_block: list[str] = []
        current_page: Optional[int] = None
        for text, page_no in child_segments:
            if current_block and page_no is not None and current_page is not None and page_no != current_page:
                list_open = DoclangVocabulary._create_list_token(
                    ordered=ordered
                ) + DoclangVocabulary._create_threading_token(thread_id=thread_id)
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
            list_open = DoclangVocabulary._create_list_token(
                ordered=ordered
            ) + DoclangVocabulary._create_threading_token(thread_id=thread_id)
            block_text = delim.join(current_block)
            out_parts.append(f"{list_open}{block_text}{delim}{list_close}")

        return create_ser_result(text="".join(out_parts), span_source=item_results)


# Linguist v9.5.0 language keys for DocLang code labels:
# https://github.com/doclang-project/doclang/blob/v0.4.0/spec.md#appendix-b-recommendations
# https://github.com/github-linguist/linguist/blob/v9.5.0/lib/linguist/languages.yml
_CODE_LANGUAGE_TO_LINGUIST: Final[dict[CodeLanguageLabel, str]] = {
    CodeLanguageLabel.ADA: "Ada",
    CodeLanguageLabel.AWK: "Awk",
    CodeLanguageLabel.BASH: "Shell",
    CodeLanguageLabel.C: "C",
    CodeLanguageLabel.C_SHARP: "C#",
    CodeLanguageLabel.C_PLUS_PLUS: "C++",
    CodeLanguageLabel.CMAKE: "CMake",
    CodeLanguageLabel.COBOL: "COBOL",
    CodeLanguageLabel.CSS: "CSS",
    CodeLanguageLabel.CEYLON: "Ceylon",
    CodeLanguageLabel.CLOJURE: "Clojure",
    CodeLanguageLabel.CRYSTAL: "Crystal",
    CodeLanguageLabel.CUDA: "Cuda",
    CodeLanguageLabel.CYTHON: "Cython",
    CodeLanguageLabel.D: "D",
    CodeLanguageLabel.DART: "Dart",
    CodeLanguageLabel.DOCKERFILE: "Dockerfile",
    CodeLanguageLabel.DOCLANG: "XML",
    CodeLanguageLabel.ELIXIR: "Elixir",
    CodeLanguageLabel.ERLANG: "Erlang",
    CodeLanguageLabel.FORTRAN: "Fortran",
    CodeLanguageLabel.FORTH: "Forth",
    CodeLanguageLabel.GO: "Go",
    CodeLanguageLabel.HTML: "HTML",
    CodeLanguageLabel.HASKELL: "Haskell",
    CodeLanguageLabel.HAXE: "Haxe",
    CodeLanguageLabel.JAVA: "Java",
    CodeLanguageLabel.JAVASCRIPT: "JavaScript",
    CodeLanguageLabel.JSON: "JSON",
    CodeLanguageLabel.JULIA: "Julia",
    CodeLanguageLabel.KOTLIN: "Kotlin",
    CodeLanguageLabel.LATEX: "TeX",
    CodeLanguageLabel.LISP: "Common Lisp",
    CodeLanguageLabel.LUA: "Lua",
    CodeLanguageLabel.MATLAB: "MATLAB",
    CodeLanguageLabel.MOONSCRIPT: "MoonScript",
    CodeLanguageLabel.NIM: "Nim",
    CodeLanguageLabel.OCAML: "OCaml",
    CodeLanguageLabel.OBJECTIVEC: "Objective-C",
    CodeLanguageLabel.OCTAVE: "MATLAB",
    CodeLanguageLabel.PHP: "PHP",
    CodeLanguageLabel.PASCAL: "Pascal",
    CodeLanguageLabel.PERL: "Perl",
    CodeLanguageLabel.PROLOG: "Prolog",
    CodeLanguageLabel.PYTHON: "Python",
    CodeLanguageLabel.RACKET: "Racket",
    CodeLanguageLabel.RUBY: "Ruby",
    CodeLanguageLabel.RUST: "Rust",
    CodeLanguageLabel.SML: "Standard ML",
    CodeLanguageLabel.SQL: "SQL",
    CodeLanguageLabel.SCALA: "Scala",
    CodeLanguageLabel.SCHEME: "Scheme",
    CodeLanguageLabel.SWIFT: "Swift",
    CodeLanguageLabel.TYPESCRIPT: "TypeScript",
    CodeLanguageLabel.VISUALBASIC: "Visual Basic .NET",
    CodeLanguageLabel.XML: "XML",
    CodeLanguageLabel.YAML: "YAML",
}
# Docling labels without a Linguist key (BC, DC, TIKZ) map to ``other``.
# OCTAVE/DOCLANG share a Linguist key with MATLAB/XML; reverse mapping prefers the latter.
_LINGUIST_TO_CODE_LANGUAGE: Final[dict[str, CodeLanguageLabel]] = {
    linguist_key: docling_lang
    for docling_lang, linguist_key in _CODE_LANGUAGE_TO_LINGUIST.items()
    if docling_lang not in {CodeLanguageLabel.OCTAVE, CodeLanguageLabel.DOCLANG}
}


def _code_language_label_to_doclang(
    lang: CodeLanguageLabel,
    *,
    interpret_unknown_as_other: bool,
) -> str:
    """Map Docling code language label to DocLang recommended label."""
    if lang == CodeLanguageLabel.UNKNOWN:
        return _DOCLANG_LABEL_OTHER if interpret_unknown_as_other else _DOCLANG_LABEL_UNDEFINED
    if linguist_key := _CODE_LANGUAGE_TO_LINGUIST.get(lang):
        return linguist_key
    return _DOCLANG_LABEL_OTHER


def _code_language_label_from_doclang(label_val: str) -> CodeLanguageLabel:
    """Map DocLang code label to Docling code language label."""
    if label_val in {
        _DOCLANG_LABEL_OTHER,
        _DOCLANG_LABEL_UNDEFINED,
        CodeLanguageLabel.UNKNOWN.value,
    }:
        return CodeLanguageLabel.UNKNOWN
    return _LINGUIST_TO_CODE_LANGUAGE.get(label_val, CodeLanguageLabel.UNKNOWN)


class DoclangTextSerializer(BaseModel, BaseTextSerializer):
    """Doclang-specific text item serializer using `<location>` tokens."""

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
        """Serialize a text item to Doclang format.

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
    ) -> tuple[Optional[str], Optional[DoclangToken]]:
        """Determine the wrapper token for a ListItem.

        Args:
            item: The ListItem to determine wrapper for.
            doc: The document containing the item.
            use_virtual_text: If True, omit ``<text>`` when the ldiv segment contains
                only that text (DocLang v0.4 virtual text mode).

        Returns:
            Tuple of (wrap_open_token, tok) where wrap_open_token is the opening tag
            string or None, and tok is the DoclangToken or None.
        """
        if item.text:
            if use_virtual_text and not self._list_item_has_segment_siblings(item=item, doc=doc):
                return None, None
            tok = DoclangToken.TEXT
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
                tok = DoclangToken.GROUP
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
        """Serialize a ``TextItem`` into Doclang markup.

        Depending on parameters, emits meta blocks, location tokens, and the
        item's textual content (prefixing code language for ``CodeItem``). For
        floating items, captions may be appended. The result can be wrapped in a
        tag derived from the item's label when applicable.

        Args:
            item: The text-like item to serialize.
            doc_serializer: The document-level serializer for delegating nested items.
            doc: The document used to resolve references and children.
            visited: Set of already visited item refs to avoid cycles.
            **kwargs: Additional serializer parameters forwarded to ``DoclangParams``.

        Returns:
            A ``SerializationResult`` with the serialized text and span source.
        """
        my_visited = visited if visited is not None else set()
        params = DoclangParams(**kwargs)

        # Determine wrapper open-token for this item using Doclang vocabulary.
        # - TitleItem: use <heading level="1"> ... </heading>.
        # - SectionHeaderItem: use <heading level="N+1"> ... </heading> where N is SectionHeaderItem.level.
        # - Other text-like items: map the label to an DoclangToken; for
        #   list items, this maps to <ldiv> and keeps the text serializer
        #   free of type-based special casing.
        wrap_open_token: Optional[str]
        tok: DoclangToken | None = None
        if isinstance(item, TitleItem):
            wrap_open_token = DoclangVocabulary._create_heading_token(level=1)
        elif isinstance(item, SectionHeaderItem):
            wrap_open_token = DoclangVocabulary._create_heading_token(level=item.level + 1)
        elif isinstance(item, ListItem):
            wrap_open_token, tok = self._determine_list_item_wrapper(
                item=item, doc=doc, use_virtual_text=params.use_virtual_text
            )
        elif isinstance(item, CodeItem):
            tok = DoclangToken.CODE
            wrap_open_token = f"<{tok.value}>"
        elif isinstance(item, TextItem) and item.label in [
            DocItemLabel.CHECKBOX_SELECTED,
            DocItemLabel.CHECKBOX_UNSELECTED,
        ]:
            if item.parent and isinstance((parent_item := item.parent.resolve(doc)), TextItem) and not parent_item.text:
                # skip re-wrapping if already in a text item
                wrap_open_token = None
            else:
                tok = DoclangToken.TEXT
                wrap_open_token = f"<{tok.value}>"
        elif isinstance(item, TextItem) and item.label == DocItemLabel.CAPTION:
            # v0.5: <caption> is only valid in a host element head, not top-level.
            tok = DoclangToken.TEXT
            wrap_open_token = f"<{tok.value}>"
        elif isinstance(item, TextItem) and (
            tok := {
                DocItemLabel.FIELD_KEY: DoclangToken.FIELD_KEY,
                DocItemLabel.FIELD_VALUE: DoclangToken.FIELD_VALUE,
                DocItemLabel.FIELD_HEADING: DoclangToken.FIELD_HEADING,
                DocItemLabel.FIELD_HINT: DoclangToken.FIELD_HINT,
                DocItemLabel.MARKER: DoclangToken.MARKER,
            }.get(item.label)
        ):
            wrap_open_token = f"<{tok.value}>"
            if isinstance(item, FieldValueItem) and item.kind != "read_only":
                wrap_open_token = f'<{tok.value} class="{item.kind}">'
            elif isinstance(item, FieldHeadingItem):
                wrap_open_token = DoclangVocabulary._create_field_heading_token(level=item.level)
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
            tok = DoclangToken.TEXT
            wrap_open_token = f"<{tok.value}>"
        else:
            label_value = str(item.label)
            try:
                tok = DoclangToken(label_value)
                wrap_open_token = f"<{tok.value}>"
            except ValueError:
                raise ValueError(f"Unsupported Doclang token for label '{label_value}'")

        parts: list[str] = []

        # For ListItems, emit <ldiv> as a separate delimiter element before content
        ldiv_element = ""
        if isinstance(item, ListItem):
            if item.marker:
                marker_text = _escape_text(item.marker, params)
                marker_element = _wrap(text=marker_text, wrap_tag=DoclangToken.MARKER.value)
                ldiv_element = _wrap(text=marker_element, wrap_tag=DoclangToken.LDIV.value)
            else:
                # Empty ldiv (self-closing)
                ldiv_element = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.LDIV)

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
                parts.append(DoclangVocabulary._create_threading_token(thread_id=thread_id))
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
                    text_part = _wrap(text=text_part, wrap_tag=DoclangToken.HANDWRITING.value)
                elif item.label in [
                    DocItemLabel.CHECKBOX_SELECTED,
                    DocItemLabel.CHECKBOX_UNSELECTED,
                ]:
                    # Add checkbox token before the text
                    checkbox_token = DoclangVocabulary._create_checkbox_token(
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


class DoclangMetaSerializer(BaseModel, BaseMetaSerializer):
    """Doclang-specific meta serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Doclang-specific meta serializer."""
        params = DoclangParams(**kwargs)

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
            texts.insert(0, f"<{DoclangToken.CUSTOM.value}>")
            texts.append(f"</{DoclangToken.CUSTOM.value}>")
        return create_ser_result(
            text=elem_delim.join(texts),
            span_source=item if isinstance(item, DocItem) else [],
        )

    def _serialize_meta_field(self, meta: BaseMeta, name: str, params: DoclangParams) -> Optional[str]:
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


class DoclangPictureSerializer(BasePictureSerializer):
    """Doclang-specific picture item serializer."""

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
        params = DoclangParams(**kwargs)
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
                    params_chart = DoclangParams(**{**params.model_dump(), "add_table_cell_location": False})
                    otsl_content = DoclangTableSerializer()._emit_otsl(
                        item=temp_table,  # type: ignore[arg-type]
                        doc_serializer=doc_serializer,
                        doc=temp_doc,
                        params=params_chart,
                        **kwargs,
                    )
                    body_parts.append(_wrap(text=otsl_content, wrap_tag=DoclangToken.TABLE.value))

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
        picture_open = f"<{DoclangToken.PICTURE.value}"
        if body_parts and any(p.startswith(f"<{DoclangToken.TABLE.value}") for p in body_parts):
            picture_open += f' {DoclangAttributeKey.CLASS.value}="chart"'
        picture_open += ">"
        picture_text = f"{picture_open}{inner}</{DoclangToken.PICTURE.value}>"

        footnote_text = ""
        if params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        if not inner and not footnote_text:
            if params.suppress_empty_elements:
                return create_ser_result()
            text_res = f"<{DoclangToken.PICTURE.value}></{DoclangToken.PICTURE.value}>"
        elif footnote_text:
            text_res = _wrap(text=picture_text + footnote_text, wrap_tag=DoclangToken.GROUP.value)
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


def _thread_table_merge_offset(existing: TableItem, prov: ProvenanceItem) -> tuple[int, int]:
    """Row/column offsets when merging a threaded table fragment into ``existing``."""
    if not existing.prov or not existing.data:
        return 0, 0
    last_page = existing.prov[-1].page_no
    if prov.page_no == last_page:
        return 0, existing.data.num_cols
    return existing.data.num_rows, 0


def _merge_table_data(
    *,
    base: TableData,
    fragment: TableData,
    row_offset: int = 0,
    col_offset: int = 0,
) -> None:
    """Merge ``fragment`` table cells into ``base`` (indices must already be offset)."""
    base.table_cells.extend(fragment.table_cells)
    base.num_rows = max(base.num_rows, row_offset + fragment.num_rows)
    base.num_cols = max(base.num_cols, col_offset + fragment.num_cols)


class DoclangTableSerializer(BaseTableSerializer):
    """Doclang-specific table item serializer."""

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
    ) -> DoclangToken:
        """Pick OTSL origin token for a cell origin inside a table fragment slice."""
        cont_left = col_idx == col_start and col_start > 0
        cont_up = rowstart < row_start and row_idx == row_start
        if cont_left and cont_up:
            return DoclangToken.XCEL
        if cont_up:
            return DoclangToken.UCEL
        if cont_left:
            return DoclangToken.LCEL
        if has_content:
            if cell.column_header and cell.row_header:
                return DoclangToken.CORN
            if cell.column_header:
                return DoclangToken.CHED
            if cell.row_header:
                return DoclangToken.RHED
            if cell.row_section:
                return DoclangToken.SROW
            return DoclangToken.FCEL
        if cell.column_header and cell.row_header:
            return DoclangToken.CORN
        return DoclangToken.ECEL

    def _emit_otsl(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        params: "DoclangParams",
        row_start: int = 0,
        row_end: Optional[int] = None,
        col_start: int = 0,
        col_end: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Emit OTSL payload using Doclang tokens and location semantics.

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
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=origin))
                    if content and origin != DoclangToken.ECEL:
                        if cell_loc:
                            parts.append(cell_loc)
                        if ContentType.TABLE_CELL in params.content_types:
                            if not isinstance(cell, RichTableCell):
                                content = _escape_text(content, params)
                                if not params.use_virtual_text:
                                    content = _wrap(text=content, wrap_tag=DoclangToken.TEXT.value)
                            parts.append(content)
                elif rowstart != i and colstart != j:
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.XCEL))
                elif rowstart != i:
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.UCEL))
                elif colstart != j:
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.LCEL))

            parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.NL))

        return "".join(parts)

    def _serialize_single_table(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        params: DoclangParams,
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

        host_token = DoclangToken.INDEX if item.label == DocItemLabel.DOCUMENT_INDEX else DoclangToken.TABLE
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
            text_res = _wrap(text=table_text + footnote_text, wrap_tag=DoclangToken.GROUP.value)
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
        params = DoclangParams(**kwargs)

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


class DoclangInlineSerializer(BaseInlineSerializer):
    """Inline serializer emitting Doclang `<inline>` and `<location>` tokens."""

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
        """Serialize inline content with optional location into Doclang."""
        my_visited = visited if visited is not None else set()
        params = DoclangParams(**kwargs)
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
                text_res = _wrap(text=text_res, wrap_tag=DoclangToken.TEXT.value)
        return create_ser_result(text=text_res, span_source=parts)


class DoclangFallbackSerializer(BaseFallbackSerializer):
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
        params = DoclangParams(**kwargs)
        delim = _get_delim(params=DoclangParams(**kwargs))
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
            tok = DoclangToken.FIELD_REGION if is_fri else DoclangToken.FIELD_ITEM
            text_res = _wrap(text=text_res, wrap_tag=tok.value)
            if isinstance(item, FieldItem):
                text_res = _wrap_in_field_region_if_needed(text=text_res, item=item, doc=doc)
            return create_ser_result(text=text_res, span_source=parts)
        return create_ser_result()


class DoclangKeyValueSerializer(BaseKeyValueSerializer):
    """No-op serializer for key/value items in Doclang."""

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


class DoclangFormSerializer(BaseFormSerializer):
    """No-op serializer for form items in Doclang."""

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


class DoclangAnnotationSerializer(BaseAnnotationSerializer):
    """No-op annotation serializer; Doclang relies on meta instead."""

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


class DoclangDocSerializer(DocSerializer):
    """Doclang document serializer."""

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

    text_serializer: BaseTextSerializer = DoclangTextSerializer()
    table_serializer: BaseTableSerializer = DoclangTableSerializer()
    picture_serializer: BasePictureSerializer = DoclangPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = DoclangKeyValueSerializer()
    form_serializer: BaseFormSerializer = DoclangFormSerializer()
    fallback_serializer: BaseFallbackSerializer = DoclangFallbackSerializer()

    list_serializer: BaseListSerializer = DoclangListSerializer()
    inline_serializer: BaseInlineSerializer = DoclangInlineSerializer()

    meta_serializer: BaseMetaSerializer = DoclangMetaSerializer()
    annotation_serializer: BaseAnnotationSerializer = DoclangAnnotationSerializer()

    params: DoclangParams = DoclangParams()

    @override
    def _meta_is_wrapped(self) -> bool:
        return True

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's captions with Doclang location tokens."""
        params = DoclangParams(**kwargs)
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
            text_res = _wrap(text=text_res, wrap_tag=DoclangToken.CAPTION.value)
        return create_ser_result(text=text_res, span_source=results)

    @override
    def serialize_footnotes(
        self,
        item: FloatingItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's footnotes with Doclang location tokens."""
        params = DoclangParams(**kwargs)
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
                        text_res = _wrap(text_res, wrap_tag=DoclangToken.FOOTNOTE.value)
                        results.append(create_ser_result(text=text_res))

        text_res = "".join([r.text for r in results])

        return create_ser_result(text=text_res, span_source=results)

    def _create_head(self) -> str:
        """Create the head section of the Doclang document."""
        parts = []
        if self.params.xsize != DOCLANG_DFLT_RESOLUTION or self.params.ysize != DOCLANG_DFLT_RESOLUTION:
            parts.append(f'<default_resolution width="{self.params.xsize}" height="{self.params.ysize}"/>')
        return _wrap(text="".join(parts), wrap_tag=DoclangToken.HEAD.value) if parts else ""

    def _is_content_tag(self, tag: str) -> bool:
        return tag in {DoclangToken.CONTENT.value}

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

        # intercept from DoclangDocSerializer to update param kwargs:
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
        """Doc-level serialization with Doclang root wrapper."""
        # Note: removed internal thread counting; not used.

        delim = _get_delim(params=self.params)

        open_token: str = DoclangVocabulary._create_doclang_root(
            namespace=DOCLANG_NAMESPACE if self.params.include_namespace else None,
            version=_DOCLANG_VERSION if self.params.include_version else None,
        )
        head = self._create_head()
        close_token: str = DoclangVocabulary._create_doclang_root(closing=True)

        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            # Always emit well-formed page breaks using the vocabulary
            page_sep = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.PAGE_BREAK)
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
                non_selfclosing = [tok for tok in DoclangToken if tok not in DoclangVocabulary.IS_SELFCLOSING]

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
        """Apply Doclang-specific bold serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.BOLD.value)

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Apply Doclang-specific italic serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.ITALIC.value)

    @override
    def serialize_underline(self, text: str, **kwargs: Any) -> str:
        """Apply Doclang-specific underline serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.UNDERLINE.value)

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Apply Doclang-specific strikethrough serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.STRIKETHROUGH.value)

    @override
    def serialize_subscript(self, text: str, **kwargs: Any) -> str:
        """Apply Doclang-specific subscript serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.SUBSCRIPT.value)

    @override
    def serialize_superscript(self, text: str, **kwargs: Any) -> str:
        """Apply Doclang-specific superscript serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.SUPERSCRIPT.value)

    def serialize_rtl(self, text: str, **kwargs: Any) -> str:
        """Apply Doclang-specific right-to-left text serialization."""
        return _wrap(text=text, wrap_tag=DoclangToken.RTL.value)

    @override
    def post_process(
        self,
        text: str,
        *,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """Apply Doclang text post-processing including RTL direction."""
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


class DoclangDeserializer(BaseModel):
    """Doclang deserializer."""

    # Internal state used while walking the tree (private instance attributes)
    _page_no: int = PrivateAttr(default=1)
    _default_resolution: int = PrivateAttr(default=DOCLANG_DFLT_RESOLUTION)
    _thread_registry: dict[tuple[str, str], NodeItem] = PrivateAttr(default_factory=dict)

    def _thread_registry_key(self, *, thread_id: str, host: str) -> tuple[str, str]:
        return (thread_id, host)

    def deserialize(
        self,
        *,
        text: str,
        page_no: int = 1,
    ) -> DoclingDocument:
        """Deserialize Doclang XML into a DoclingDocument.

        Args:
            text: Doclang XML string to parse.
            page_no: Starting page number (default 1).

        Returns:
            A populated `DoclingDocument` parsed from the input.
        """
        try:
            root_node = parseString(text).documentElement
        except Exception as e:
            ctx = _xml_error_context(text, e)
            raise ValueError(f"Invalid Doclang XML: {e}\n--- XML context ---\n{ctx}") from e
        if root_node is None:
            raise ValueError("Invalid Doclang XML: missing documentElement")
        root: Element = cast(Element, root_node)
        if root.tagName != DoclangToken.DOCUMENT.value:
            candidates = root.getElementsByTagName(DoclangToken.DOCUMENT.value)
            if candidates:
                root = cast(Element, candidates[0])

        doc = DoclingDocument(name="Document")
        self._page_no = page_no
        self._default_resolution = DOCLANG_DFLT_RESOLUTION
        self._thread_registry = {}
        self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)
        self._parse_document_root(doc=doc, root=root)
        return doc

    def _extract_thread_id_from_nodes(self, nodes: Sequence[Node]) -> Optional[str]:
        """Read ``thread_id`` from a ``<thread/>`` element in a node sequence."""
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.THREAD.value:
                thread_id = node.getAttribute(DoclangAttributeKey.THREAD_ID.value)
                if thread_id:
                    return thread_id
        return None

    def _extract_thread_id(self, el: Element) -> Optional[str]:
        """Read ``thread_id`` from an element head."""
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._extract_thread_id_from_nodes(head_nodes)

    def _register_thread(self, *, thread_id: str, host: str, item: NodeItem) -> None:
        self._thread_registry[self._thread_registry_key(thread_id=thread_id, host=host)] = item

    def _get_thread_item(self, thread_id: str, *, host: str) -> Optional[NodeItem]:
        return self._thread_registry.get(self._thread_registry_key(thread_id=thread_id, host=host))

    def _advance_page_break(self, *, doc: DoclingDocument) -> None:
        self._page_no += 1
        self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)

    def _provenance_from_nodes_with_page_breaks(
        self,
        *,
        doc: DoclingDocument,
        nodes: Sequence[Node],
    ) -> list[ProvenanceItem]:
        """Collect provenance quartets, advancing ``_page_no`` at ``<page_break/>``."""
        provs: list[ProvenanceItem] = []
        batch: list[Node] = []
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.PAGE_BREAK.value:
                provs.extend(self._provenance_from_location_nodes(doc=doc, nodes=batch))
                batch = []
                self._advance_page_break(doc=doc)
            elif isinstance(node, Element) and node.tagName == DoclangToken.LOCATION.value:
                batch.append(node)
        provs.extend(self._provenance_from_location_nodes(doc=doc, nodes=batch))
        return provs

    def _virtual_text_from_nodes_with_page_breaks(self, nodes: Sequence[Node]) -> str:
        """Extract virtual-text payload, ignoring element-head tokens and page breaks."""
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.PAGE_BREAK.value:
                continue
            if isinstance(node, Element) and self._is_element_head_tag(node):
                continue
            if isinstance(node, Text):
                if not node.data.strip():
                    continue
                parts.append(node.data)
            elif isinstance(node, Element) and node.tagName == DoclangToken.CONTENT.value:
                parts.append(self._get_text(node))
        return "".join(parts)

    def _merge_threaded_text_item(
        self,
        *,
        text: str,
        prov_list: list[ProvenanceItem],
        existing: TextItem,
    ) -> TextItem:
        _append_textual_fragment(existing, text=text, prov_list=prov_list)
        return existing

    def _apply_initial_text_provenance(
        self,
        item: TextItem,
        *,
        text: str,
        prov_list: list[ProvenanceItem],
    ) -> None:
        if not prov_list:
            return
        item.prov = _provenance_with_charspan(prov_list[:1], (0, len(text)))
        for prov in prov_list[1:]:
            item.prov.append(prov)

    # ------------- Core walkers -------------
    def _parse_document_root(self, *, doc: DoclingDocument, root: Element) -> None:
        for node in root.childNodes:
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=None)

    def _dispatch_element(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        name = el.tagName
        if name in {
            DoclangToken.TEXT.value,
            DoclangToken.CAPTION.value,
            DoclangToken.FOOTNOTE.value,
            DoclangToken.PAGE_HEADER.value,
            DoclangToken.PAGE_FOOTER.value,
            DoclangToken.CODE.value,
            DoclangToken.FORMULA.value,
            DoclangToken.LDIV.value,
            DoclangToken.BOLD.value,
            DoclangToken.ITALIC.value,
            DoclangToken.UNDERLINE.value,
            DoclangToken.STRIKETHROUGH.value,
            DoclangToken.SUBSCRIPT.value,
            DoclangToken.SUPERSCRIPT.value,
            DoclangToken.CONTENT.value,
        }:
            self._parse_text_like(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.PAGE_BREAK.value:
            # Start a new page; keep a default square page using the configured resolution
            self._page_no += 1
            self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)
        elif name == DoclangToken.HEADING.value:
            self._parse_heading(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.FIELD_HEADING.value:
            self._parse_field_heading(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.LIST.value:
            self._parse_list(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.GROUP.value:
            # Float + footnote siblings: parse as one unit (not a Docling GroupItem).
            if self._first_child(el, DoclangToken.TABLE.value) or self._first_child(el, DoclangToken.INDEX.value):
                self._parse_table(doc=doc, el=el, parent=parent)
            elif self._first_child(el, DoclangToken.PICTURE.value):
                self._parse_picture(doc=doc, el=el, parent=parent)
            else:
                self._walk_children(doc=doc, el=el, parent=parent)
        elif name in {DoclangToken.TABLE.value, DoclangToken.INDEX.value}:
            self._parse_table(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.PICTURE.value:
            self._parse_picture(doc=doc, el=el, parent=parent)
        else:
            self._walk_children(doc=doc, el=el, parent=parent)

    def _walk_children(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        for node in el.childNodes:
            if isinstance(node, Element):
                # Ignore geometry/meta containers at this level; pass through page breaks
                if node.tagName in {
                    DoclangToken.HEAD.value,
                    DoclangToken.LOCATION.value,
                    DoclangToken.LAYER.value,
                    DoclangToken.LABEL.value,
                    DoclangToken.CUSTOM.value,
                    DoclangToken.CAPTION.value,
                    DoclangToken.SRC.value,
                }:
                    continue
                self._dispatch_element(doc=doc, el=node, parent=parent)

    # ------------- Text blocks -------------

    def _should_preserve_space(self, el: Element) -> bool:
        return el.tagName == DoclangToken.CONTENT.value  # and el.getAttribute("xml:space") == "preserve"

    def _get_children_simple_text_block(self, element: Element) -> Optional[str]:
        result = None
        for el in element.childNodes:
            if isinstance(el, Element):
                if self._is_element_head_tag(el):
                    continue
                if el.tagName not in {
                    DoclangToken.LOCATION.value,
                    DoclangToken.LAYER.value,
                    DoclangToken.LABEL.value,
                    DoclangToken.BR.value,
                    DoclangToken.BOLD.value,
                    DoclangToken.ITALIC.value,
                    DoclangToken.UNDERLINE.value,
                    DoclangToken.STRIKETHROUGH.value,
                    DoclangToken.SUBSCRIPT.value,
                    DoclangToken.SUPERSCRIPT.value,
                    DoclangToken.RTL.value,
                    DoclangToken.HANDWRITING.value,
                    DoclangToken.CHECKBOX.value,
                    DoclangToken.CONTENT.value,
                }:
                    return None
                elif tmp := self._get_children_simple_text_block(el):
                    result = tmp
            elif isinstance(el, Text) and el.data.strip():  # TODO should still support whitespace-only
                if result is None:
                    result = el.data if element.tagName == DoclangToken.CONTENT.value else el.data.strip()
                else:
                    return None
        return result

    def _parse_text_like(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        """Parse text-like tokens (text, caption, footnotes, code, formula)."""
        element_children = [
            node for node in el.childNodes if isinstance(node, Element) and not self._is_element_head_tag(node)
        ]

        thread_id = self._extract_thread_id(el)
        simple_text = self._get_children_simple_text_block(el)
        if len(element_children) > 1 or (simple_text is None and thread_id is None):
            self._parse_inline_group(doc=doc, el=el, parent=parent)
            return

        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text, formatting = self._extract_text_with_formatting(el)
        if not text:
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=el.tagName)) is not None
                and isinstance(existing, TextItem)
            ):
                if prov_list:
                    self._merge_threaded_text_item(text="", prov_list=prov_list, existing=existing)
            return

        nm = el.tagName

        # Handle code separately (language + content extraction)
        if nm == DoclangToken.CODE.value:
            code_text, lang_label = self._extract_code_content_and_language(el)
            if not code_text.strip():
                return
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=nm)) is not None
                and isinstance(existing, CodeItem)
            ):
                self._merge_threaded_text_item(text=code_text, prov_list=prov_list, existing=existing)
                return
            item = doc.add_code(
                text=code_text,
                code_language=lang_label,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
            )
            self._apply_initial_text_provenance(item, text=code_text, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=nm, item=item)

        # Map text-like tokens to text item labels
        elif nm in (
            text_label_map := {
                DoclangToken.TEXT.value: DocItemLabel.TEXT,
                DoclangToken.CAPTION.value: DocItemLabel.CAPTION,
                DoclangToken.FOOTNOTE.value: DocItemLabel.FOOTNOTE,
                DoclangToken.PAGE_HEADER.value: DocItemLabel.PAGE_HEADER,
                DoclangToken.PAGE_FOOTER.value: DocItemLabel.PAGE_FOOTER,
                DoclangToken.BOLD.value: DocItemLabel.TEXT,
                DoclangToken.ITALIC.value: DocItemLabel.TEXT,
                DoclangToken.UNDERLINE.value: DocItemLabel.TEXT,
                DoclangToken.STRIKETHROUGH.value: DocItemLabel.TEXT,
                DoclangToken.SUBSCRIPT.value: DocItemLabel.TEXT,
                DoclangToken.SUPERSCRIPT.value: DocItemLabel.TEXT,
                DoclangToken.RTL.value: DocItemLabel.TEXT,
                DoclangToken.CONTENT.value: DocItemLabel.TEXT,
            }
        ):
            is_bold = nm == DoclangToken.BOLD.value
            is_italic = nm == DoclangToken.ITALIC.value
            is_underline = nm == DoclangToken.UNDERLINE.value
            is_strikethrough = nm == DoclangToken.STRIKETHROUGH.value
            is_subscript = nm == DoclangToken.SUBSCRIPT.value
            is_superscript = nm == DoclangToken.SUPERSCRIPT.value

            if is_bold or is_italic or is_underline or is_strikethrough or is_subscript or is_superscript:
                formatting = formatting or Formatting()
                if is_bold:
                    formatting.bold = True
                elif is_italic:
                    formatting.italic = True
                elif is_underline:
                    formatting.underline = True
                elif is_strikethrough:
                    formatting.strikethrough = True
                elif is_subscript:
                    formatting.script = Script.SUB
                elif is_superscript:
                    formatting.script = Script.SUPER
            label = text_label_map[nm]
            if nm == DoclangToken.TEXT.value and any(
                c.tagName == DoclangToken.HANDWRITING.value for c in element_children
            ):
                label = DocItemLabel.HANDWRITTEN_TEXT
            elif nm == DoclangToken.TEXT.value:
                # Check for checkbox elements with class attribute
                for c in element_children:
                    if c.tagName == DoclangToken.CHECKBOX.value:
                        checkbox_class = c.getAttribute(DoclangAttributeKey.CLASS.value)
                        if checkbox_class == DoclangAttributeValue.SELECTED.value:
                            label = DocItemLabel.CHECKBOX_SELECTED
                            break
                        elif checkbox_class == DoclangAttributeValue.UNSELECTED.value:
                            label = DocItemLabel.CHECKBOX_UNSELECTED
                            break
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=nm)) is not None
                and isinstance(existing, TextItem)
            ):
                self._merge_threaded_text_item(text=text, prov_list=prov_list, existing=existing)
                return
            item = doc.add_text(
                label=label,
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
                content_layer=content_layer,
            )
            self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=nm, item=item)

        elif nm == DoclangToken.FORMULA.value:
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=nm)) is not None
                and isinstance(existing, FormulaItem)
            ):
                self._merge_threaded_text_item(text=text, prov_list=prov_list, existing=existing)
                return
            item = doc.add_formula(
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=nm, item=item)

    def _extract_code_content_and_language(self, el: Element) -> tuple[str, CodeLanguageLabel]:
        """Extract code content and language from a <code> element."""
        lang_label = CodeLanguageLabel.UNKNOWN
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.LABEL.value:
                label_val = node.getAttribute(DoclangAttributeKey.VALUE.value)
                if label_val:
                    lang_label = _code_language_label_from_doclang(label_val)
                break
        parts: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                if node.data.strip():
                    parts.append(node.data)
            elif isinstance(node, Element):
                nm_child = node.tagName
                if nm_child in {
                    DoclangToken.LOCATION.value,
                    DoclangToken.LAYER.value,
                    DoclangToken.LABEL.value,
                }:
                    continue
                elif nm_child == DoclangToken.BR.value:
                    parts.append("\n")
                else:
                    parts.append(self._get_text(node))

        return "".join(parts), lang_label

    def _parse_heading(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        lvl_txt = el.getAttribute(DoclangAttributeKey.LEVEL.value) or "1"
        try:
            level = int(lvl_txt)
        except Exception:
            level = 1
        # Extract provenance from heading token (if any)
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text = self._get_text(el)
        text_stripped = text.strip()
        if text_stripped:
            thread_id = self._extract_thread_id(el)
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=DoclangToken.HEADING.value)) is not None
                and isinstance(existing, TextItem)
            ):
                self._merge_threaded_text_item(text=text_stripped, prov_list=prov_list, existing=existing)
                return
            # Level 1 maps to TitleItem, level > 1 maps to SectionHeaderItem with level-1
            if level == 1:
                item = doc.add_title(
                    text=text_stripped,
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                )
            else:
                item = doc.add_heading(
                    text=text_stripped,
                    level=level - 1,
                    parent=parent,
                    prov=(prov_list[0] if prov_list else None),
                    content_layer=content_layer,
                )
            self._apply_initial_text_provenance(item, text=text_stripped, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=DoclangToken.HEADING.value, item=item)

    def _parse_field_heading(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        lvl_txt = el.getAttribute(DoclangAttributeKey.LEVEL.value) or "1"
        try:
            level = int(lvl_txt)
        except Exception:
            level = 1
        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text = self._get_text(el)
        text_stripped = text.strip()
        if text_stripped:
            thread_id = self._extract_thread_id(el)
            if (
                thread_id
                and (existing := self._get_thread_item(thread_id, host=DoclangToken.FIELD_HEADING.value)) is not None
                and isinstance(existing, TextItem)
            ):
                self._merge_threaded_text_item(text=text_stripped, prov_list=prov_list, existing=existing)
                return
            item = doc.add_field_heading(
                text=text_stripped,
                level=level,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
            )
            self._apply_initial_text_provenance(item, text=text_stripped, prov_list=prov_list)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=DoclangToken.FIELD_HEADING.value, item=item)

    def _first_non_whitespace_node(self, nodes: Sequence[Node]) -> Optional[Node]:
        """Return the first node that is not whitespace-only text."""
        for node in nodes:
            if isinstance(node, Text) and not node.data.strip():
                continue
            return node
        return None

    def _token_category(self, tag: str) -> Optional[DoclangCategory]:
        try:
            return DoclangVocabulary._get_category(DoclangToken(tag))
        except ValueError:
            return None

    def _is_element_head_tag(self, el: Element) -> bool:
        """Return True when ``el`` is an element allowed in an element head."""
        return el.tagName in _ELEMENT_HEAD_TAGS

    def _is_ignorable_head_whitespace(self, node: Node) -> bool:
        return isinstance(node, Text) and not node.data.strip()

    def _split_element_children_head_body(
        self,
        el: Element,
        *,
        body_starts_at: Optional[Callable[[Node], bool]] = None,
    ) -> tuple[list[Node], list[Node]]:
        """Split immediate children into element-head prefix and body."""
        head_nodes: list[Node] = []
        body_nodes: list[Node] = []
        in_body = False

        for node in el.childNodes:
            if not in_body:
                if body_starts_at is not None and body_starts_at(node):
                    in_body = True
                    body_nodes.append(node)
                    continue
                if self._is_ignorable_head_whitespace(node):
                    head_nodes.append(node)
                    continue
                if isinstance(node, Element) and self._is_element_head_tag(node):
                    head_nodes.append(node)
                    continue
                in_body = True
                body_nodes.append(node)
            else:
                body_nodes.append(node)

        return head_nodes, body_nodes

    def _nodes_to_xml(self, nodes: Sequence[Node]) -> str:
        """Serialize a node sequence to XML/text (unwrap ``<content>`` elements)."""
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element):
                if node.tagName == DoclangToken.CONTENT.value:
                    parts.append(self._nodes_to_xml(node.childNodes))
                else:
                    parts.append(node.toxml())
        return "".join(parts)

    _LIST_ITEM_VIRTUAL_TEXT_CONTENT_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DoclangToken.CONTENT.value,
            DoclangToken.BOLD.value,
            DoclangToken.ITALIC.value,
            DoclangToken.UNDERLINE.value,
            DoclangToken.STRIKETHROUGH.value,
            DoclangToken.SUPERSCRIPT.value,
            DoclangToken.SUBSCRIPT.value,
            DoclangToken.HANDWRITING.value,
            DoclangToken.RTL.value,
            DoclangToken.BR.value,
            DoclangToken.CHECKBOX.value,
        }
    )

    _LIST_ITEM_SEGMENT_SIBLING_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DoclangToken.LIST.value,
            DoclangToken.PICTURE.value,
        }
    )

    def _is_list_item_virtual_text(self, nodes: Sequence[Node]) -> bool:
        """Decide whether list-item content after ``<ldiv>`` is virtual ``<text>``.

        Inspect the first non-whitespace node after the delimiter:
        - element head tokens, raw text, ``<content>``, or inline formatting → virtual text
        - semantic/grouping elements (``<text>``, ``<code>``, ``<list>``, …) → not virtual text
        """
        first = self._first_non_whitespace_node(nodes)
        if first is None:
            return False
        if isinstance(first, Text):
            return True
        if not isinstance(first, Element):
            return False
        if self._is_element_head_tag(first):
            return True
        if first.tagName in self._LIST_ITEM_VIRTUAL_TEXT_CONTENT_TAGS:
            return True
        category = self._token_category(first.tagName)
        if category in {DoclangCategory.FORMATTING, DoclangCategory.CONTENT}:
            return True
        return category not in {DoclangCategory.SEMANTIC, DoclangCategory.GROUPING}

    def _content_nodes_after_list_item_head(self, nodes: Sequence[Node]) -> list[Node]:
        """Drop leading property/head tokens; keep virtual-text body nodes."""
        content_nodes: list[Node] = []
        skipping_head = True
        for node in nodes:
            if skipping_head:
                if isinstance(node, Text) and not node.data.strip():
                    continue
                if isinstance(node, Element) and self._is_element_head_tag(node):
                    continue
                skipping_head = False
            content_nodes.append(node)
        return content_nodes

    def _split_virtual_text_leading_text(self, nodes: Sequence[Node]) -> tuple[str, list[Node]]:
        """Split virtual-text body into leading plain text and remaining nodes."""
        text_parts: list[str] = []
        rest_start = 0
        for i, node in enumerate(nodes):
            if isinstance(node, Text):
                text_parts.append(node.data)
                rest_start = i + 1
            elif isinstance(node, Element) and node.tagName == DoclangToken.CONTENT.value:
                text_parts.append(self._get_text(node))
                rest_start = i + 1
            else:
                break
        leading = "".join(text_parts).strip()
        rest = [node for node in nodes[rest_start:] if not (isinstance(node, Text) and not node.data.strip())]
        return leading, rest

    def _parse_list_item_virtual_text(
        self,
        *,
        doc: DoclingDocument,
        el: Element,
        li_group: ListGroup,
        ordered: bool,
        marker_text: str,
        all_content_nodes: Sequence[Node],
    ) -> None:
        """Parse list-item body emitted as virtual ``<text>`` after ``<ldiv>``."""
        prov_list = self._provenance_from_location_nodes(doc=doc, nodes=all_content_nodes)
        body_nodes = self._content_nodes_after_list_item_head(all_content_nodes)
        leading_text, rest_nodes = self._split_virtual_text_leading_text(body_nodes)
        rest_elements = [node for node in rest_nodes if isinstance(node, Element)]

        if (
            leading_text
            and rest_elements
            and all(el.tagName in self._LIST_ITEM_SEGMENT_SIBLING_TAGS for el in rest_elements)
        ):
            li = self._add_list_item_with_provenance(
                doc=doc,
                text=leading_text,
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
            for content_el in rest_elements:
                self._dispatch_element(doc=doc, el=content_el, parent=li)
        elif not rest_nodes and leading_text:
            self._add_list_item_with_provenance(
                doc=doc,
                text=leading_text,
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
        elif self._is_simple_virtual_text_nodes(body_nodes):
            text = self._get_text_from_nodes(body_nodes).strip()
            self._add_list_item_with_provenance(
                doc=doc,
                text=text,
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
        else:
            li = self._add_list_item_with_provenance(
                doc=doc,
                text="",
                parent=li_group,
                enumerated=ordered,
                marker=marker_text,
                prov_list=prov_list,
            )
            self._parse_inline_group(doc=doc, el=el, parent=li, nodes=body_nodes)

    def _is_simple_virtual_text_nodes(self, nodes: Sequence[Node]) -> bool:
        """True when virtual-text body is plain text (optionally via ``<content>``)."""
        for node in nodes:
            if isinstance(node, Text):
                continue
            if isinstance(node, Element):
                if node.tagName == DoclangToken.CONTENT.value:
                    continue
                return False
        return any(
            (isinstance(node, Text) and node.data.strip())
            or (isinstance(node, Element) and node.tagName == DoclangToken.CONTENT.value)
            for node in nodes
        )

    def _get_text_from_nodes(self, nodes: Sequence[Node]) -> str:
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element) and node.tagName == DoclangToken.CONTENT.value:
                parts.append(self._get_text(node))
        return "".join(parts)

    def _provenance_from_location_nodes(self, *, doc: DoclingDocument, nodes: Sequence[Node]) -> list[ProvenanceItem]:
        """Collect ``<location>`` quartets from a flat node sequence (element head)."""
        values: list[int] = []
        res_for_group: Optional[int] = None
        provs: list[ProvenanceItem] = []

        for node in nodes:
            if not isinstance(node, Element) or node.tagName != DoclangToken.LOCATION.value:
                continue
            try:
                v = int(node.getAttribute(DoclangAttributeKey.VALUE.value) or "0")
            except Exception:
                v = 0
            try:
                r = int(node.getAttribute(DoclangAttributeKey.RESOLUTION.value) or str(self._default_resolution))
            except Exception:
                r = self._default_resolution
            values.append(v)
            res_for_group = r
            if len(values) == 4:
                self._ensure_page_exists(
                    doc=doc,
                    page_no=self._page_no,
                    resolution=res_for_group or self._default_resolution,
                )
                l = float(min(values[0], values[2]))
                t = float(min(values[1], values[3]))
                rgt = float(max(values[0], values[2]))
                btm = float(max(values[1], values[3]))
                bbox = BoundingBox.from_tuple((l, t, rgt, btm), origin=CoordOrigin.TOPLEFT)
                provs.append(ProvenanceItem(page_no=self._page_no, bbox=bbox, charspan=(0, 0)))
                values = []
                res_for_group = None

        return provs

    def _add_list_item_with_provenance(
        self,
        *,
        doc: DoclingDocument,
        text: str,
        parent: NodeItem,
        enumerated: bool,
        marker: str,
        prov_list: list[ProvenanceItem],
    ) -> ListItem:
        item = doc.add_list_item(
            text=text,
            parent=parent,
            enumerated=enumerated,
            marker=marker,
            prov=(prov_list[0] if prov_list else None),
        )
        self._apply_initial_text_provenance(item, text=text, prov_list=prov_list)
        return item

    def _parse_list(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        ordered = el.getAttribute(DoclangAttributeKey.CLASS.value) == DoclangAttributeValue.ORDERED.value
        list_head_nodes = [node for node in el.childNodes if isinstance(node, Element)]
        thread_id = self._extract_thread_id_from_nodes(list_head_nodes)
        if (
            thread_id
            and (existing := self._get_thread_item(thread_id, host=DoclangToken.LIST.value)) is not None
            and isinstance(existing, ListGroup)
        ):
            li_group = existing
        else:
            li_group = doc.add_list_group(parent=parent)
            if thread_id:
                self._register_thread(thread_id=thread_id, host=DoclangToken.LIST.value, item=li_group)
        actual_children = [
            ch for ch in el.childNodes if isinstance(ch, Element) and ch.tagName not in {DoclangToken.LOCATION.value}
        ]

        # Find all ldiv boundaries (delimiters)
        boundaries = [
            i for i, n in enumerate(actual_children) if isinstance(n, Element) and n.tagName == DoclangToken.LDIV.value
        ]

        # Create ranges: each range is from an ldiv to the next ldiv (or end)
        ranges = [
            (
                boundaries[i],
                (boundaries[i + 1] if i < len(boundaries) - 1 else len(actual_children)),
            )
            for i in range(len(boundaries))
        ]

        for start, end in ranges:
            # The ldiv element itself
            ldiv_el = actual_children[start]

            # Extract marker if present within the ldiv
            marker_text = ""
            for ch in ldiv_el.childNodes:
                if isinstance(ch, Element) and ch.tagName == DoclangToken.MARKER.value:
                    marker_text = self._get_text(ch).strip()
                    break

            # Get ALL nodes (including Text nodes) between this ldiv and the next
            # We need to check the original childNodes, not just actual_children
            ldiv_index_in_all = list(el.childNodes).index(ldiv_el)
            next_ldiv_index = None
            if end < len(actual_children):
                next_ldiv_index = list(el.childNodes).index(actual_children[end])

            # Get all nodes between ldivs (including Text nodes and head tokens)
            if next_ldiv_index is not None:
                all_content_nodes = list(el.childNodes)[ldiv_index_in_all + 1 : next_ldiv_index]
            else:
                all_content_nodes = list(el.childNodes)[ldiv_index_in_all + 1 :]

            # Content elements come after the ldiv (start+1 to end) - only Element nodes,
            # excluding property tokens that may appear between ldiv and body content.
            content_elements = [
                node
                for node in actual_children[start + 1 : end]
                if not (isinstance(node, Element) and self._is_element_head_tag(node))
            ]

            is_virtual_text = self._is_list_item_virtual_text(all_content_nodes)

            if not all_content_nodes:
                # Empty list item (just ldiv, no content)
                doc.add_list_item(
                    text="",
                    parent=li_group,
                    enumerated=ordered,
                    marker=marker_text,
                )
            elif is_virtual_text:
                self._parse_list_item_virtual_text(
                    doc=doc,
                    el=el,
                    li_group=li_group,
                    ordered=ordered,
                    marker_text=marker_text,
                    all_content_nodes=all_content_nodes,
                )
            elif len(content_elements) == 1 and isinstance(content_elements[0], Element):
                # Single element after ldiv
                content_el = content_elements[0]
                if content_el.tagName == DoclangToken.TEXT.value:
                    # Check if it's a simple text item or has complex content (code, formula, etc.)
                    element_children = [
                        node
                        for node in content_el.childNodes
                        if isinstance(node, Element)
                        and node.tagName not in {DoclangToken.LOCATION.value, DoclangToken.LAYER.value}
                    ]

                    # If it has complex content (multiple elements or non-simple content), dispatch it
                    if len(element_children) > 1 or self._get_children_simple_text_block(content_el) is None:
                        # Complex content - create empty list item and dispatch the text element
                        li = doc.add_list_item(
                            text="",
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                        )
                        self._dispatch_element(doc=doc, el=content_el, parent=li)
                    else:
                        # Simple text item
                        prov_list = self._extract_provenance(doc=doc, el=content_el)
                        text = self._get_text(content_el).strip()
                        doc.add_list_item(
                            text=text,
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                            prov=(prov_list[0] if prov_list else None),
                        )
                else:
                    # Other single element (heading, code, nested list, etc.)
                    li = doc.add_list_item(
                        text="",
                        parent=li_group,
                        enumerated=ordered,
                        marker=marker_text,
                    )
                    self._dispatch_element(doc=doc, el=content_el, parent=li)
            else:
                # Multiple content elements after ldiv
                # Special case: if first element is a simple <text> and remaining are <list> elements,
                # treat the text as the ListItem's text (collapsed representation)
                first_el = content_elements[0]
                remaining_els = content_elements[1:]

                if (
                    isinstance(first_el, Element)
                    and first_el.tagName == DoclangToken.TEXT.value
                    and all(
                        isinstance(el, Element) and el.tagName in self._LIST_ITEM_SEGMENT_SIBLING_TAGS
                        for el in remaining_els
                    )
                ):
                    # Check if the text element is simple (no complex content)
                    element_children = [
                        node
                        for node in first_el.childNodes
                        if isinstance(node, Element)
                        and node.tagName not in {DoclangToken.LOCATION.value, DoclangToken.LAYER.value}
                    ]

                    if len(element_children) <= 1 and self._get_children_simple_text_block(first_el) is not None:
                        # Simple text - use it as the ListItem's text
                        prov_list = self._extract_provenance(doc=doc, el=first_el)
                        text = self._get_text(first_el).strip()
                        li = doc.add_list_item(
                            text=text,
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                            prov=(prov_list[0] if prov_list else None),
                        )
                        # Dispatch the nested list(s) as children
                        for content_el in remaining_els:
                            self._dispatch_element(doc=doc, el=content_el, parent=li)
                    else:
                        # Complex text content - dispatch all elements
                        li = doc.add_list_item(
                            text="",
                            parent=li_group,
                            enumerated=ordered,
                            marker=marker_text,
                        )
                        for content_el in content_elements:
                            self._dispatch_element(doc=doc, el=content_el, parent=li)
                else:
                    # General case: dispatch all elements as children
                    li = doc.add_list_item(
                        text="",
                        parent=li_group,
                        enumerated=ordered,
                        marker=marker_text,
                    )
                    for content_el in content_elements:
                        self._dispatch_element(doc=doc, el=content_el, parent=li)

    # ------------- Inline groups -------------
    def _parse_inline_group(
        self,
        *,
        doc: DoclingDocument,
        el: Element,
        parent: Optional[NodeItem],
        nodes: Optional[Sequence[Node]] = None,
    ) -> None:
        """Parse <inline> elements into InlineGroup objects."""
        # Create the inline group
        inline_group = doc.add_inline_group(parent=parent)

        # Process all child elements, adding them as children of the inline group
        my_nodes = nodes or el.childNodes
        for node in my_nodes:
            if isinstance(node, Element):
                # Recursively dispatch child elements with the inline group as parent
                self._dispatch_element(doc=doc, el=node, parent=inline_group)
            elif isinstance(node, Text):
                # Handle direct text content
                text_content = node.data.strip()
                if text_content:
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text_content,
                        parent=inline_group,
                    )

    # ------------- Floating items (table / picture) -------------

    @staticmethod
    def _table_label_from_otsl_element(el: Element) -> DocItemLabel:
        """Resolve table label from ``<table>`` or ``<index>`` (v0.5: no ``class`` on ``<table>``)."""
        if el.tagName == DoclangToken.INDEX.value:
            return DocItemLabel.DOCUMENT_INDEX
        if el.tagName != DoclangToken.TABLE.value:
            raise ValueError(f"Expected table or index element, got '{el.tagName}'.")
        if el.getAttribute(DoclangAttributeKey.CLASS.value):
            raise ValueError("table element must not have a class attribute.")
        return DocItemLabel.TABLE

    def _parse_table(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        """Parse ``<table>``, ``<index>``, or a ``<group>`` wrapping them (with footnotes)."""
        otsl_el: Optional[Element]
        footnotes: list[TextItem] = []
        if el.tagName in {DoclangToken.TABLE.value, DoclangToken.INDEX.value}:
            caption = self._extract_caption(doc=doc, el=el)
            otsl_el = el
            table_label = self._table_label_from_otsl_element(el)
        else:
            footnotes = self._extract_footnotes(doc=doc, el=el)
            otsl_el = self._first_child(el, DoclangToken.TABLE.value) or self._first_child(el, DoclangToken.INDEX.value)
            caption = self._extract_caption(doc=doc, el=el)
            if caption is None and otsl_el is not None:
                caption = self._extract_caption(doc=doc, el=otsl_el)
            if otsl_el is None:
                tbl = doc.add_table(data=TableData(), caption=caption, parent=parent)
                for ftn in footnotes:
                    tbl.footnotes.append(ftn.get_ref())
                return
            table_label = self._table_label_from_otsl_element(otsl_el)

        head_nodes, body_nodes = self._split_element_children_head_body(otsl_el)
        tbl_provs = self._provenance_from_location_nodes(doc=doc, nodes=head_nodes)
        content_layer = self._layer_from_nodes(head_nodes)
        thread_id = self._extract_thread_id_from_nodes(head_nodes)
        table_host = otsl_el.tagName
        if (
            thread_id
            and (existing := self._get_thread_item(thread_id, host=table_host)) is not None
            and isinstance(existing, TableItem)
        ):
            row_offset, col_offset = _thread_table_merge_offset(existing, tbl_provs[0]) if tbl_provs else (0, 0)
            for prov in tbl_provs:
                existing.prov.append(prov)
            inner = self._nodes_to_xml(body_nodes)
            if inner.strip():
                tbl_content = _wrap(text=inner, wrap_tag=DoclangToken.TABLE.value)
                fragment_td = self._parse_otsl_table_content(
                    otsl_content=tbl_content,
                    doc=doc,
                    parent=existing,
                    row_offset=row_offset,
                    col_offset=col_offset,
                )
                if existing.data is None:
                    existing.data = fragment_td
                else:
                    _merge_table_data(
                        base=existing.data,
                        fragment=fragment_td,
                        row_offset=row_offset,
                        col_offset=col_offset,
                    )
            return
        inner = self._nodes_to_xml(body_nodes)
        tbl = doc.add_table(
            data=TableData(),
            caption=caption,
            parent=parent,
            prov=(tbl_provs[0] if tbl_provs else None),
            content_layer=content_layer,
            label=table_label,
        )
        tbl_content = _wrap(text=inner, wrap_tag=DoclangToken.TABLE.value)
        td = self._parse_otsl_table_content(otsl_content=tbl_content, doc=doc, parent=tbl)
        tbl.data = td
        for p in tbl_provs[1:]:
            tbl.prov.append(p)
        if thread_id:
            self._register_thread(thread_id=thread_id, host=table_host, item=tbl)
        for ftn in footnotes:
            tbl.footnotes.append(ftn.get_ref())

    def _parse_picture(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        """Parse ``<picture>`` or a ``<group>`` wrapping it (with footnotes)."""
        picture_el: Optional[Element]
        footnotes: list[TextItem] = []
        if el.tagName == DoclangToken.PICTURE.value:
            caption = self._extract_caption(doc=doc, el=el)
            picture_el = el
        else:
            footnotes = self._extract_footnotes(doc=doc, el=el)
            picture_el = self._first_child(el, DoclangToken.PICTURE.value)
            caption = self._extract_caption(doc=doc, el=el)
            if caption is None and picture_el is not None:
                caption = self._extract_caption(doc=doc, el=picture_el)

        prov_list: list[ProvenanceItem] = []
        content_layer: Optional[ContentLayer] = None
        if picture_el is not None:
            prov_list = self._extract_provenance(doc=doc, el=picture_el)
            content_layer = self._extract_layer(el=picture_el)

        pic = doc.add_picture(
            caption=caption,
            parent=parent,
            prov=(prov_list[0] if prov_list else None),
            content_layer=content_layer,
        )
        for p in prov_list[1:]:
            pic.prov.append(p)
        for ftn in footnotes:
            pic.footnotes.append(ftn.get_ref())

        if picture_el is not None:
            if label_val := self._extract_label_value(el=picture_el):
                if class_name := _picture_classification_label_from_doclang(label_val):
                    if pic.meta is None:
                        pic.meta = PictureMeta()
                    pic.meta.classification = PictureClassificationMetaField(
                        predictions=[
                            PictureClassificationPrediction(
                                class_name=class_name,
                                confidence=1.0,
                            )
                        ]
                    )
            otsl_el = self._first_child(picture_el, DoclangToken.TABLE.value)
            if otsl_el is not None:
                head_nodes, body_nodes = self._split_element_children_head_body(otsl_el)
                inner = self._nodes_to_xml(body_nodes)
                td = self._parse_otsl_table_content(_wrap(inner, DoclangToken.TABLE.value))
                if pic.meta is None:
                    pic.meta = PictureMeta()
                pic.meta.tabular_chart = TabularChartMetaField(chart_data=td)

    # ------------- Helpers -------------
    def _extract_caption(self, *, doc: DoclingDocument, el: Element) -> Optional[TextItem]:
        """Extract caption from element head or from a ``<group>`` wrapper around a float."""
        cap_el = self._first_child(el, DoclangToken.CAPTION.value)
        if cap_el is None:
            return None
        text = self._get_text(cap_el).strip()
        if not text:
            return None
        prov_list = self._extract_provenance(doc=doc, el=cap_el)
        item = doc.add_text(
            label=DocItemLabel.CAPTION,
            text=text,
            prov=(prov_list[0] if prov_list else None),
        )
        for p in prov_list[1:]:
            item.prov.append(p)
        return item

    def _extract_footnotes(self, *, doc: DoclingDocument, el: Element) -> list[TextItem]:
        footnotes: list[TextItem] = []
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.FOOTNOTE.value:
                text = self._get_text(node).strip()
                if text:
                    prov_list = self._extract_provenance(doc=doc, el=node)
                    item = doc.add_text(
                        label=DocItemLabel.FOOTNOTE,
                        text=text,
                        prov=(prov_list[0] if prov_list else None),
                    )
                    for p in prov_list[1:]:
                        item.prov.append(p)
                    footnotes.append(item)
        return footnotes

    def _first_child(self, el: Element, tag_name: str) -> Optional[Element]:
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == tag_name:
                return node
        return None

    def _inner_xml(self, el: Element, exclude_tags: Optional[set[str]] = None) -> str:
        """Extract inner XML content, optionally excluding specific element tags.

        Args:
            el: The element to extract content from
            exclude_tags: Optional set of tag names to exclude from the output
        """
        parts: list[str] = []
        exclude_tags = exclude_tags or set()
        for node in el.childNodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element):
                if node.tagName == DoclangToken.CONTENT.value:
                    res = self._inner_xml(node, exclude_tags=exclude_tags)
                    parts.append(res)
                elif node.tagName not in exclude_tags:
                    parts.append(node.toxml())
        return "".join(parts)

    def _layer_from_nodes(self, nodes: Sequence[Node]) -> Optional[ContentLayer]:
        """Extract content layer from ``<layer value=\"...\"/>`` in element head nodes."""
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.LAYER.value:
                if layer_value := node.getAttribute(DoclangAttributeKey.VALUE.value):
                    try:
                        return ContentLayer(layer_value)
                    except ValueError:
                        pass
        return None

    def _label_value_from_nodes(self, nodes: Sequence[Node]) -> Optional[str]:
        """Extract ``<label value=\"...\"/>`` from element head nodes."""
        for node in nodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.LABEL.value:
                if label_val := node.getAttribute(DoclangAttributeKey.VALUE.value):
                    return label_val
        return None

    # --------- OTSL table parsing (inlined) ---------
    _OTSL_STRUCTURAL_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            DoclangToken.FCEL.value,
            DoclangToken.ECEL.value,
            DoclangToken.LCEL.value,
            DoclangToken.UCEL.value,
            DoclangToken.XCEL.value,
            DoclangToken.NL.value,
            DoclangToken.CHED.value,
            DoclangToken.RHED.value,
            DoclangToken.SROW.value,
            DoclangToken.CORN.value,
        }
    )

    def _bbox_from_location_text_fragments(
        self, *, doc: DoclingDocument, fragments: list[str]
    ) -> Optional[BoundingBox]:
        """Build a TOPLEFT bbox from four ``<location value=\"...\"/>`` XML fragments."""
        if len(fragments) != 4:
            return None
        values: list[int] = []
        res_for_group: Optional[int] = None
        for fragment in fragments:
            frag_dom = parseString(fragment)
            loc_el = frag_dom.documentElement
            if loc_el is None or loc_el.tagName != DoclangToken.LOCATION.value:
                return None
            try:
                v = int(loc_el.getAttribute(DoclangAttributeKey.VALUE.value) or "0")
            except Exception:
                v = 0
            try:
                r = int(loc_el.getAttribute(DoclangAttributeKey.RESOLUTION.value) or str(self._default_resolution))
            except Exception:
                r = self._default_resolution
            values.append(v)
            res_for_group = r
        self._ensure_page_exists(
            doc=doc,
            page_no=self._page_no,
            resolution=res_for_group or self._default_resolution,
        )
        l = float(min(values[0], values[2]))
        t = float(min(values[1], values[3]))
        rgt = float(max(values[0], values[2]))
        btm = float(max(values[1], values[3]))
        return BoundingBox.from_tuple((l, t, rgt, btm), origin=CoordOrigin.TOPLEFT)

    def _consume_leading_location_fragments(
        self,
        *,
        doc: Optional[DoclingDocument],
        texts: list[str],
        start: int,
    ) -> tuple[int, Optional[BoundingBox]]:
        """Consume a leading quartet of location fragments; return next index and bbox."""
        frags: list[str] = []
        idx = start
        loc_tag = f"<{DoclangToken.LOCATION.value}"
        while idx < len(texts) and texts[idx].strip().startswith(loc_tag):
            frags.append(texts[idx])
            idx += 1
            if len(frags) == 4:
                bbox = self._bbox_from_location_text_fragments(doc=doc, fragments=frags) if doc is not None else None
                return idx, bbox
        return start, None

    def _consume_otsl_cell_body_parts(self, texts: list[str], start: int) -> tuple[int, list[str]]:
        """Collect OTSL cell body fragments until the next structural token."""
        structural = {
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.FCEL),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.ECEL),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.LCEL),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.UCEL),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.XCEL),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.NL),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.CHED),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.RHED),
            DoclangVocabulary._create_selfclosing_token(token=DoclangToken.SROW),
        }
        parts: list[str] = []
        idx = start
        while idx < len(texts) and texts[idx] not in structural:
            parts.append(texts[idx])
            idx += 1
        return idx, parts

    def _otsl_extract_tokens_and_text(self, s: str) -> tuple[list[str], list[str]]:
        """Extract OTSL structural tokens and interleaved text.

        Strips the outer wrapper and preserves OTSL body content including per-cell
        ``<location>`` tokens. Handles nested XML elements (like
        ``<text><italic>...</italic></text>``) by keeping them as single units.
        """

        tokens: list[str] = []
        parts: list[str] = []

        dom = parseString(s)
        otsl_el = dom.documentElement
        if otsl_el is None:
            raise ValueError("No document element found")

        otsl_tokens = {
            DoclangToken.FCEL.value,
            DoclangToken.ECEL.value,
            DoclangToken.LCEL.value,
            DoclangToken.UCEL.value,
            DoclangToken.XCEL.value,
            DoclangToken.NL.value,
            DoclangToken.CHED.value,
            DoclangToken.RHED.value,
            DoclangToken.SROW.value,
            DoclangToken.CORN.value,
        }

        for node in otsl_el.childNodes:
            if isinstance(node, Text):
                text = node.data.strip()
                if text:
                    parts.append(text)
            elif isinstance(node, Element):
                tag_name = node.tagName
                if tag_name in otsl_tokens:
                    token_str = f"<{tag_name}/>"
                    tokens.append(token_str)
                    parts.append(token_str)
                else:
                    # This is a nested element (like <text>, <italic>, etc.)
                    # Keep it as a complete XML string
                    xml_str = node.toxml()
                    parts.append(xml_str)

        return tokens, parts

    def _otsl_parse_texts(
        self,
        texts: list[str],
        tokens: list[str],
        doc: Optional["DoclingDocument"] = None,
        parent: Optional[NodeItem] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> tuple[list[TableCell], list[list[str]]]:
        """Parse OTSL interleaved texts+tokens into TableCell list and row tokens."""
        # Token strings used in the stream (normalized to <name>)

        fcel = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.FCEL)
        ecel = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.ECEL)
        lcel = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.LCEL)
        ucel = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.UCEL)
        xcel = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.XCEL)
        nl = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.NL)
        ched = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.CHED)
        rhed = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.RHED)
        srow = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.SROW)
        corn = DoclangVocabulary._create_selfclosing_token(token=DoclangToken.CORN)

        # Clean tokens to only structural OTSL markers
        clean_tokens: list[str] = []
        for t in tokens:
            if t in [ecel, fcel, lcel, ucel, xcel, nl, ched, rhed, srow, corn]:
                clean_tokens.append(t)
        tokens = clean_tokens

        # Split into rows by NL markers while keeping segments
        split_row_tokens = [list(group) for is_sep, group in groupby(tokens, key=lambda z: z == nl) if not is_sep]

        table_cells: list[TableCell] = []
        r_idx = 0
        c_idx = 0

        def count_right(rows: list[list[str]], c: int, r: int, which: list[str]) -> int:
            span = 0
            j = c
            while j < len(rows[r]) and rows[r][j] in which:
                j += 1
                span += 1
            return span

        def count_down(rows: list[list[str]], c: int, r: int, which: list[str]) -> int:
            span = 0
            i = r
            while i < len(rows) and c < len(rows[i]) and rows[i][c] in which:
                i += 1
                span += 1
            return span

        origin_tokens = [fcel, ecel, ched, rhed, srow, corn]
        continuation_origin_tokens = [lcel, ucel, xcel]

        for i, t in enumerate(texts):
            cell_text = ""
            if t in origin_tokens + continuation_origin_tokens:
                row_span = 1
                col_span = 1
                cell_bbox: Optional[BoundingBox] = None
                content_idx = i + 1
                cell_parts: list[str] = []
                if t != ecel and content_idx < len(texts):
                    content_idx, cell_bbox = self._consume_leading_location_fragments(
                        doc=doc,
                        texts=texts,
                        start=content_idx,
                    )
                    content_idx, cell_parts = self._consume_otsl_cell_body_parts(texts, content_idx)
                    cell_text = "".join(cell_parts)

                is_continuation_origin = t in continuation_origin_tokens
                if is_continuation_origin and not cell_text.strip() and not cell_parts:
                    pass
                else:
                    next_right = texts[content_idx] if content_idx < len(texts) else ""
                    next_bottom = (
                        split_row_tokens[r_idx + 1][c_idx]
                        if (r_idx + 1) < len(split_row_tokens) and c_idx < len(split_row_tokens[r_idx + 1])
                        else ""
                    )

                    if next_right in [lcel, xcel]:
                        col_span += count_right(split_row_tokens, c_idx + 1, r_idx, [lcel, xcel])
                    if next_bottom in [ucel, xcel]:
                        row_span += count_down(split_row_tokens, c_idx, r_idx + 1, [ucel, xcel])

                    cell_text_stripped = cell_text.strip()
                    xml_parts = [
                        part.strip()
                        for part in cell_parts
                        if part.strip().startswith("<") and part.strip().endswith(">")
                    ]
                    cell_added = False
                    if xml_parts and doc is not None and parent is not None:
                        cell_group = doc.add_group(parent=parent, label=GroupLabel.UNSPECIFIED)
                        text_parts: list[str] = []
                        for part in xml_parts:
                            wrapped_xml = f"<root>{part}</root>"
                            dom = parseString(wrapped_xml)
                            root_el = dom.documentElement
                            if root_el is None:
                                raise ValueError("No document element found")
                            for child_node in root_el.childNodes:
                                if isinstance(child_node, Element):
                                    self._dispatch_element(doc=doc, el=child_node, parent=cell_group)
                                    text_parts.append(self._get_text(child_node))
                        actual_text = "".join(text_parts).strip() or cell_text_stripped
                        table_cells.append(
                            RichTableCell(
                                text=actual_text,
                                row_span=row_span,
                                col_span=col_span,
                                start_row_offset_idx=r_idx + row_offset,
                                end_row_offset_idx=r_idx + row_span + row_offset,
                                start_col_offset_idx=c_idx + col_offset,
                                end_col_offset_idx=c_idx + col_span + col_offset,
                                ref=cell_group.get_ref(),
                                bbox=cell_bbox,
                            )
                        )
                        cell_added = True

                    if not cell_added:
                        table_cells.append(
                            TableCell(
                                text=cell_text_stripped,
                                row_span=row_span,
                                col_span=col_span,
                                start_row_offset_idx=r_idx + row_offset,
                                end_row_offset_idx=r_idx + row_span + row_offset,
                                start_col_offset_idx=c_idx + col_offset,
                                end_col_offset_idx=c_idx + col_span + col_offset,
                                column_header=t in [ched, corn],
                                row_header=t in [rhed, corn],
                                row_section=t == srow,
                                bbox=cell_bbox,
                            )
                        )

            if t in origin_tokens + continuation_origin_tokens:
                c_idx += 1
            if t == nl:
                r_idx += 1
                c_idx = 0

        return table_cells, split_row_tokens

    def _parse_otsl_table_content(
        self,
        otsl_content: str,
        doc: Optional["DoclingDocument"] = None,
        parent: Optional[NodeItem] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> TableData:
        """Parse OTSL content into TableData (inlined from utils)."""
        tokens, mixed = self._otsl_extract_tokens_and_text(otsl_content)
        table_cells, split_rows = self._otsl_parse_texts(
            mixed,
            tokens,
            doc=doc,
            parent=parent,
            row_offset=row_offset,
            col_offset=col_offset,
        )
        return TableData(
            num_rows=len(split_rows),
            num_cols=(max(len(r) for r in split_rows) if split_rows else 0),
            table_cells=table_cells,
        )

    def _extract_text_with_formatting(self, el: Element) -> tuple[str, Optional[Formatting]]:
        """Extract text content and formatting from an element.

        If the element contains a single formatting child (bold, italic, etc.),
        recursively extract the text and build up the Formatting object.

        Returns:
            Tuple of (text_content, formatting_object or None)
        """
        # Get non-whitespace, non-location child elements
        child_elements = [
            node
            for node in el.childNodes
            if isinstance(node, Element) and node.tagName not in {DoclangToken.LOCATION.value}
        ]

        # Check if we have a single child that is a formatting tag
        if len(child_elements) == 1:
            child = child_elements[0]
            tag_name = child.tagName

            # Mapping of format tags to Formatting attributes
            format_tags = {
                DoclangToken.BOLD,
                DoclangToken.ITALIC,
                DoclangToken.STRIKETHROUGH,
                DoclangToken.UNDERLINE,
                DoclangToken.SUPERSCRIPT,
                DoclangToken.SUBSCRIPT,
                DoclangToken.RTL,
            }

            if tag_name in format_tags:
                # Recursively extract text and formatting from the child
                text, child_formatting = self._extract_text_with_formatting(child)

                # Build up the formatting object
                if child_formatting is None:
                    child_formatting = Formatting()

                # Apply the current formatting tag
                if tag_name == DoclangToken.BOLD.value:
                    child_formatting.bold = True
                elif tag_name == DoclangToken.ITALIC.value:
                    child_formatting.italic = True
                elif tag_name == DoclangToken.STRIKETHROUGH.value:
                    child_formatting.strikethrough = True
                elif tag_name == DoclangToken.UNDERLINE.value:
                    child_formatting.underline = True
                elif tag_name == DoclangToken.SUPERSCRIPT.value:
                    child_formatting.script = Script.SUPER
                elif tag_name == DoclangToken.SUBSCRIPT.value:
                    child_formatting.script = Script.SUB

                return text, child_formatting

        # No formatting found, just extract plain text
        return self._get_text(el), None

    def _get_text(self, el: Element) -> str:
        out: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                # Skip pure indentation/pretty-print whitespace
                if node.data.strip():
                    out.append(node.data if el.tagName == DoclangToken.CONTENT.value else node.data.strip())
            elif isinstance(node, Element):
                nm = node.tagName
                if nm in {DoclangToken.LOCATION.value}:
                    continue
                if nm == DoclangToken.BR.value:
                    out.append("\n")
                else:
                    out.append(self._get_text(node))
        return "".join(out)

    # --------- Location helpers ---------
    def _ensure_page_exists(self, *, doc: DoclingDocument, page_no: int, resolution: int) -> None:
        # If the page already exists, do nothing; otherwise add with a square size based on resolution
        if page_no not in doc.pages:
            doc.add_page(page_no=page_no, size=Size(width=resolution, height=resolution))

    def _extract_provenance(self, *, doc: DoclingDocument, el: Element) -> list[ProvenanceItem]:
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._provenance_from_location_nodes(doc=doc, nodes=head_nodes)

    def _extract_layer(self, *, el: Element) -> Optional[ContentLayer]:
        """Extract content layer from element-head ``<layer value=\"...\"/>``."""
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._layer_from_nodes(head_nodes)

    def _extract_label_value(self, *, el: Element) -> Optional[str]:
        """Extract ``<label value=\"...\"/>`` from element head."""
        head_nodes, _ = self._split_element_children_head_body(el)
        return self._label_value_from_nodes(head_nodes)
