"""Shared DocLang vocabulary and helpers (private)."""

import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, ClassVar, Final, Literal, Optional, cast
from xml.dom.minidom import Element, Node, Text

from pydantic import BaseModel

from docling_core.types.doc import (
    DocItem,
    DoclingDocument,
    NodeItem,
    ProvenanceItem,
    TableData,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import FieldItem, FieldRegionItem
from docling_core.types.doc.labels import CodeLanguageLabel, PictureClassificationLabel

DOCLANG_NAMESPACE: Final = "https://www.doclang.ai/ns/v0"
_DOCLANG_VERSION: Final = "0.9"
DOCLANG_DFLT_RESOLUTION: int = 512

ET.register_namespace("", DOCLANG_NAMESPACE)  # prevent prefix from ET.tostring()


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


def _escape_xml_attr(val: str) -> str:
    """Escape special XML characters in an attribute value.

    Args:
        val: The attribute value to escape.

    Returns:
        The escaped attribute value safe for inclusion inside double quotes.
    """
    return val.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


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
    return _wrap(text=text, wrap_tag=DocLangToken.FIELD_REGION.value)


def _wrap_in_field_item_if_needed(*, text: str, item: DocItem, doc: DoclingDocument) -> str:
    """Wrap serialized key/value markup in ``<field_item>`` when not already under one."""
    if _has_field_item_ancestor(item=item, doc=doc):
        return text
    return _wrap(text=text, wrap_tag=DocLangToken.FIELD_ITEM.value)


def _wrap_field_kv_markup_if_needed(*, text: str, item: DocItem, doc: DoclingDocument) -> str:
    """Ensure key/value XML is nested under ``field_item`` (and ``field_region`` when orphan)."""
    text = _wrap_in_field_item_if_needed(text=text, item=item, doc=doc)
    # Keys/values under an existing field_item rely on that item's field_region wrapper.
    if not _has_field_item_ancestor(item=item, doc=doc):
        text = _wrap_in_field_region_if_needed(text=text, item=item, doc=doc)
    return text


def _wrap_token(*, text: str, open_token: str) -> str:
    close_token = DocLangVocabulary._create_closing_token(token=open_token)
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


def _quantize_to_resolution(value: int, resolution: int) -> int:
    """Clamp discrete location value to ``[0, resolution)``."""
    if value < 0:
        warnings.warn(f"Location {value=} less than 0; returning 0", stacklevel=2)
        return 0
    if value == resolution:
        return resolution - 1
    if value > resolution:
        warnings.warn(
            f"Location {value=} greater than {resolution - 1}; returning {resolution - 1=}",
            stacklevel=2,
        )
        return resolution - 1
    return value


def _create_location_tokens_for_bbox(
    *,
    bbox: tuple[float, float, float, float],
    page_w: float,
    page_h: float,
    xres: int,
    yres: int,
) -> str:
    """Create four `<location .../>` tokens for x0,y0,x1,y1 given a bbox."""
    x0 = min(bbox[0], bbox[2]) / page_w
    y0 = min(bbox[1], bbox[3]) / page_h
    x1 = max(bbox[0], bbox[2]) / page_w
    y1 = max(bbox[1], bbox[3]) / page_h

    x0v = _quantize_to_resolution(round(xres * x0), xres)
    y0v = _quantize_to_resolution(round(yres * y0), yres)
    x1v = _quantize_to_resolution(round(xres * x1), xres)
    y1v = _quantize_to_resolution(round(yres * y1), yres)

    return (
        DocLangVocabulary._create_location_token(value=x0v, resolution=xres)
        + DocLangVocabulary._create_location_token(value=y0v, resolution=yres)
        + DocLangVocabulary._create_location_token(value=x1v, resolution=xres)
        + DocLangVocabulary._create_location_token(value=y1v, resolution=yres)
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
        out.append(
            _create_location_tokens_for_bbox(
                bbox=bbox,
                page_w=page_w,
                page_h=page_h,
                xres=xres,
                yres=yres,
            )
        )

    # Multi-provenance items emit one location set per fragment
    if len(out) > 1:
        res = []
        for i, _ in enumerate(item.prov):
            res.append(f"{i} {_}")
        err = "\n".join(res)

        raise ValueError(f"We have more than 1 location for this item [{item.label}]:\n\n{err}\n\n{out}")

    return "".join(out)


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


class DocLangCategory(str, Enum):
    """DocLangCategory."""

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


class DocLangToken(str, Enum):
    """DocLangToken."""

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
    DESCRIPTION = "description"
    SUMMARY = "summary"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PICTURE = "picture"
    FORMULA = "formula"
    CODE = "code"
    LDIV = "ldiv"
    CHECKBOX = "checkbox"
    TABLE = "table"
    TABULAR = "tabular"
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


class DocLangAttributeKey(str, Enum):
    """Attribute keys allowed on DocLang tokens."""

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
    NAME = "name"


class DocLangAttributeValue(str, Enum):
    """Enumerated values for specific DocLang attributes."""

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


class DocLangVocabulary(BaseModel):
    """DocLangVocabulary."""

    # Allowed attributes per token (defined outside the Enum to satisfy mypy)
    ALLOWED_ATTRIBUTES: ClassVar[dict[DocLangToken, set["DocLangAttributeKey"]]] = {
        DocLangToken.DOCUMENT: {
            DocLangAttributeKey.XMLNS,
            DocLangAttributeKey.VERSION,
        },
        DocLangToken.LOCATION: {
            DocLangAttributeKey.VALUE,
            DocLangAttributeKey.RESOLUTION,
        },
        DocLangToken.LAYER: {DocLangAttributeKey.VALUE},
        DocLangToken.LABEL: {DocLangAttributeKey.VALUE},
        DocLangToken.SRC: {DocLangAttributeKey.URI},
        DocLangToken.HREF: {DocLangAttributeKey.URI},
        DocLangToken.HOUR: {DocLangAttributeKey.VALUE},
        DocLangToken.MINUTE: {DocLangAttributeKey.VALUE},
        DocLangToken.SECOND: {DocLangAttributeKey.VALUE},
        DocLangToken.CENTISECOND: {DocLangAttributeKey.VALUE},
        DocLangToken.HEADING: {DocLangAttributeKey.LEVEL},
        DocLangToken.FIELD_HEADING: {DocLangAttributeKey.LEVEL},
        DocLangToken.CHECKBOX: {DocLangAttributeKey.CLASS},
        DocLangToken.LIST: {DocLangAttributeKey.CLASS},
        DocLangToken.THREAD: {DocLangAttributeKey.THREAD_ID},
        DocLangToken.XREF: {DocLangAttributeKey.THREAD_ID},
        DocLangToken.GROUP: {DocLangAttributeKey.NAME},
    }

    # Allowed values for specific attributes (enumerations)
    # Structure: token -> attribute name -> set of allowed string values
    ALLOWED_ATTRIBUTE_VALUES: ClassVar[
        dict[
            DocLangToken,
            dict["DocLangAttributeKey", set["DocLangAttributeValue"]],
        ]
    ] = {
        # Grouping and inline enumerations
        DocLangToken.LIST: {
            DocLangAttributeKey.CLASS: {
                DocLangAttributeValue.ORDERED,
                DocLangAttributeValue.UNORDERED,
            }
        },
        DocLangToken.CHECKBOX: {
            DocLangAttributeKey.CLASS: {
                DocLangAttributeValue.SELECTED,
                DocLangAttributeValue.UNSELECTED,
            }
        },
        # Other attributes (e.g., level, type, thread_id) are not enumerated here
    }

    ALLOWED_ATTRIBUTE_RANGE: ClassVar[dict[DocLangToken, dict["DocLangAttributeKey", tuple[int, int]]]] = {
        # Geometric: value in [0, res]; resolution optional.
        # Keep conservative defaults aligned with existing usage.
        DocLangToken.LOCATION: {
            DocLangAttributeKey.VALUE: (0, DOCLANG_DFLT_RESOLUTION),  # TODO: review
            DocLangAttributeKey.RESOLUTION: (
                DOCLANG_DFLT_RESOLUTION,
                DOCLANG_DFLT_RESOLUTION,
            ),  # TODO: review
        },
        # Temporal components
        DocLangToken.HOUR: {DocLangAttributeKey.VALUE: (0, 99)},
        DocLangToken.MINUTE: {DocLangAttributeKey.VALUE: (0, 59)},
        DocLangToken.SECOND: {DocLangAttributeKey.VALUE: (0, 59)},
        DocLangToken.CENTISECOND: {DocLangAttributeKey.VALUE: (0, 99)},
        # Levels (N ≥ 1)
        DocLangToken.HEADING: {DocLangAttributeKey.LEVEL: (1, 6)},
        DocLangToken.FIELD_HEADING: {DocLangAttributeKey.LEVEL: (1, 6)},
        # Continuation markers (thread_id length constraints)
        DocLangToken.THREAD: {DocLangAttributeKey.THREAD_ID: (1, 10)},
        DocLangToken.XREF: {DocLangAttributeKey.THREAD_ID: (1, 10)},
    }

    # Self-closing tokens set
    IS_SELFCLOSING: ClassVar[set[DocLangToken]] = {
        DocLangToken.PAGE_BREAK,
        DocLangToken.LOCATION,
        DocLangToken.LAYER,
        DocLangToken.LABEL,
        DocLangToken.SRC,
        DocLangToken.HREF,
        DocLangToken.HOUR,
        DocLangToken.MINUTE,
        DocLangToken.SECOND,
        DocLangToken.CENTISECOND,
        DocLangToken.BR,
        DocLangToken.CHECKBOX,
        DocLangToken.LDIV,
        # OTSL structural tokens are emitted as self-closing markers
        DocLangToken.FCEL,
        DocLangToken.ECEL,
        DocLangToken.CHED,
        DocLangToken.RHED,
        DocLangToken.CORN,
        DocLangToken.SROW,
        DocLangToken.LCEL,
        DocLangToken.UCEL,
        DocLangToken.XCEL,
        DocLangToken.NL,
        # Continuation markers
        DocLangToken.THREAD,
    }

    # Token to category mapping
    TOKEN_CATEGORIES: ClassVar[dict[DocLangToken, DocLangCategory]] = {
        # Root
        DocLangToken.DOCUMENT: DocLangCategory.ROOT,
        # Metadata
        DocLangToken.HEAD: DocLangCategory.METADATA,
        # Special
        DocLangToken.PAGE_BREAK: DocLangCategory.SPECIAL,
        # Geometric
        DocLangToken.LOCATION: DocLangCategory.GEOMETRIC,
        DocLangToken.LAYER: DocLangCategory.GEOMETRIC,
        # Temporal
        DocLangToken.HOUR: DocLangCategory.TEMPORAL,
        DocLangToken.MINUTE: DocLangCategory.TEMPORAL,
        DocLangToken.SECOND: DocLangCategory.TEMPORAL,
        DocLangToken.CENTISECOND: DocLangCategory.TEMPORAL,
        # Semantic
        DocLangToken.HEADING: DocLangCategory.SEMANTIC,
        DocLangToken.TEXT: DocLangCategory.SEMANTIC,
        DocLangToken.CAPTION: DocLangCategory.SEMANTIC,
        DocLangToken.FOOTNOTE: DocLangCategory.SEMANTIC,
        DocLangToken.PAGE_HEADER: DocLangCategory.SEMANTIC,
        DocLangToken.PAGE_FOOTER: DocLangCategory.SEMANTIC,
        DocLangToken.PICTURE: DocLangCategory.SEMANTIC,
        DocLangToken.FIELD_REGION: DocLangCategory.SEMANTIC,
        DocLangToken.FIELD_ITEM: DocLangCategory.SEMANTIC,
        DocLangToken.FIELD_HEADING: DocLangCategory.SEMANTIC,
        DocLangToken.FIELD_HINT: DocLangCategory.SEMANTIC,
        DocLangToken.FORMULA: DocLangCategory.SEMANTIC,
        DocLangToken.CODE: DocLangCategory.SEMANTIC,
        DocLangToken.CHECKBOX: DocLangCategory.SEMANTIC,
        DocLangToken.LIST: DocLangCategory.SEMANTIC,
        DocLangToken.TABLE: DocLangCategory.SEMANTIC,
        DocLangToken.TABULAR: DocLangCategory.SEMANTIC,
        # Grouping
        DocLangToken.GROUP: DocLangCategory.GROUPING,
        # Formatting
        DocLangToken.BOLD: DocLangCategory.FORMATTING,
        DocLangToken.ITALIC: DocLangCategory.FORMATTING,
        DocLangToken.STRIKETHROUGH: DocLangCategory.FORMATTING,
        DocLangToken.SUPERSCRIPT: DocLangCategory.FORMATTING,
        DocLangToken.SUBSCRIPT: DocLangCategory.FORMATTING,
        DocLangToken.HANDWRITING: DocLangCategory.FORMATTING,
        DocLangToken.RTL: DocLangCategory.FORMATTING,
        DocLangToken.BR: DocLangCategory.FORMATTING,
        # Structural
        DocLangToken.LDIV: DocLangCategory.STRUCTURAL,
        DocLangToken.FCEL: DocLangCategory.STRUCTURAL,
        DocLangToken.ECEL: DocLangCategory.STRUCTURAL,
        DocLangToken.CHED: DocLangCategory.STRUCTURAL,
        DocLangToken.RHED: DocLangCategory.STRUCTURAL,
        DocLangToken.CORN: DocLangCategory.STRUCTURAL,
        DocLangToken.SROW: DocLangCategory.STRUCTURAL,
        DocLangToken.LCEL: DocLangCategory.STRUCTURAL,
        DocLangToken.UCEL: DocLangCategory.STRUCTURAL,
        DocLangToken.XCEL: DocLangCategory.STRUCTURAL,
        DocLangToken.NL: DocLangCategory.STRUCTURAL,
        DocLangToken.FIELD_KEY: DocLangCategory.STRUCTURAL,
        DocLangToken.FIELD_VALUE: DocLangCategory.STRUCTURAL,
        # Continuation
        DocLangToken.THREAD: DocLangCategory.CONTINUATION,
        # Content/Binary data
        DocLangToken.HREF: DocLangCategory.CONTENT,
        DocLangToken.XREF: DocLangCategory.CONTENT,
        DocLangToken.MARKER: DocLangCategory.CONTENT,
        DocLangToken.CONTENT: DocLangCategory.CONTENT,
    }

    @classmethod
    def _get_category(cls, token: DocLangToken) -> DocLangCategory:
        """Get the category for a given DocLang token.

        Args:
            token: The DocLang token to look up.

        Returns:
            The corresponding DocLangCategory for the token.

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
                DocLangToken(name)
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
            tok_enum = DocLangToken(name)
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
        namespace: str | None = None,
        version: str | None = None,
        closing: bool = False,
    ) -> str:
        """Create the document root tag.

        - When `closing` is True, returns the closing root tag.
        """
        if closing:
            return f"</{DocLangToken.DOCUMENT.value}>"
        else:
            parts = [DocLangToken.DOCUMENT.value]
            if namespace is not None:
                parts.append(f'{DocLangAttributeKey.XMLNS.value}="{namespace}"')
            if version is not None:
                parts.append(f'{DocLangAttributeKey.VERSION.value}="{version}"')
            return f"<{' '.join(parts)}>"

    @classmethod
    def _create_threading_token(cls, *, thread_id: str) -> str:
        """Create a vertical continuation threading token.

        Emits `<thread thread_id="..."/>`. Validates required attributes
        against the class schema and basic value sanity.
        """
        token = DocLangToken.THREAD
        assert DocLangAttributeKey.THREAD_ID in cls.ALLOWED_ATTRIBUTES.get(token, set())

        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][DocLangAttributeKey.THREAD_ID]
        length = len(thread_id)
        if not (lo <= length <= hi):
            raise ValueError(f"thread_id length must be in [{lo}, {hi}]")

        return f'<{token.value} {DocLangAttributeKey.THREAD_ID.value}="{thread_id}"/>'

    @classmethod
    def _create_group_token(cls, *, name: str | None = None, closing: bool = False) -> str:
        """Create a group tag.

        - When `closing` is True, returns the closing tag.
        - When `name` is given, returns an opening tag carrying it, otherwise an
          opening tag without attributes.
        """
        token = DocLangToken.GROUP
        if closing:
            return f"</{token.value}>"
        if name is None:
            return f"<{token.value}>"
        assert DocLangAttributeKey.NAME in cls.ALLOWED_ATTRIBUTES.get(token, set())
        safe = _escape_xml_attr(name)
        return f'<{token.value} {DocLangAttributeKey.NAME.value}="{safe}">'

    @classmethod
    def _create_list_token(cls, *, ordered: bool, closing: bool = False) -> str:
        """Create a list tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag with an `ordered` boolean attribute.
        """
        if closing:
            return f"</{DocLangToken.LIST.value}>"
        elif ordered:
            return (
                f'<{DocLangToken.LIST.value} {DocLangAttributeKey.CLASS.value}="{DocLangAttributeValue.ORDERED.value}">'
            )
        else:
            return f"<{DocLangToken.LIST.value}>"

    @classmethod
    def _create_level_open_token(cls, *, token: DocLangToken, level: int) -> str:
        """Create an opening tag; level 1 omits the ``level`` attribute."""
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][DocLangAttributeKey.LEVEL]
        if not (lo <= level <= hi):
            raise ValueError(f"level must be in [{lo}, {hi}]")
        if level == 1:
            return f"<{token.value}>"
        return f'<{token.value} {DocLangAttributeKey.LEVEL.value}="{level}">'

    @classmethod
    def _create_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a heading tag with validated level.

        Level 1 is emitted as bare ``<heading>``; levels 2-6 use ``level="N"``.
        """
        if closing:
            return f"</{DocLangToken.HEADING.value}>"
        return cls._create_level_open_token(token=DocLangToken.HEADING, level=level)

    @classmethod
    def _create_field_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a field-heading tag with validated level.

        Level 1 is emitted as bare ``<field_heading>``; levels 2-6 use ``level="N"``.
        """
        if closing:
            return f"</{DocLangToken.FIELD_HEADING.value}>"
        return cls._create_level_open_token(token=DocLangToken.FIELD_HEADING, level=level)

    @classmethod
    def _create_location_token(cls, *, value: int, resolution: int) -> str:
        """Create a location token with value and resolution.

        Validates both attributes using the configured ranges and ensures
        `value` lies within [0, resolution]. Always emits the resolution
        attribute for explicitness.
        """
        if not (0 <= value < resolution):
            raise ValueError(f"value ({value}) must be in [0, {resolution})")

        return f'<{DocLangToken.LOCATION.value} {DocLangAttributeKey.VALUE.value}="{value}"/>'

    @classmethod
    def get_special_tokens(
        cls,
        *,
        include_location_tokens: bool = True,
        include_temporal_tokens: bool = True,
    ) -> list[str]:
        """Return all DocLang special tokens.

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
            DocLangToken.HOUR,
            DocLangToken.MINUTE,
            DocLangToken.SECOND,
            DocLangToken.CENTISECOND,
        }

        for token in DocLangToken:
            # Optional gating for location/temporal tokens
            if not include_location_tokens and token is DocLangToken.LOCATION:
                continue
            if not include_temporal_tokens and token in temporal_tokens:
                continue

            name = token.value
            is_selfclosing = token in cls.IS_SELFCLOSING

            # Attribute-aware emission
            attrs = cls.ALLOWED_ATTRIBUTES.get(token, set())
            if attrs:
                if token is DocLangToken.LIST:
                    special_tokens.append(f"<{name}>")
                    special_tokens.append(f"</{name}>")
                    special_tokens.append(
                        f'<{name} {DocLangAttributeKey.CLASS.value}="{DocLangAttributeValue.ORDERED.value}">'
                    )
                    special_tokens.append(f"</{name}>")
                    continue
                if token in {DocLangToken.HEADING, DocLangToken.FIELD_HEADING}:
                    level_attr = DocLangAttributeKey.LEVEL
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
                    if token is DocLangToken.LOCATION and attr_name is DocLangAttributeKey.RESOLUTION:
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
        token: DocLangToken,
        attrs: dict["DocLangAttributeKey", Any] | None = None,
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
            token=DocLangToken.CHECKBOX,
            attrs={
                DocLangAttributeKey.CLASS: (
                    DocLangAttributeValue.SELECTED if selected else DocLangAttributeValue.UNSELECTED
                ),
            },
        )


_DOCLANG_LABEL_UNDEFINED = "undefined"
_DOCLANG_LABEL_OTHER = "other"


def _picture_classification_label_to_doclang(class_name: str) -> str:
    """Map Docling picture classification label to DocLang recommended label."""
    if class_name == PictureClassificationLabel.OTHER.value:
        return _DOCLANG_LABEL_OTHER
    return class_name


def _picture_classification_label_from_doclang(label_val: str) -> str | None:
    """Map DocLang picture label to Docling picture classification label."""
    if label_val == _DOCLANG_LABEL_UNDEFINED:
        return None
    if label_val == _DOCLANG_LABEL_OTHER:
        return PictureClassificationLabel.OTHER.value
    return label_val


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


# DocLang custom-vocabulary tags for standard Docling meta fields (``namespace__field``).
_DOCLANG_META_NAMESPACE: Final = "docling"
_DOCLANG_META_TAG_SUMMARY: Final = f"{_DOCLANG_META_NAMESPACE}__summary"
_DOCLANG_META_TAG_DESCRIPTION: Final = f"{_DOCLANG_META_NAMESPACE}__description"
_DOCLANG_META_TAG_SMILES: Final = f"{_DOCLANG_META_NAMESPACE}__smiles"

# Tokens allowed in an element head (before body content). Mirrors
# ``_element_head_prefix`` serialization order plus continuation/temporal tokens.
_ELEMENT_HEAD_TAGS: Final[frozenset[str]] = frozenset(
    {
        DocLangToken.LABEL.value,
        DocLangToken.LAYER.value,
        DocLangToken.HREF.value,
        DocLangToken.LOCATION.value,
        DocLangToken.CAPTION.value,
        DocLangToken.DESCRIPTION.value,
        DocLangToken.SUMMARY.value,
        DocLangToken.CUSTOM.value,
        DocLangToken.THREAD.value,
        DocLangToken.XREF.value,
        DocLangToken.HOUR.value,
        DocLangToken.MINUTE.value,
        DocLangToken.SECOND.value,
        DocLangToken.CENTISECOND.value,
    }
)

# Formatting tags, which can themselves hold mixed content, e.g. <bold>a <italic>b</italic></bold>.
_FORMATTING_TAGS: Final[frozenset[str]] = frozenset(
    {
        DocLangToken.BOLD.value,
        DocLangToken.ITALIC.value,
        DocLangToken.UNDERLINE.value,
        DocLangToken.STRIKETHROUGH.value,
        DocLangToken.SUBSCRIPT.value,
        DocLangToken.SUPERSCRIPT.value,
    }
)

# Elements whose body allows raw text; a text block spans nested ones (e.g. <bold> in <text>).
_RAW_TEXT_TAGS: Final[frozenset[str]] = frozenset(
    {
        DocLangToken.TEXT.value,
        DocLangToken.HEADING.value,
        DocLangToken.CAPTION.value,
        DocLangToken.DESCRIPTION.value,
        DocLangToken.SUMMARY.value,
        DocLangToken.FOOTNOTE.value,
        DocLangToken.PAGE_HEADER.value,
        DocLangToken.PAGE_FOOTER.value,
        DocLangToken.FORMULA.value,
        DocLangToken.CODE.value,
        DocLangToken.FIELD_KEY.value,
        DocLangToken.FIELD_VALUE.value,
        DocLangToken.FIELD_HEADING.value,
        DocLangToken.FIELD_HINT.value,
        DocLangToken.MARKER.value,
        DocLangToken.HANDWRITING.value,
        DocLangToken.RTL.value,
        DocLangToken.CONTENT.value,
        "chapter",  # <track> elements, not in the vocabulary yet
        "voice",
    }
    | _FORMATTING_TAGS
)
# Tags that may continue a virtual <text> (a list item or cell given as unwrapped text).
_VIRTUAL_TEXT_INLINE_TAGS: Final[frozenset[str]] = _FORMATTING_TAGS | {
    DocLangToken.HANDWRITING.value,
    DocLangToken.RTL.value,
    DocLangToken.CONTENT.value,
    DocLangToken.BR.value,
}
# Semantic elements that are inline within a virtual <text>, but a block on their own.
_VIRTUAL_TEXT_SEMANTIC_TAGS: Final[frozenset[str]] = frozenset({DocLangToken.CODE.value, DocLangToken.FORMULA.value})


def _is_virtual_text_part(node: Node) -> bool:
    """True when ``node`` may be part of a virtual <text> (in a list item or cell)."""
    return isinstance(node, Text) or (
        isinstance(node, Element)
        and (node.tagName in _VIRTUAL_TEXT_INLINE_TAGS or node.tagName in _VIRTUAL_TEXT_SEMANTIC_TAGS)
    )


def _is_virtual_text(nodes: Sequence[Node]) -> bool:
    """True when ``nodes`` hold raw text or text-level elements of their own.

    Only then are semantic elements among them (e.g. ``<code>``) inline; otherwise they are blocks.
    """
    return any(
        (isinstance(node, Text) and node.data.strip(_XML_WHITESPACE))
        or (isinstance(node, Element) and node.tagName in _VIRTUAL_TEXT_INLINE_TAGS)
        for node in nodes
    )


# DocLang 0.9 whitespace handling (spec "Content encoding and whitespace handling").
_XML_WHITESPACE: Final = " \t\r\n"

# Left as written: <custom> holds foreign markup, and <track> is not supported yet.
_WHITESPACE_OPAQUE_TAGS: Final[frozenset[str]] = frozenset({DocLangToken.CUSTOM.value, "track"})

# A flattened text block: unprotected text nodes, <content> elements, and "content" for any
# other visible item (e.g. <br/>, an inline <picture>).
_WsToken = Text | Element | Literal["content"]
_TextBlockVisitor = Callable[[Sequence[Node]], None]


def _for_each_text_block(el: Element, visit: _TextBlockVisitor) -> None:
    """Call ``visit`` with the nodes of each text block under ``el``, innermost blocks first."""
    if el.tagName in _WHITESPACE_OPAQUE_TAGS:
        return
    if el.tagName in _RAW_TEXT_TAGS:
        visit(el.childNodes)
        return
    # Not a text block itself; unwrapped text between its child elements is a virtual <text>.
    segment: list[Node] = []

    def visit_segment() -> None:
        if _is_virtual_text(segment):
            visit(segment)
            return
        # No text of its own: whitespace between blocks, which are read on their own.
        visit([node for node in segment if not isinstance(node, Element) or node.tagName in _ELEMENT_HEAD_TAGS])
        for node in segment:
            if isinstance(node, Element) and node.tagName in _VIRTUAL_TEXT_SEMANTIC_TAGS:
                _for_each_text_block(node, visit)

    for node in el.childNodes:
        if _is_virtual_text_part(node) or (isinstance(node, Element) and node.tagName in _ELEMENT_HEAD_TAGS):
            segment.append(node)
            continue
        visit_segment()
        segment = []
        if isinstance(node, Element):
            _for_each_text_block(node, visit)
    visit_segment()


def _flatten_text_block(nodes: Sequence[Node], visit: _TextBlockVisitor) -> list[_WsToken]:
    """Flatten a text block, visiting the text blocks nested in it (element heads etc.)."""
    out: list[_WsToken] = []
    for node in nodes:
        if isinstance(node, Text):
            out.append(node)
        elif not isinstance(node, Element):
            continue  # comments and processing instructions are disregarded
        elif node.tagName in _ELEMENT_HEAD_TAGS:
            _for_each_text_block(node, visit)  # an element head is read separately, as if removed
        elif node.tagName == DocLangToken.CONTENT.value:
            out.append(node)  # literal; its whitespace never joins a sequence outside it
        elif node.tagName == DocLangToken.PAGE_BREAK.value:
            continue  # a page boundary within the text, transparent to its reading
        elif node.tagName in _RAW_TEXT_TAGS:
            out.extend(_flatten_text_block(node.childNodes, visit))
        else:
            out.append("content")
            if node.tagName != DocLangToken.BR.value:
                _for_each_text_block(node, visit)
    return out


def _content_text(el: Element) -> str:
    return "".join(node.data for node in el.childNodes if isinstance(node, Text))


def _read_text_block(tokens: Sequence[_WsToken]) -> list[str]:
    """Return the reading of each token: rules 2-4 for text nodes, literal for <content>.

    "content" tokens read as ``"\0"``, a placeholder for something visible that is not a space.
    """
    reading: list[list[str]] = [[] for _ in tokens]
    seen_content = False
    in_sequence = False
    # Indexes of the text nodes holding the first character of each sequence since the last content.
    pending: list[int] = []

    def resolve_pending() -> None:
        # Content follows: sequences that also have content before them read as a space.
        if seen_content:
            for idx in pending:
                reading[idx].append(" ")
        pending.clear()

    for idx, tok in enumerate(tokens):
        if isinstance(tok, Text):
            for char in tok.data:
                if char in _XML_WHITESPACE:
                    if not in_sequence:
                        pending.append(idx)
                        in_sequence = True
                    continue
                resolve_pending()
                in_sequence = False
                seen_content = True
                reading[idx].append(char)
            continue
        literal = _content_text(tok) if isinstance(tok, Element) else "\0"
        if literal:
            resolve_pending()
            seen_content = True
            reading[idx].append(literal)
        in_sequence = False
    # Sequences still pending sit at the end of the block: dropped (rule 2).
    return ["".join(chars) for chars in reading]


def _apply_whitespace_rules(root: Element) -> None:
    """Rewrite the text nodes under ``root`` in place, as DocLang 0.9 reads them.

    Whitespace outside text blocks and at their edges is dropped; any other whitespace
    sequence becomes one space, kept in the text node holding its first character.
    ``<content>`` is literal. Afterwards, a text node holds exactly its own reading.
    """

    def visit(nodes: Sequence[Node]) -> None:
        tokens = _flatten_text_block(nodes, visit)
        for tok, text in zip(tokens, _read_text_block(tokens)):
            if isinstance(tok, Text):
                tok.data = text

    _for_each_text_block(root, visit)


# Whitespace that DocLang does not read as written: anything but a single space.
_COLLAPSIBLE_WHITESPACE_RE: Final = re.compile(r"[\t\r\n]|  ")


def _unwrap_redundant_content(root: Element) -> None:
    """Unwrap each ``<content>`` under ``root`` whose text reads the same without it.

    The spec asks producers to use ``<content>`` only where the whitespace rules would change
    the reading. Whether they do depends on the neighbours, which a run serializer cannot see:
    a run's trailing space needs no wrapper before ``<bold>x``, but does at the end of a block.
    """

    def visit(nodes: Sequence[Node]) -> None:
        tokens = _flatten_text_block(nodes, visit)
        reading = _read_text_block(tokens)
        for idx, tok in enumerate(tokens):
            if not isinstance(tok, Element) or not (text := reading[idx]) or _COLLAPSIBLE_WHITESPACE_RE.search(text):
                continue
            before = next((r[-1] for r in reversed(reading[:idx]) if r), None)
            after = next((r[0] for r in reading[idx + 1 :] if r), None)
            # Bare, an edge space would be dropped at the block edge, or merge with a neighbouring one.
            # Next to a non-text item ("\0", e.g. a <checkbox/>) it is kept too: readers map such
            # items to items of their own, which leaves the space at an item edge.
            if text[0] == " " and before in (None, " ", "\0"):
                continue
            if text[-1] == " " and after in (None, " ", "\0"):
                continue
            parent = cast(Node, tok.parentNode)
            for child in list(tok.childNodes):
                parent.insertBefore(child, tok)
            parent.removeChild(tok)

    _for_each_text_block(root, visit)
