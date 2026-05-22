"""Define classes for Doclang serialization."""

import copy
import re
import warnings
import xml.etree.ElementTree as ET
from enum import Enum
from itertools import groupby
from typing import Any, ClassVar, Final, Optional, cast
from xml.dom.minidom import Element, Node, Text

from defusedxml.ElementTree import fromstring
from defusedxml.minidom import parseString
from pydantic import BaseModel, PrivateAttr
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
    create_ser_result,
)
from docling_core.types.doc import (
    BaseMeta,
    BoundingBox,
    CodeItem,
    ContentLayer,
    DescriptionMetaField,
    KeywordsMetaField,
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
    TopicsMetaField,
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

# Note: Intentionally avoid importing DocumentToken here to ensure
# Doclang uses only its own token vocabulary.

DOCLANG_NAMESPACE: Final = "https://www.doclang.ai/ns/v0"
DOCLANG_VERSION: Final = "0.2"
DOCLANG_DFLT_RESOLUTION: int = 512

ET.register_namespace("", DOCLANG_NAMESPACE)  # prevent prefix from ET.tostring()


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


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

    # In a proper serialization, we should use <thread id="1|2|3|...|"/> to link different
    # sections together ...
    if len(out) > 1:
        res = []
        for i, _ in enumerate(item.prov):
            res.append(f"{i} {_}")
        err = "\n".join(res)

        raise ValueError(f"We have more than 1 location for this item [{item.label}]:\n\n{err}\n\n{out}")

    return "".join(out)


class DoclangCategory(str, Enum):
    """DoclangCtegory.

    Doclang defines the following categories of elements:

    - **root**: Elements that establish document scope such as
      `doclang`.
    - **special**: Elements that establish document pagination, such as
      `page_break`, and `time_break`.
    - **geometric**: Elements that capture geometric position as normalized
      coordinates/bounding boxes (via repeated `location`) anchoring
      block-level content to the page.
    - **temporal**: Elements that capture temporal positions using
      `<hour value={integer}/><minute value={integer}/><second value={integer}/>`
      and `<centisecond value={integer}/>` for a timestamp and a double
      timestamp for time intervals.
    - **semantic**: Block-level elements that convey document meaning
      (e.g., headings, paragraphs, captions, lists, forms, tables, formulas,
      code, pictures), optionally preceded by location tokens.
    - **formatting**: Inline elements that modify textual presentation within
      semantic content (e.g., `bold`, `italic`, `strikethrough`,
      `superscript`, `subscript`, `rtl`, `inline class="formula|code|picture"`,
      `br`).
    - **grouping**: Elements that organize semantic blocks into logical
      hierarchies and composites (e.g., `section`, `list`, `group type=*`)
      and never carry location tokens.
    - **structural**: Sequence tokens that define internal structure for
      complex constructs (primarily OTSL table layout: `otsl`, `fcel`,
      `ecel`, `lcel`, `ucel`, `xcel`, `nl`, `ched`, `rhed`, `corn`, `srow`;
      and form parts like `key`/`value`).
    - **content**: Lightweight content helpers used inside semantic blocks for
      explicit payload and annotations (e.g., `marker`).
    - **binary data**: Elements that embed or reference non-text payloads for
      media—either inline as `base64` or via `uri`—allowed under `picture`,
      `inline class="picture"`, or at page level.
    - **metadata**: Elements that provide metadata about the document or its
      components, contained within `head` and `meta` respectively.
    - **continuation** tokens: Markers that indicate content spanning pages or
      table boundaries (e.g., `thread`, `h_thread`, each with a required
      `id` attribute) to stitch split content (e.g., across columns or pages).
    """

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
    META = "meta"

    # Special
    PAGE_BREAK = "page_break"
    TIME_BREAK = "time_break"

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
    WATERMARK = "watermark"
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
    SECTION = "section"
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
    H_THREAD = "h_thread"

    # Binary data / content helpers
    URI = "uri"
    MARKER = "marker"
    FACETS = "facets"
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
    ID = "id"


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

    # Table class values
    INDEX = "index"
    DATA = "data"

    # Group class values (deprecated, kept for backward compatibility in deserialization)
    DOCUMENT_INDEX = "document_index"
    TABLE = "table"
    FORM = "form"


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
        DoclangToken.LAYER: {DoclangAttributeKey.CLASS},
        DoclangToken.HOUR: {DoclangAttributeKey.VALUE},
        DoclangToken.MINUTE: {DoclangAttributeKey.VALUE},
        DoclangToken.SECOND: {DoclangAttributeKey.VALUE},
        DoclangToken.CENTISECOND: {DoclangAttributeKey.VALUE},
        DoclangToken.HEADING: {DoclangAttributeKey.LEVEL},
        DoclangToken.FIELD_HEADING: {DoclangAttributeKey.LEVEL},
        DoclangToken.CHECKBOX: {DoclangAttributeKey.CLASS},
        DoclangToken.LIST: {DoclangAttributeKey.CLASS},
        DoclangToken.TABLE: {DoclangAttributeKey.CLASS},
        DoclangToken.GROUP: {DoclangAttributeKey.TYPE},
        DoclangToken.THREAD: {DoclangAttributeKey.ID},
        DoclangToken.H_THREAD: {DoclangAttributeKey.ID},
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
        DoclangToken.TABLE: {
            DoclangAttributeKey.CLASS: {
                DoclangAttributeValue.INDEX,
                DoclangAttributeValue.DATA,
            }
        },
        # Other attributes (e.g., level, type, id) are not enumerated here
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
        # Continuation markers (id length constraints)
        DoclangToken.THREAD: {DoclangAttributeKey.ID: (1, 10)},
        DoclangToken.H_THREAD: {DoclangAttributeKey.ID: (1, 10)},
    }

    # Self-closing tokens set
    IS_SELFCLOSING: ClassVar[set[DoclangToken]] = {
        DoclangToken.PAGE_BREAK,
        DoclangToken.TIME_BREAK,
        DoclangToken.LOCATION,
        DoclangToken.LAYER,
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
        DoclangToken.H_THREAD,
    }

    # Token to category mapping
    TOKEN_CATEGORIES: ClassVar[dict[DoclangToken, DoclangCategory]] = {
        # Root
        DoclangToken.DOCUMENT: DoclangCategory.ROOT,
        # Metadata
        DoclangToken.HEAD: DoclangCategory.METADATA,
        DoclangToken.META: DoclangCategory.METADATA,
        # Special
        DoclangToken.PAGE_BREAK: DoclangCategory.SPECIAL,
        DoclangToken.TIME_BREAK: DoclangCategory.SPECIAL,
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
        DoclangToken.WATERMARK: DoclangCategory.SEMANTIC,
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
        DoclangToken.H_THREAD: DoclangCategory.CONTINUATION,
        # Content/Binary data
        DoclangToken.URI: DoclangCategory.CONTENT,
        DoclangToken.MARKER: DoclangCategory.CONTENT,
        DoclangToken.FACETS: DoclangCategory.CONTENT,
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
    def _create_threading_token(cls, *, id: str, horizontal: bool = False) -> str:
        """Create a continuation threading token.

        Emits `<thread id="..."/>` or `<h_thread id="..."/>` depending on
        the `horizontal` flag. Validates required attributes against the
        class schema and basic value sanity.
        """
        token = DoclangToken.H_THREAD if horizontal else DoclangToken.THREAD
        # Ensure the required attribute is declared for this token
        assert DoclangAttributeKey.ID in cls.ALLOWED_ATTRIBUTES.get(token, set())

        # Validate id length if a range is specified
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][DoclangAttributeKey.ID]
        length = len(id)
        if not (lo <= length <= hi):
            raise ValueError(f"id length must be in [{lo}, {hi}]")

        return f'<{token.value} {DoclangAttributeKey.ID.value}="{id}"/>'

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
            return f'<{DoclangToken.LIST.value} {DoclangAttributeKey.CLASS.value}="{DoclangAttributeValue.UNORDERED.value}">'

    @classmethod
    def _create_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a heading tag with validated level.

        When `closing` is False, emits an opening tag with level attribute.
        When `closing` is True, emits the corresponding closing tag.
        """
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[DoclangToken.HEADING][DoclangAttributeKey.LEVEL]
        if not (lo <= level <= hi):
            raise ValueError(f"level must be in [{lo}, {hi}]")

        if closing:
            return f"</{DoclangToken.HEADING.value}>"
        return f'<{DoclangToken.HEADING.value} {DoclangAttributeKey.LEVEL.value}="{level}">'

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
    WRAP_WHEN_NEEDED = "wrap_when_needed"  # wrap text only if it has leading or trailing whitespace


class LayerMode(str, Enum):
    """Layer mode for Doclang output."""

    ALWAYS = "always"  # always include layer element
    MINIMAL = "minimal"  # include layer element only when it differs from default


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
    include_namespace: bool = True
    include_version: bool = True
    # Virtual text mode: when True, list items and table cells omit <text> wrapper
    use_virtual_texts: bool = False


def _create_layer_token(
    *,
    item: DocItem,
    params: DoclangParams,
) -> str:
    """Create `<layer .../>` token for an item's content layer if needed."""
    if params.layer_mode == LayerMode.ALWAYS or (
        params.layer_mode == LayerMode.MINIMAL and item.content_layer != ContentLayer.BODY
    ):
        return DoclangVocabulary._create_selfclosing_token(
            token=DoclangToken.LAYER,
            attrs={DoclangAttributeKey.CLASS: item.content_layer.value},
        )
    return ""


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
        child_texts: list[str] = []

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
                    child_texts.append(sub_res.text)
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
                child_texts.append(child_res.text)

            # After the <ldiv>, append any nested lists (children of this ListItem)
            # as siblings at the same level (not wrapped in <ldiv>).
            for subref in child.children:
                sub = subref.resolve(doc)
                if isinstance(sub, ListGroup) and sub.self_ref not in my_visited and sub.self_ref not in excluded:
                    my_visited.add(sub.self_ref)
                    sub_res = doc_serializer.serialize(
                        item=sub,
                        list_level=list_level + 1,
                        is_inline_scope=is_inline_scope,
                        visited=my_visited,
                        **kwargs,
                    )
                    if sub_res.text:
                        child_texts.append(sub_res.text)
                    item_results.append(sub_res)

        delim = _get_delim(params=params)
        if child_texts:
            text_res = delim.join(child_texts)
            text_res = f"{text_res}{delim}"
            open_token = (
                DoclangVocabulary._create_list_token(ordered=True)
                if item.first_item_is_enumerated(doc)
                else DoclangVocabulary._create_list_token(ordered=False)
            )
            text_res = _wrap_token(text=text_res, open_token=open_token)
        else:
            text_res = ""
        return create_ser_result(text=text_res, span_source=item_results)


class _LinguistLabel(str, Enum):
    """Linguist-compatible labels for Doclang output."""

    # compatible with GitHub Linguist v9.4.0:
    # https://github.com/github-linguist/linguist/blob/v9.4.0/lib/linguist/languages.yml

    ADA = "Ada"
    AWK = "Awk"
    C = "C"
    C_SHARP = "C#"
    C_PLUS_PLUS = "C++"
    CMAKE = "CMake"
    COBOL = "COBOL"
    CSS = "CSS"
    CEYLON = "Ceylon"
    CLOJURE = "Clojure"
    CRYSTAL = "Crystal"
    CUDA = "Cuda"
    CYTHON = "Cython"
    D = "D"
    DART = "Dart"
    DOCKERFILE = "Dockerfile"
    ELIXIR = "Elixir"
    ERLANG = "Erlang"
    FORTRAN = "Fortran"
    FORTH = "Forth"
    GO = "Go"
    HTML = "HTML"
    HASKELL = "Haskell"
    HAXE = "Haxe"
    JAVA = "Java"
    JAVASCRIPT = "JavaScript"
    JSON = "JSON"
    JULIA = "Julia"
    KOTLIN = "Kotlin"
    COMMON_LISP = "Common Lisp"
    LUA = "Lua"
    MATLAB = "MATLAB"
    MOONSCRIPT = "MoonScript"
    NIM = "Nim"
    OCAML = "OCaml"
    OBJECTIVE_C = "Objective-C"
    PHP = "PHP"
    PASCAL = "Pascal"
    PERL = "Perl"
    PROLOG = "Prolog"
    PYTHON = "Python"
    RACKET = "Racket"
    RUBY = "Ruby"
    RUST = "Rust"
    SHELL = "Shell"
    STANDARD_ML = "Standard ML"
    SQL = "SQL"
    SCALA = "Scala"
    SCHEME = "Scheme"
    SWIFT = "Swift"
    TYPESCRIPT = "TypeScript"
    VISUAL_BASIC_DOT_NET = "Visual Basic .NET"
    XML = "XML"
    YAML = "YAML"

    @classmethod
    def from_code_language_label(self, lang: CodeLanguageLabel) -> Optional["_LinguistLabel"]:
        mapping: dict[CodeLanguageLabel, Optional[_LinguistLabel]] = {
            CodeLanguageLabel.ADA: _LinguistLabel.ADA,
            CodeLanguageLabel.AWK: _LinguistLabel.AWK,
            CodeLanguageLabel.BASH: _LinguistLabel.SHELL,
            CodeLanguageLabel.BC: None,
            CodeLanguageLabel.C: _LinguistLabel.C,
            CodeLanguageLabel.C_SHARP: _LinguistLabel.C_SHARP,
            CodeLanguageLabel.C_PLUS_PLUS: _LinguistLabel.C_PLUS_PLUS,
            CodeLanguageLabel.CMAKE: _LinguistLabel.CMAKE,
            CodeLanguageLabel.COBOL: _LinguistLabel.COBOL,
            CodeLanguageLabel.CSS: _LinguistLabel.CSS,
            CodeLanguageLabel.CEYLON: _LinguistLabel.CEYLON,
            CodeLanguageLabel.CLOJURE: _LinguistLabel.CLOJURE,
            CodeLanguageLabel.CRYSTAL: _LinguistLabel.CRYSTAL,
            CodeLanguageLabel.CUDA: _LinguistLabel.CUDA,
            CodeLanguageLabel.CYTHON: _LinguistLabel.CYTHON,
            CodeLanguageLabel.D: _LinguistLabel.D,
            CodeLanguageLabel.DART: _LinguistLabel.DART,
            CodeLanguageLabel.DC: None,
            CodeLanguageLabel.DOCKERFILE: _LinguistLabel.DOCKERFILE,
            CodeLanguageLabel.ELIXIR: _LinguistLabel.ELIXIR,
            CodeLanguageLabel.ERLANG: _LinguistLabel.ERLANG,
            CodeLanguageLabel.FORTRAN: _LinguistLabel.FORTRAN,
            CodeLanguageLabel.FORTH: _LinguistLabel.FORTH,
            CodeLanguageLabel.GO: _LinguistLabel.GO,
            CodeLanguageLabel.HTML: _LinguistLabel.HTML,
            CodeLanguageLabel.HASKELL: _LinguistLabel.HASKELL,
            CodeLanguageLabel.HAXE: _LinguistLabel.HAXE,
            CodeLanguageLabel.JAVA: _LinguistLabel.JAVA,
            CodeLanguageLabel.JAVASCRIPT: _LinguistLabel.JAVASCRIPT,
            CodeLanguageLabel.JSON: _LinguistLabel.JSON,
            CodeLanguageLabel.JULIA: _LinguistLabel.JULIA,
            CodeLanguageLabel.KOTLIN: _LinguistLabel.KOTLIN,
            CodeLanguageLabel.LISP: _LinguistLabel.COMMON_LISP,
            CodeLanguageLabel.LUA: _LinguistLabel.LUA,
            CodeLanguageLabel.MATLAB: _LinguistLabel.MATLAB,
            CodeLanguageLabel.MOONSCRIPT: _LinguistLabel.MOONSCRIPT,
            CodeLanguageLabel.NIM: _LinguistLabel.NIM,
            CodeLanguageLabel.OCAML: _LinguistLabel.OCAML,
            CodeLanguageLabel.OBJECTIVEC: _LinguistLabel.OBJECTIVE_C,
            CodeLanguageLabel.OCTAVE: _LinguistLabel.MATLAB,
            CodeLanguageLabel.PHP: _LinguistLabel.PHP,
            CodeLanguageLabel.PASCAL: _LinguistLabel.PASCAL,
            CodeLanguageLabel.PERL: _LinguistLabel.PERL,
            CodeLanguageLabel.PROLOG: _LinguistLabel.PROLOG,
            CodeLanguageLabel.PYTHON: _LinguistLabel.PYTHON,
            CodeLanguageLabel.RACKET: _LinguistLabel.RACKET,
            CodeLanguageLabel.RUBY: _LinguistLabel.RUBY,
            CodeLanguageLabel.RUST: _LinguistLabel.RUST,
            CodeLanguageLabel.SML: _LinguistLabel.STANDARD_ML,
            CodeLanguageLabel.SQL: _LinguistLabel.SQL,
            CodeLanguageLabel.SCALA: _LinguistLabel.SCALA,
            CodeLanguageLabel.SCHEME: _LinguistLabel.SCHEME,
            CodeLanguageLabel.SWIFT: _LinguistLabel.SWIFT,
            CodeLanguageLabel.TYPESCRIPT: _LinguistLabel.TYPESCRIPT,
            CodeLanguageLabel.UNKNOWN: None,
            CodeLanguageLabel.VISUALBASIC: _LinguistLabel.VISUAL_BASIC_DOT_NET,
            CodeLanguageLabel.XML: _LinguistLabel.XML,
            CodeLanguageLabel.YAML: _LinguistLabel.YAML,
        }
        return mapping.get(lang)

    @classmethod
    def to_code_language_label(cls, lang: "_LinguistLabel") -> CodeLanguageLabel:
        mapping: dict[_LinguistLabel, CodeLanguageLabel] = {
            _LinguistLabel.ADA: CodeLanguageLabel.ADA,
            _LinguistLabel.AWK: CodeLanguageLabel.AWK,
            _LinguistLabel.C: CodeLanguageLabel.C,
            _LinguistLabel.C_SHARP: CodeLanguageLabel.C_SHARP,
            _LinguistLabel.C_PLUS_PLUS: CodeLanguageLabel.C_PLUS_PLUS,
            _LinguistLabel.CMAKE: CodeLanguageLabel.CMAKE,
            _LinguistLabel.COBOL: CodeLanguageLabel.COBOL,
            _LinguistLabel.CSS: CodeLanguageLabel.CSS,
            _LinguistLabel.CEYLON: CodeLanguageLabel.CEYLON,
            _LinguistLabel.CLOJURE: CodeLanguageLabel.CLOJURE,
            _LinguistLabel.CRYSTAL: CodeLanguageLabel.CRYSTAL,
            _LinguistLabel.CUDA: CodeLanguageLabel.CUDA,
            _LinguistLabel.CYTHON: CodeLanguageLabel.CYTHON,
            _LinguistLabel.D: CodeLanguageLabel.D,
            _LinguistLabel.DART: CodeLanguageLabel.DART,
            _LinguistLabel.DOCKERFILE: CodeLanguageLabel.DOCKERFILE,
            _LinguistLabel.ELIXIR: CodeLanguageLabel.ELIXIR,
            _LinguistLabel.ERLANG: CodeLanguageLabel.ERLANG,
            _LinguistLabel.FORTRAN: CodeLanguageLabel.FORTRAN,
            _LinguistLabel.FORTH: CodeLanguageLabel.FORTH,
            _LinguistLabel.GO: CodeLanguageLabel.GO,
            _LinguistLabel.HTML: CodeLanguageLabel.HTML,
            _LinguistLabel.HASKELL: CodeLanguageLabel.HASKELL,
            _LinguistLabel.HAXE: CodeLanguageLabel.HAXE,
            _LinguistLabel.JAVA: CodeLanguageLabel.JAVA,
            _LinguistLabel.JAVASCRIPT: CodeLanguageLabel.JAVASCRIPT,
            _LinguistLabel.JSON: CodeLanguageLabel.JSON,
            _LinguistLabel.JULIA: CodeLanguageLabel.JULIA,
            _LinguistLabel.KOTLIN: CodeLanguageLabel.KOTLIN,
            _LinguistLabel.COMMON_LISP: CodeLanguageLabel.LISP,
            _LinguistLabel.LUA: CodeLanguageLabel.LUA,
            _LinguistLabel.MATLAB: CodeLanguageLabel.MATLAB,
            _LinguistLabel.MOONSCRIPT: CodeLanguageLabel.MOONSCRIPT,
            _LinguistLabel.NIM: CodeLanguageLabel.NIM,
            _LinguistLabel.OCAML: CodeLanguageLabel.OCAML,
            _LinguistLabel.OBJECTIVE_C: CodeLanguageLabel.OBJECTIVEC,
            _LinguistLabel.PHP: CodeLanguageLabel.PHP,
            _LinguistLabel.PASCAL: CodeLanguageLabel.PASCAL,
            _LinguistLabel.PERL: CodeLanguageLabel.PERL,
            _LinguistLabel.PROLOG: CodeLanguageLabel.PROLOG,
            _LinguistLabel.PYTHON: CodeLanguageLabel.PYTHON,
            _LinguistLabel.RACKET: CodeLanguageLabel.RACKET,
            _LinguistLabel.RUBY: CodeLanguageLabel.RUBY,
            _LinguistLabel.RUST: CodeLanguageLabel.RUST,
            _LinguistLabel.SHELL: CodeLanguageLabel.BASH,
            _LinguistLabel.STANDARD_ML: CodeLanguageLabel.SML,
            _LinguistLabel.SQL: CodeLanguageLabel.SQL,
            _LinguistLabel.SCALA: CodeLanguageLabel.SCALA,
            _LinguistLabel.SCHEME: CodeLanguageLabel.SCHEME,
            _LinguistLabel.SWIFT: CodeLanguageLabel.SWIFT,
            _LinguistLabel.TYPESCRIPT: CodeLanguageLabel.TYPESCRIPT,
            _LinguistLabel.VISUAL_BASIC_DOT_NET: CodeLanguageLabel.VISUALBASIC,
            _LinguistLabel.XML: CodeLanguageLabel.XML,
            _LinguistLabel.YAML: CodeLanguageLabel.YAML,
        }
        return mapping.get(lang, CodeLanguageLabel.UNKNOWN)


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
        if len(item.prov) > 1:
            # Split multi-provenance items into per-provenance items to preserve
            # geometry and spans, then merge text while keeping span mapping.

            # FIXME: if we have an inline group with a multi-provenance, then
            # we will need to do something more complex I believe ...
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
                    **kwargs,
                )
                res.append(tres)

            out = "".join([t.text for t in res])
            return create_ser_result(text=out, span_source=res)

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

    def _determine_list_item_wrapper(
        self, *, item: ListItem, doc: DoclingDocument, use_virtual_texts: bool = False
    ) -> tuple[Optional[str], Optional[DoclangToken]]:
        """Determine the wrapper token for a ListItem.

        Args:
            item: The ListItem to determine wrapper for.
            doc: The document containing the item.
            use_virtual_texts: If True, skip <text> wrapper for list items (virtual text mode).

        Returns:
            Tuple of (wrap_open_token, tok) where wrap_open_token is the opening tag
            string or None, and tok is the DoclangToken or None.
        """
        if item.text:
            # Virtual text mode: no wrapper if use_virtual_texts=True
            if use_virtual_texts:
                return None, None
            else:
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
                item=item, doc=doc, use_virtual_texts=params.use_virtual_texts
            )
        elif isinstance(item, CodeItem):
            tok = DoclangToken.CODE
            if (linguist_lang := _LinguistLabel.from_code_language_label(item.code_language)) is not None:
                wrap_open_token = f'<{tok.value} {DoclangAttributeKey.CLASS.value}="{linguist_lang.value}">'
            else:
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
            if isinstance(item, FieldValueItem):
                wrap_open_token = f'<{tok.value} class="{item.kind}">'
            elif isinstance(item, FieldHeadingItem):
                wrap_open_token = f'<{tok.value} level="{item.level}">'
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

        if item.meta:
            meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
            if meta_res.text:
                parts.append(meta_res.text)

        # Skip adding location tokens if this is a ListItem with InlineGroup child
        # (InlineSerializer will handle location tokens using parent's provenance)
        skip_location = isinstance(item, ListItem) and self._should_skip_location_for_list_item(item=item, doc=doc)

        if params.add_location and not skip_location:
            # Use Doclang `<location>` tokens instead of `<loc_.../>`
            loc = _create_location_tokens_for_item(item=item, doc=doc, xres=params.xsize, yres=params.ysize)
            if loc:
                parts.append(loc)

        if layer_token := _create_layer_token(item=item, params=params):
            parts.append(layer_token)

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
                    sub_parts = [
                        doc_serializer.serialize(item=child_item, visited=my_visited, **kwargs).text
                        for child_ref in item.children
                        # special case: nested lists are serialized as siblings, not children
                        if not (
                            isinstance(child_item := child_ref.resolve(doc), ListGroup) and isinstance(item, ListItem)
                        )
                    ]
                    text_part = _get_delim(params=params).join(sub_parts)
            else:
                text_part = _escape_text(item.text, params)
                text_part = doc_serializer.post_process(
                    text=text_part,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
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
            texts.insert(0, "<meta>")
            texts.append("</meta>")
        return create_ser_result(
            text=elem_delim.join(texts),
            span_source=item if isinstance(item, DocItem) else [],
        )

    def _serialize_meta_field(self, meta: BaseMeta, name: str, params: DoclangParams) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None:
            if name == MetaFieldName.SUMMARY and isinstance(field_val, SummaryMetaField):
                escaped_text = _escape_text(field_val.text, params)
                txt = f"<summary>{escaped_text}</summary>"
            elif name == MetaFieldName.DESCRIPTION and isinstance(field_val, DescriptionMetaField):
                escaped_text = _escape_text(field_val.text, params)
                txt = f"<description>{escaped_text}</description>"
            elif name == MetaFieldName.CLASSIFICATION and isinstance(field_val, PictureClassificationMetaField):
                class_name = self._humanize_text(field_val.get_main_prediction().class_name)
                escaped_class_name = _escape_text(class_name, params)
                txt = f"<classification>{escaped_class_name}</classification>"
            elif name == MetaFieldName.MOLECULE and isinstance(field_val, MoleculeMetaField):
                escaped_smi = _escape_text(field_val.smi, params)
                txt = f"<molecule>{escaped_smi}</molecule>"
            elif name == MetaFieldName.TABULAR_CHART and isinstance(field_val, TabularChartMetaField):
                # suppressing tabular chart serialization
                return None
            elif name == MetaFieldName.KEYWORDS and isinstance(field_val, KeywordsMetaField):
                escaped = ", ".join(_escape_text(v, params) for v in field_val.values)
                txt = f"<keywords>{escaped}</keywords>"
            elif name == MetaFieldName.TOPICS and isinstance(field_val, TopicsMetaField):
                escaped = ", ".join(_escape_text(v, params) for v in field_val.values)
                txt = f"<topics>{escaped}</topics>"
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

        open_token: str = DoclangVocabulary._create_group_token()
        close_token: str = DoclangVocabulary._create_group_token(closing=True)

        # Build caption (as a sibling of the picture within the floating_group)
        res_parts: list[SerializationResult] = []
        caption_text = ""
        if params.add_referenced_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                caption_text = cap_res.text
                res_parts.append(cap_res)

        # Build picture inner content (meta + body) that will go inside <picture> ... </picture>
        picture_inner_parts: list[str] = []
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""
            if params.add_location:
                body += _create_location_tokens_for_item(item=item, doc=doc, xres=params.xsize, yres=params.ysize)

            if layer_token := _create_layer_token(item=item, params=params):
                body += layer_token

            uri: Optional[str] = None
            if params.image_mode in [ImageRefMode.REFERENCED, ImageRefMode.EMBEDDED] and item.image and item.image.uri:
                uri = str(item.image.uri)
            elif params.image_mode == ImageRefMode.EMBEDDED and (img := item.get_image(doc)):
                imgb64 = item._image_to_base64(img)
                uri = f"data:image/png;base64,{imgb64}"

            if uri:
                body += _wrap(text=uri, wrap_tag=DoclangToken.URI.value)

            is_chart = self._picture_is_chart(item)
            is_chem = self._picture_is_chem(item)

            # Visibility & meta rules for picture sub-types:
            #
            # PICTURE present → ALL pictures are emitted (regular, chart,
            #   chem) with classification-only meta (molecule / tabular_chart
            #   blocked).  If a specific type (CHART / CHEMISTRY) is also
            #   present, matching pictures are upgraded to full meta.
            #
            # Only CHART / CHEMISTRY (no PICTURE) → only the matching
            #   pictures are emitted, with full meta.
            has_picture_ct = ContentType.PICTURE in params.content_types
            specific_match = (is_chart and ContentType.CHART in params.content_types) or (
                is_chem and ContentType.CHEMISTRY in params.content_types
            )
            # Decide whether this picture should appear at all:
            #  - has_picture_ct: every picture is shown
            #  - specific_match: this particular picture's type was requested
            any_match = has_picture_ct or specific_match

            if any_match and item.meta:
                if specific_match:
                    # Full meta: classification + specialised fields
                    meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
                elif is_chart or is_chem:
                    # Chart/chem picture without its specific type enabled:
                    # classification only - block everything else
                    meta_kwargs = dict(**kwargs)
                    meta_kwargs["allowed_meta_names"] = {MetaFieldName.CLASSIFICATION}
                    meta_res = doc_serializer.serialize_meta(item=item, **meta_kwargs)
                else:
                    # Regular picture: block specialised fields only
                    meta_kwargs = dict(**kwargs)
                    blocked = set(params.blocked_meta_names)
                    blocked.add(MetaFieldName.MOLECULE)
                    blocked.add(MetaFieldName.TABULAR_CHART)
                    meta_kwargs["blocked_meta_names"] = blocked
                    meta_res = doc_serializer.serialize_meta(item=item, **meta_kwargs)

                if meta_res.text:
                    picture_inner_parts.append(meta_res.text)
                    res_parts.append(meta_res)

                # handle tabular chart data (only when specific type matches)
                if specific_match:
                    chart_data: Optional[TableData] = None
                    if item.meta and item.meta.tabular_chart:
                        chart_data = item.meta.tabular_chart.chart_data
                    if chart_data and chart_data.table_cells:
                        temp_doc = DoclingDocument(name="temp")
                        temp_table = temp_doc.add_table(data=chart_data)
                        # Reuse the Doclang table emission for chart data
                        params_chart = DoclangParams(
                            **{
                                **params.model_dump(),
                                "add_table_cell_location": False,
                            }
                        )
                        otsl_content = DoclangTableSerializer()._emit_otsl(
                            item=temp_table,  # type: ignore[arg-type]
                            doc_serializer=doc_serializer,
                            doc=temp_doc,
                            params=params_chart,
                            **kwargs,
                        )
                        otsl_payload = _wrap(text=otsl_content, wrap_tag=DoclangToken.TABLE.value)
                        body += otsl_payload

            if body:
                picture_inner_parts.append(body)
                res_parts.append(create_ser_result(text=body, span_source=item))

        picture_text = "".join(picture_inner_parts)

        # Build footnotes (as siblings of the picture within the group)
        footnote_text = ""
        if params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        # If we have caption or footnote, always emit <picture> tag (even if empty)
        # so deserializer can identify the group type
        has_caption_or_footnote = bool(caption_text or footnote_text)
        if picture_text or has_caption_or_footnote:
            picture_text = _wrap(text=picture_text, wrap_tag=DoclangToken.PICTURE.value)

        # Emit a bare <picture> only when it is the sole non-empty element.
        # When picture has no content but exists (e.g., PLACEHOLDER mode), emit empty <picture></picture>
        parts = [part for part in (caption_text, picture_text, footnote_text) if part]
        if not parts:
            if params.suppress_empty_elements:
                return create_ser_result()
            # If picture exists but has no content, emit empty <picture></picture> instead of empty <group></group>
            if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
                text_res = _wrap(text="", wrap_tag=DoclangToken.PICTURE.value)
            else:
                text_res = f"{open_token}{close_token}"
        elif len(parts) == 1 and picture_text:
            text_res = picture_text
        else:
            text_res = f"{open_token}{''.join(parts)}{close_token}"

        return create_ser_result(text=text_res, span_source=res_parts)


class DoclangTableSerializer(BaseTableSerializer):
    """Doclang-specific table item serializer."""

    # _get_table_token no longer needed; OTSL tokens are emitted via vocabulary

    def _emit_otsl(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        params: "DoclangParams",
        **kwargs: Any,
    ) -> str:
        """Emit OTSL payload using Doclang tokens and location semantics.

        Location tokens are included only when all required information is available
        (cell bboxes, provenance, page info, valid page size). Otherwise, location
        tokens are omitted without raising errors.
        """
        if not item.data or not item.data.table_cells:
            return ""

        nrows, ncols = item.data.num_rows, item.data.num_cols

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
        for i in range(nrows):
            for j in range(ncols):
                cell = item.data.grid[i][j]
                content = cell._get_text(doc=doc, doc_serializer=doc_serializer, **kwargs).strip()

                rowspan, rowstart = cell.row_span, cell.start_row_offset_idx
                colspan, colstart = cell.col_span, cell.start_col_offset_idx

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
                    if content:
                        if cell.column_header:
                            parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.CHED))
                        elif cell.row_header:
                            parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.RHED))
                        elif cell.row_section:
                            parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.SROW))
                        else:
                            parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.FCEL))

                        if cell_loc:
                            parts.append(cell_loc)
                        if ContentType.TABLE_CELL in params.content_types:
                            # Apply XML escaping to table cell content
                            if not isinstance(cell, RichTableCell):
                                content = _escape_text(content, params)
                                # Wrap in <text> tags unless use_virtual_texts is True
                                if not params.use_virtual_texts:
                                    content = _wrap(text=content, wrap_tag=DoclangToken.TEXT.value)
                            parts.append(content)
                    else:
                        parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.ECEL))
                elif rowstart != i and colspan == 1:  # FIXME: I believe we should have colstart == j
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.UCEL))
                elif colstart != j and rowspan == 1:  # FIXME: I believe we should have rowstart == i
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.LCEL))
                else:
                    parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.XCEL))

            parts.append(DoclangVocabulary._create_selfclosing_token(token=DoclangToken.NL))

        return "".join(parts)

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

        # Check the label to distinguish between TABLE and DOCUMENT_INDEX label
        open_token: str = DoclangVocabulary._create_group_token()
        close_token: str = DoclangVocabulary._create_group_token(closing=True)

        res_parts: list[SerializationResult] = []

        # Caption as sibling of the OTSL payload within the floating group
        caption_text = ""
        if params.add_referenced_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                caption_text = cap_res.text
                res_parts.append(cap_res)

        # Build table payload: location (if any) + OTSL content inside <otsl> ... </otsl>
        otsl_payload = ""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""
            if params.add_location:
                body += _create_location_tokens_for_item(item=item, doc=doc, xres=params.xsize, yres=params.ysize)

            if layer_token := _create_layer_token(item=item, params=params):
                body += layer_token

            if ContentType.TABLE in params.content_types:
                otsl_text = self._emit_otsl(
                    item=item,
                    doc_serializer=doc_serializer,
                    doc=doc,
                    params=params,
                    visited=visited,
                    **kwargs,
                )
                body += otsl_text
            if body:
                # Add class="index" attribute for DOCUMENT_INDEX tables
                from docling_core.types.doc.labels import DocItemLabel

                if item.label == DocItemLabel.DOCUMENT_INDEX:
                    table_open = f'<{DoclangToken.TABLE.value} {DoclangAttributeKey.CLASS.value}="{DoclangAttributeValue.INDEX.value}">'
                    table_close = f"</{DoclangToken.TABLE.value}>"
                    otsl_payload = f"{table_open}{body}{table_close}"
                else:
                    otsl_payload = _wrap(text=body, wrap_tag=DoclangToken.TABLE.value)
                res_parts.append(create_ser_result(text=body, span_source=item))

        # Footnote as sibling of the OTSL payload within the floating group
        footnote_text = ""
        if params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        # Emit a bare <table> only when it is the sole non-empty element.
        parts = [part for part in (caption_text, otsl_payload, footnote_text) if part]
        if not parts:
            if params.suppress_empty_elements:
                return create_ser_result()
            text_res = f"{open_token}{close_token}"
        elif len(parts) == 1 and otsl_payload:
            text_res = otsl_payload
        else:
            text_res = f"{open_token}{''.join(parts)}{close_token}"

        return create_ser_result(text=text_res, span_source=res_parts)


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
        should_wrap = parent_item is None or isinstance(parent_item, ListItem) or not isinstance(parent_item, TextItem)
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
            # add any location tokens for FieldRegionItem
            if (is_fri := isinstance(item, FieldRegionItem)) and params.add_location:
                loc_str = _create_location_tokens_for_item(item=item, doc=doc, xres=params.xsize, yres=params.ysize)
                if loc_str:
                    parts.append(create_ser_result(text=loc_str, span_source=item))
            if is_fri:
                if layer_token := _create_layer_token(item=item, params=params):
                    parts.append(create_ser_result(text=layer_token, span_source=item))
            parts.extend(doc_serializer.get_parts(item=item, **kwargs))
            text_res = delim.join([p.text for p in parts if p.text])
            tok = DoclangToken.FIELD_REGION if is_fri else DoclangToken.FIELD_ITEM
            text_res = _wrap(text=text_res, wrap_tag=tok.value)
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
            if cap_res.text and params.add_location:
                for caption in item.captions:
                    if caption.cref not in self.get_excluded_refs(**kwargs):
                        if isinstance(cap := caption.resolve(self.doc), DocItem):
                            loc_txt = _create_location_tokens_for_item(
                                item=cap,
                                doc=self.doc,
                                xres=params.xsize,
                                yres=params.ysize,
                            )
                            results.append(create_ser_result(text=loc_txt))

                            if layer_token := _create_layer_token(item=cap, params=params):
                                results.append(create_ser_result(text=layer_token))
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
                    location = ""
                    if params.add_location:
                        location = _create_location_tokens_for_item(
                            item=ftn, doc=self.doc, xres=params.xsize, yres=params.ysize
                        )

                    layer_token = _create_layer_token(item=ftn, params=params)

                    content = ""
                    if ftn.text and ContentType.REF_FOOTNOTE in params.content_types:
                        content = _escape_text(ftn.text, params)

                    text_res = f"{location}{layer_token}{content}"
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
        return tag in {DoclangToken.CONTENT.value, DoclangToken.META.value}

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
            version=DOCLANG_VERSION if self.params.include_version else None,
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


class DoclangDeserializer(BaseModel):
    """Doclang deserializer."""

    # Internal state used while walking the tree (private instance attributes)
    _page_no: int = PrivateAttr(default=1)
    _default_resolution: int = PrivateAttr(default=DOCLANG_DFLT_RESOLUTION)

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
        self._ensure_page_exists(doc=doc, page_no=self._page_no, resolution=self._default_resolution)
        self._parse_document_root(doc=doc, root=root)
        return doc

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
        elif name == DoclangToken.LIST.value:
            self._parse_list(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.GROUP.value:
            self._parse_group(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.TABLE.value:
            self._parse_table_group(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.PICTURE.value:
            self._parse_picture_group(doc=doc, el=el, parent=parent)
        else:
            self._walk_children(doc=doc, el=el, parent=parent)

    def _walk_children(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        for node in el.childNodes:
            if isinstance(node, Element):
                # Ignore geometry/meta containers at this level; pass through page breaks
                if node.tagName in {
                    DoclangToken.HEAD.value,
                    DoclangToken.META.value,
                    DoclangToken.LOCATION.value,
                    DoclangToken.LAYER.value,
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
                if el.tagName not in {
                    DoclangToken.LOCATION.value,
                    DoclangToken.LAYER.value,
                    DoclangToken.BR.value,
                    DoclangToken.BOLD.value,
                    DoclangToken.ITALIC.value,
                    DoclangToken.UNDERLINE.value,
                    DoclangToken.STRIKETHROUGH.value,
                    DoclangToken.SUBSCRIPT.value,
                    DoclangToken.SUPERSCRIPT.value,
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
            node
            for node in el.childNodes
            if isinstance(node, Element) and node.tagName not in {DoclangToken.LOCATION.value, DoclangToken.LAYER.value}
        ]

        if len(element_children) > 1 or self._get_children_simple_text_block(el) is None:
            self._parse_inline_group(doc=doc, el=el, parent=parent)
            return

        prov_list = self._extract_provenance(doc=doc, el=el)
        content_layer = self._extract_layer(el=el)
        text, formatting = self._extract_text_with_formatting(el)
        if not text:
            return

        nm = el.tagName

        # Handle code separately (language + content extraction)
        if nm == DoclangToken.CODE.value:
            code_text, lang_label = self._extract_code_content_and_language(el)
            if not code_text.strip():
                return
            item = doc.add_code(
                text=code_text,
                code_language=lang_label,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                content_layer=content_layer,
            )
            for p in prov_list[1:]:
                item.prov.append(p)

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
            item = doc.add_text(
                label=label,
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
                content_layer=content_layer,
            )
            for p in prov_list[1:]:
                item.prov.append(p)

        elif nm == DoclangToken.FORMULA.value:
            item = doc.add_formula(
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            for p in prov_list[1:]:
                item.prov.append(p)

    def _extract_code_content_and_language(self, el: Element) -> tuple[str, CodeLanguageLabel]:
        """Extract code content and language from a <code> element."""
        try:
            linguist_lang = _LinguistLabel(el.getAttribute(DoclangAttributeKey.CLASS.value))
            lang_label = _LinguistLabel.to_code_language_label(linguist_lang)
        except ValueError:
            lang_label = CodeLanguageLabel.UNKNOWN
        parts: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                if node.data.strip():
                    parts.append(node.data)
            elif isinstance(node, Element):
                nm_child = node.tagName
                if nm_child == DoclangToken.LOCATION.value:
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
            for p in prov_list[1:]:
                item.prov.append(p)

    def _has_virtual_text_content(self, nodes: Any) -> bool:
        """Check if a list of nodes contains raw text or <content> elements.

        This indicates virtual text mode where content should be treated as a text element.

        Args:
            nodes: List of DOM nodes (can be Text or Element nodes)
        """
        for node in nodes:
            if isinstance(node, Text):
                # Check if it's not just whitespace
                if node.data.strip():
                    return True
            elif isinstance(node, Element) and node.tagName == DoclangToken.CONTENT.value:
                return True
        return False

    def _parse_list(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        ordered = el.getAttribute(DoclangAttributeKey.CLASS.value) == DoclangAttributeValue.ORDERED.value
        li_group = doc.add_list_group(parent=parent)
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

            # Get all nodes between ldivs (including Text nodes)
            if next_ldiv_index is not None:
                all_content_nodes = list(el.childNodes)[ldiv_index_in_all + 1 : next_ldiv_index]
            else:
                all_content_nodes = list(el.childNodes)[ldiv_index_in_all + 1 :]

            # Filter out location elements from all_content_nodes
            all_content_nodes = [
                n
                for n in all_content_nodes
                if not (isinstance(n, Element) and n.tagName == DoclangToken.LOCATION.value)
            ]

            # Content elements come after the ldiv (start+1 to end) - only Element nodes
            content_elements = actual_children[start + 1 : end]

            # Check if we have virtual text content (raw text or <content> elements)
            has_virtual_text = self._has_virtual_text_content(all_content_nodes)

            if len(content_elements) == 0 and not has_virtual_text:
                # Empty list item (just ldiv, no content)
                doc.add_list_item(
                    text="",
                    parent=li_group,
                    enumerated=ordered,
                    marker=marker_text,
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
                    and all(isinstance(el, Element) and el.tagName == DoclangToken.LIST.value for el in remaining_els)
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
        nodes: Optional[list[Node]] = None,
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

    # ------------- Groups -------------
    def _parse_group(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        # Check for legacy class attribute for backward compatibility
        cls_val = el.getAttribute(DoclangAttributeKey.CLASS.value)
        if cls_val == DoclangAttributeValue.TABLE.value:
            self._parse_table_group(doc=doc, el=el, parent=parent)
        elif cls_val == DoclangAttributeValue.PICTURE.value:
            self._parse_picture_group(doc=doc, el=el, parent=parent)
        else:
            # For new format without class attribute, determine type by children
            # Check if it contains a <table> or <picture> element
            has_table = self._first_child(el, DoclangToken.TABLE.value) is not None
            has_picture = self._first_child(el, DoclangToken.PICTURE.value) is not None

            if has_table:
                self._parse_table_group(doc=doc, el=el, parent=parent)
            elif has_picture:
                self._parse_picture_group(doc=doc, el=el, parent=parent)
            else:
                self._walk_children(doc=doc, el=el, parent=parent)

    def _parse_table_group(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        from docling_core.types.doc.labels import DocItemLabel

        otsl_el: Optional[Element]
        if el.tagName == DoclangToken.TABLE.value:
            caption = None
            footnotes = []
            otsl_el = el
        else:
            caption = self._extract_caption(doc=doc, el=el)
            footnotes = self._extract_footnotes(doc=doc, el=el)
            otsl_el = self._first_child(el, DoclangToken.TABLE.value)
            if otsl_el is None:
                tbl = doc.add_table(data=TableData(), caption=caption, parent=parent)
                for ftn in footnotes:
                    tbl.footnotes.append(ftn.get_ref())
                return

        # Check for class attribute to determine table type
        table_label = DocItemLabel.TABLE
        if otsl_el:
            cls_val = otsl_el.getAttribute(DoclangAttributeKey.CLASS.value)
            if cls_val:
                # Validate that class value is one of the allowed values
                if cls_val == DoclangAttributeValue.INDEX.value:
                    table_label = DocItemLabel.DOCUMENT_INDEX
                elif cls_val == DoclangAttributeValue.DATA.value:
                    table_label = DocItemLabel.TABLE
                else:
                    raise ValueError(
                        f"Invalid class attribute value '{cls_val}' for table element. "
                        f"Allowed values are: '{DoclangAttributeValue.INDEX.value}', '{DoclangAttributeValue.DATA.value}'"
                    )
            # If no class attribute, default to TABLE

        # Extract table provenance from <otsl> leading <location/> tokens
        tbl_provs = self._extract_provenance(doc=doc, el=otsl_el)
        content_layer = self._extract_layer(el=otsl_el)
        # Get inner XML excluding location and layer tokens (work directly with parsed DOM)
        inner = self._inner_xml(otsl_el, exclude_tags={"location", "layer"})
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
        for ftn in footnotes:
            tbl.footnotes.append(ftn.get_ref())

    def _parse_picture_group(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        picture_el: Optional[Element]
        if el.tagName == DoclangToken.PICTURE.value:
            caption = None
            footnotes = []
            picture_el = el
        else:
            # Extract caption from the floating group
            caption = self._extract_caption(doc=doc, el=el)
            footnotes = self._extract_footnotes(doc=doc, el=el)
            picture_el = self._first_child(el, DoclangToken.PICTURE.value)

        # Extract provenance and layer from the <picture> block (locations and layer appear inside it)
        prov_list: list[ProvenanceItem] = []
        content_layer: Optional[ContentLayer] = None
        if picture_el is not None:
            prov_list = self._extract_provenance(doc=doc, el=picture_el)
            content_layer = self._extract_layer(el=picture_el)

        # Create the picture item first, attach caption and provenance
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

        # If there is a <picture> child and it contains an <otsl>,
        # parse it as TabularChartMetaField and attach to picture.meta
        if picture_el is not None:
            otsl_el = self._first_child(picture_el, DoclangToken.TABLE.value)
            if otsl_el is not None:
                inner = self._inner_xml(otsl_el, exclude_tags={"location"})
                td = self._parse_otsl_table_content(_wrap(inner, DoclangToken.TABLE.value))
                if pic.meta is None:
                    pic.meta = PictureMeta()
                pic.meta.tabular_chart = TabularChartMetaField(chart_data=td)

    # ------------- Helpers -------------
    def _extract_caption(self, *, doc: DoclingDocument, el: Element) -> Optional[TextItem]:
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

    # --------- OTSL table parsing (inlined) ---------
    def _otsl_extract_tokens_and_text(self, s: str) -> tuple[list[str], list[str]]:
        """Extract OTSL structural tokens and interleaved text.

        Strips the outer <otsl> wrapper and ignores location tokens (expected
        to be removed before). Handles nested XML elements (like <text><italic>...</italic></text>)
        by keeping them as single units.
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

        # Clean tokens to only structural OTSL markers
        clean_tokens: list[str] = []
        for t in tokens:
            if t in [ecel, fcel, lcel, ucel, xcel, nl, ched, rhed, srow]:
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

        for i, t in enumerate(texts):
            cell_text = ""
            if t in [fcel, ecel, ched, rhed, srow]:
                row_span = 1
                col_span = 1
                right_offset = 1
                if t != ecel and (i + 1) < len(texts):
                    cell_text = texts[i + 1]
                    right_offset = 2

                next_right = texts[i + right_offset] if i + right_offset < len(texts) else ""
                next_bottom = (
                    split_row_tokens[r_idx + 1][c_idx]
                    if (r_idx + 1) < len(split_row_tokens) and c_idx < len(split_row_tokens[r_idx + 1])
                    else ""
                )

                if next_right in [lcel, xcel]:
                    col_span += count_right(split_row_tokens, c_idx + 1, r_idx, [lcel, xcel])
                if next_bottom in [ucel, xcel]:
                    row_span += count_down(split_row_tokens, c_idx, r_idx + 1, [ucel, xcel])

                # Check if cell_text contains XML content (rich cell)
                cell_text_stripped = cell_text.strip()
                cell_added = False
                if (
                    cell_text_stripped.startswith("<")
                    and cell_text_stripped.endswith(">")
                    and doc is not None
                    and parent is not None
                ):
                    # Wrap in a root element to ensure valid XML
                    wrapped_xml = f"<root>{cell_text_stripped}</root>"
                    dom = parseString(wrapped_xml)
                    root_el = dom.documentElement

                    if root_el is None:
                        raise ValueError("No document element found")

                    # Get the number of children before parsing
                    children_before = len(parent.children)

                    # Parse the child elements and create document items
                    parsed_element = None
                    for child_node in root_el.childNodes:
                        if isinstance(child_node, Element):
                            parsed_element = child_node
                            # Dispatch to parse this element (creates items as side effect)
                            self._dispatch_element(doc=doc, el=child_node, parent=parent)
                            break  # Only process first element

                    # Check if a new child was added
                    if len(parent.children) > children_before:
                        # Get the newly created item
                        child_item = parent.children[-1].resolve(doc=doc)
                        # Extract the actual text content from the parsed element
                        actual_text = self._get_text(parsed_element) if parsed_element else cell_text_stripped
                        # Create a RichTableCell with reference to the parsed content
                        table_cells.append(
                            RichTableCell(
                                text=actual_text,
                                row_span=row_span,
                                col_span=col_span,
                                start_row_offset_idx=r_idx,
                                end_row_offset_idx=r_idx + row_span,
                                start_col_offset_idx=c_idx,
                                end_col_offset_idx=c_idx + col_span,
                                ref=child_item.get_ref(),
                            )
                        )
                        cell_added = True

                if not cell_added:
                    # Regular text cell
                    table_cells.append(
                        TableCell(
                            text=cell_text_stripped,
                            row_span=row_span,
                            col_span=col_span,
                            start_row_offset_idx=r_idx,
                            end_row_offset_idx=r_idx + row_span,
                            start_col_offset_idx=c_idx,
                            end_col_offset_idx=c_idx + col_span,
                        )
                    )

            if t in [fcel, ecel, ched, rhed, srow, lcel, ucel, xcel]:
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
    ) -> TableData:
        """Parse OTSL content into TableData (inlined from utils)."""
        tokens, mixed = self._otsl_extract_tokens_and_text(otsl_content)
        table_cells, split_rows = self._otsl_parse_texts(mixed, tokens, doc=doc, parent=parent)
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
        # Collect immediate child <location value=.. resolution=.. /> tokens in groups of 4
        values: list[int] = []
        res_for_group: Optional[int] = None
        provs: list[ProvenanceItem] = []

        for node in el.childNodes:
            if not isinstance(node, Element):
                continue
            if node.tagName != DoclangToken.LOCATION.value:
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
            # For a group, remember the last seen resolution
            res_for_group = r
            if len(values) == 4:
                # Ensure page exists (and set consistent default size for this page)
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

    def _extract_layer(self, *, el: Element) -> Optional[ContentLayer]:
        """Extract content layer from <layer class="..."/> token if present."""
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == DoclangToken.LAYER.value:
                if layer_value := node.getAttribute(DoclangAttributeKey.CLASS.value):
                    try:
                        return ContentLayer(layer_value)
                    except ValueError:
                        pass
        return None
