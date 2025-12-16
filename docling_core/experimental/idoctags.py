"""Define classes for DocTags serialization."""

import copy
import html
import re
from enum import Enum
from html.parser import HTMLParser
from itertools import groupby
from typing import Any, ClassVar, Final, Optional, cast
from xml.dom.minidom import Element, Text, parseString

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
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.labels import CodeLanguageLabel, DocItemLabel

# Note: Intentionally avoid importing DocumentToken here to ensure
# IDocTags uses only its own token vocabulary.

DOCTAGS_VERSION: Final = "1.0.0"
DOCTAGS_RESOLUTION: int = 512


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


def _wrap_token(*, text: str, open_token: str) -> str:
    close_token = IDocTagsVocabulary.create_closing_token(token=open_token)
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
    """Quantize normalized value in [0,1] to [0,resolution]."""
    n = int(round(resolution * value))
    if n < 0:
        return 0
    if n > resolution:
        return resolution
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
        IDocTagsVocabulary.create_location_token(value=x0v, resolution=xres)
        + IDocTagsVocabulary.create_location_token(value=y0v, resolution=yres)
        + IDocTagsVocabulary.create_location_token(value=x1v, resolution=xres)
        + IDocTagsVocabulary.create_location_token(value=y1v, resolution=yres)
    )


def _create_location_tokens_for_item(
    *,
    item: "DocItem",
    doc: "DoclingDocument",
    xres: int = DOCTAGS_RESOLUTION,
    yres: int = DOCTAGS_RESOLUTION,
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
                bbox=bbox, page_w=page_w, page_h=page_h, xres=xres, yres=yres
            )
        )

    # In a proper serialization, we should use <thread id="1|2|3|...|"/> to link different
    # sections together ...
    if len(out) > 1:
        res = []
        for i, _ in enumerate(item.prov):
            res.append(f"{i} {_}")
        err = "\n".join(res)

        raise ValueError(
            f"We have more than 1 location for this item [{item.label}]:\n\n{err}\n\n{out}"
        )

    return "".join(out)


class IDocTagsCategory(str, Enum):
    """IDocTagsCtegory.

    DocTags defines the following categories of elements:

    - **root**: Elements that establish document scope such as
      `doctag`.
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
      (e.g., titles, paragraphs, captions, lists, forms, tables, formulas,
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
    BINARY_DATA = "binary_data"
    CONTINUATION = "continuation"


class IDocTagsToken(str, Enum):
    """IDocTagsToken.

    This class implements the tokens from the Token table,

    | # | Category | Token | Self-Closing [Yes/No] | Parametrized [Yes/No] | Attributes | Description |
    |---|----------|-------|-----------------------|-----------------------|------------|-------------|
    | 1 | Root Elements | `doctag` | No | Yes | `version` | Root container; optional semantic version `version`. |
    | 2 | Special Elements | `page_break` | Yes | No | — | Page delimiter. |
    | 3 |  | `time_break` | Yes | No | — | Temporal segment delimiter. |
    | 4 | Metadata Containers | `head` | No | No | — | Document-level metadata container. |
    | 5 |  | `meta` | No | No | — | Component-level metadata container. |
    | 6 | Geometric Tokens | `location` | Yes | Yes | `value`, `resolution?` |
      Geometric coordinate; `value` in [0, res]; optional `resolution`. |
    | 7 | Temmporal Tokens | `hour` | Yes | Yes | `value` | Hours component; `value` in [0, 99]. |
    | 8 |  | `minute` | Yes | Yes | `value` | Minutes component; `value` in [0, 59]. |
    | 9 |  | `second` | Yes | Yes | `value` | Seconds component; `value` in [0, 59]. |
    | 10 |  | `centisecond` | Yes | Yes | `value` | Centiseconds component; `value` in [0, 99]. |
    | 11 | Semantic Tokens | `title` | No | No | — | Document or section title (content). |
    | 12 |  | `heading` | No | Yes | `level` | Section header; `level` (N ≥ 1). |
    | 13 |  | `text` | No | No | — | Generic text content. |
    | 14 |  | `caption` | No | No | — | Caption for floating/grouped elements. |
    | 15 |  | `footnote` | No | No | — | Footnote content. |
    | 16 |  | `page_header` | No | No | — | Page header content. |
    | 17 |  | `page_footer` | No | No | — | Page footer content. |
    | 18 |  | `watermark` | No | No | — | Watermark indicator or content. |
    | 19 |  | `picture` | No | No | — |
      Block image/graphic; at most one of `base64`/`uri`; may include `meta`
      for classification; `otsl` may encode chart data. |
    | 20 |  | `form` | No | No | — | Form structure container. |
    | 21 |  | `formula` | No | No | — | Mathematical expression block. |
    | 22 |  | `code` | No | No | — | Code block. |
    | 23 |  | `list_text` | No | No | — | List item content. |
    | 24 |  | `checkbox` | No | Yes | `selected` |
      Checkbox item; optional `selected` in {`true`,`false`} defaults to
      `false`. |
    | 25 |  | `form_item` | No | No | — |
      Form item; exactly one `key`; one or more of
      `value`/`checkbox`/`marker`/`hint`. |
    | 26 |  | `form_heading` | No | Yes | `level?` | Form header; optional `level` (N ≥ 1). |
    | 27 |  | `form_text` | No | No | — | Form text block. |
    | 28 |  | `hint` | No | No | — | Hint for a fillable field (format/example/description). |
    | 29 | Grouping Tokens | `section` | No | Yes | `level` | Document section; `level` (N ≥ 1). |
    | 30 |  | `list` | No | Yes | `ordered` |
      List container; optional `ordered` in {`true`,`false`} defaults to
      `false`. |
    | 31 |  | `group` | No | Yes | `type?` |
      Generic group; no `location` tokens; associates composite content
      (e.g., captions/footnotes). |
    | 32 |  | `floating_group` | No | Yes | `class` in {`table`,`picture`,`form`,`code`} |
      Floating container that groups a floating component with its caption,
      footnotes, and metadata; no `location` tokens. |
    | 33 | Formatting Tokens | `bold` | No | No | — | Bold text. |
    | 34 |  | `italic` | No | No | — | Italic text. |
    | 35 |  | `strikethrough` | No | No | — | Strike-through text. |
    | 36 |  | `superscript` | No | No | — | Superscript text. |
    | 37 |  | `subscript` | No | No | — | Subscript text. |
    | 38 |  | `rtl` | No | No | — | Right-to-left text direction. |
    | 39 |  | `inline` | No | Yes | `class` in {`formula`,`code`,`picture`} |
      Inline content; if `class="picture"`, may include one of `base64` or
      `uri`. |
    | 40 |  | `br` | Yes | No | — | Line break. |
    | 41 | Structural Tokens (OTSL) | `otsl` | No | No | — | Table structure container. |
    | 42 |  | `fcel` | Yes | No | — | New cell with content. |
    | 43 |  | `ecel` | Yes | No | — | New cell without content. |
    | 44 |  | `ched` | Yes | No | — | Column header cell. |
    | 45 |  | `rhed` | Yes | No | — | Row header cell. |
    | 46 |  | `corn` | Yes | No | — | Corner header cell. |
    | 47 |  | `srow` | Yes | No | — | Section row separator cell. |
    | 48 |  | `lcel` | Yes | No | — | Merge with left neighbor (horizontal span). |
    | 49 |  | `ucel` | Yes | No | — | Merge with upper neighbor (vertical span). |
    | 50 |  | `xcel` | Yes | No | — | Merge with left and upper neighbors (2D span). |
    | 51 |  | `nl` | Yes | No | — | New line (row separator). |
    | 52 | Continuation Tokens | `thread` | Yes | Yes | `id` |
      Continuation marker for split content; reuse same `id` across parts. |
    | 53 |  | `h_thread` | Yes | Yes | `id` | Horizontal stitching marker for split tables; reuse same `id`. |
    | 54 | Binary Data Tokens | `base64` | No | No | — | Embedded binary data (base64). |
    | 55 |  | `uri` | No | No | — | External resource reference. |
    | 56 | Content Tokens | `marker` | No | No | — | List/form marker content. |
    | 57 |  | `facets` | No | No | — | Container for application-specific derived properties. |
    | 58 | Structural Tokens (Form) | `key` | No | No | — | Form item key (child of `form_item`). |
    | 59 |  | `value` | No | No | — | Form item value (child of `form_item`). |
    """

    # Root and metadata
    DOCUMENT = "doctag"
    HEAD = "head"
    META = "meta"

    # Special
    PAGE_BREAK = "page_break"
    TIME_BREAK = "time_break"

    # Geometric and temporal
    LOCATION = "location"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    CENTISECOND = "centisecond"

    # Semantic
    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    WATERMARK = "watermark"
    PICTURE = "picture"
    FORM = "form"
    FORM_ITEM = "form_item"
    FORM_HEADING = "form_heading"
    FORM_TEXT = "form_text"
    HINT = "hint"
    FORMULA = "formula"
    CODE = "code"
    LIST_TEXT = "list_text"
    CHECKBOX = "checkbox"
    OTSL = "otsl"  # this will take care of the structure in the table.

    # Grouping
    SECTION = "section"
    LIST = "list"
    GROUP = "group"
    FLOATING_GROUP = "floating_group"
    INLINE = "inline"

    # Formatting
    BOLD = "bold"  # instead of "strong"
    ITALIC = "italic"  # instead of "em"
    STRIKETHROUGH = "strikethrough"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"

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
    # -- Forms
    KEY = "key"
    IMPLICIT_KEY = "implicit_key"
    VALUE = "value"

    # Continuation
    THREAD = "thread"
    H_THREAD = "h_thread"

    # Binary data / content helpers
    BASE64 = "base64"
    URI = "uri"
    MARKER = "marker"
    FACETS = "facets"


class IDocTagsAttributeKey(str, Enum):
    """Attribute keys allowed on DocTags tokens."""

    VERSION = "version"
    VALUE = "value"
    RESOLUTION = "resolution"
    LEVEL = "level"
    SELECTED = "selected"
    ORDERED = "ordered"
    TYPE = "type"
    CLASS = "class"
    ID = "id"


class IDocTagsAttributeValue(str, Enum):
    """Enumerated values for specific DocTags attributes."""

    # Generic boolean-like values
    TRUE = "true"
    FALSE = "false"

    # Inline class values
    FORMULA = "formula"
    CODE = "code"
    PICTURE = "picture"

    # Floating group class values
    DOCUMENT_INDEX = "document_index"
    TABLE = "table"
    FORM = "form"


class IDocTagsVocabulary(BaseModel):
    """IDocTagsVocabulary."""

    # Allowed attributes per token (defined outside the Enum to satisfy mypy)
    ALLOWED_ATTRIBUTES: ClassVar[dict[IDocTagsToken, set["IDocTagsAttributeKey"]]] = {
        IDocTagsToken.DOCUMENT: {
            IDocTagsAttributeKey.VERSION,
        },
        IDocTagsToken.LOCATION: {
            IDocTagsAttributeKey.VALUE,
            IDocTagsAttributeKey.RESOLUTION,
        },
        IDocTagsToken.HOUR: {IDocTagsAttributeKey.VALUE},
        IDocTagsToken.MINUTE: {IDocTagsAttributeKey.VALUE},
        IDocTagsToken.SECOND: {IDocTagsAttributeKey.VALUE},
        IDocTagsToken.CENTISECOND: {IDocTagsAttributeKey.VALUE},
        IDocTagsToken.HEADING: {IDocTagsAttributeKey.LEVEL},
        IDocTagsToken.FORM_HEADING: {IDocTagsAttributeKey.LEVEL},
        IDocTagsToken.CHECKBOX: {IDocTagsAttributeKey.SELECTED},
        IDocTagsToken.SECTION: {IDocTagsAttributeKey.LEVEL},
        IDocTagsToken.LIST: {IDocTagsAttributeKey.ORDERED},
        IDocTagsToken.GROUP: {IDocTagsAttributeKey.TYPE},
        IDocTagsToken.FLOATING_GROUP: {IDocTagsAttributeKey.CLASS},
        IDocTagsToken.INLINE: {IDocTagsAttributeKey.CLASS},
        IDocTagsToken.THREAD: {IDocTagsAttributeKey.ID},
        IDocTagsToken.H_THREAD: {IDocTagsAttributeKey.ID},
    }

    # Allowed values for specific attributes (enumerations)
    # Structure: token -> attribute name -> set of allowed string values
    ALLOWED_ATTRIBUTE_VALUES: ClassVar[
        dict[
            IDocTagsToken,
            dict["IDocTagsAttributeKey", set["IDocTagsAttributeValue"]],
        ]
    ] = {
        # Grouping and inline enumerations
        IDocTagsToken.LIST: {
            IDocTagsAttributeKey.ORDERED: {
                IDocTagsAttributeValue.TRUE,
                IDocTagsAttributeValue.FALSE,
            }
        },
        IDocTagsToken.CHECKBOX: {
            IDocTagsAttributeKey.SELECTED: {
                IDocTagsAttributeValue.TRUE,
                IDocTagsAttributeValue.FALSE,
            }
        },
        IDocTagsToken.INLINE: {
            IDocTagsAttributeKey.CLASS: {
                IDocTagsAttributeValue.FORMULA,
                IDocTagsAttributeValue.CODE,
                IDocTagsAttributeValue.PICTURE,
            }
        },
        IDocTagsToken.FLOATING_GROUP: {
            IDocTagsAttributeKey.CLASS: {
                IDocTagsAttributeValue.DOCUMENT_INDEX,
                IDocTagsAttributeValue.TABLE,
                IDocTagsAttributeValue.PICTURE,
                IDocTagsAttributeValue.FORM,
                IDocTagsAttributeValue.CODE,
            }
        },
        # Other attributes (e.g., level, type, id) are not enumerated here
    }

    ALLOWED_ATTRIBUTE_RANGE: ClassVar[
        dict[IDocTagsToken, dict["IDocTagsAttributeKey", tuple[int, int]]]
    ] = {
        # Geometric: value in [0, res]; resolution optional.
        # Keep conservative defaults aligned with existing usage.
        IDocTagsToken.LOCATION: {
            IDocTagsAttributeKey.VALUE: (0, DOCTAGS_RESOLUTION),
            IDocTagsAttributeKey.RESOLUTION: (DOCTAGS_RESOLUTION, DOCTAGS_RESOLUTION),
        },
        # Temporal components
        IDocTagsToken.HOUR: {IDocTagsAttributeKey.VALUE: (0, 99)},
        IDocTagsToken.MINUTE: {IDocTagsAttributeKey.VALUE: (0, 59)},
        IDocTagsToken.SECOND: {IDocTagsAttributeKey.VALUE: (0, 59)},
        IDocTagsToken.CENTISECOND: {IDocTagsAttributeKey.VALUE: (0, 99)},
        # Levels (N ≥ 1)
        IDocTagsToken.HEADING: {IDocTagsAttributeKey.LEVEL: (1, 6)},
        IDocTagsToken.FORM_HEADING: {IDocTagsAttributeKey.LEVEL: (1, 6)},
        IDocTagsToken.SECTION: {IDocTagsAttributeKey.LEVEL: (1, 6)},
        # Continuation markers (id length constraints)
        IDocTagsToken.THREAD: {IDocTagsAttributeKey.ID: (1, 10)},
        IDocTagsToken.H_THREAD: {IDocTagsAttributeKey.ID: (1, 10)},
    }

    # Self-closing tokens map
    IS_SELFCLOSING: ClassVar[dict[IDocTagsToken, bool]] = {
        IDocTagsToken.PAGE_BREAK: True,
        IDocTagsToken.TIME_BREAK: True,
        IDocTagsToken.LOCATION: True,
        IDocTagsToken.HOUR: True,
        IDocTagsToken.MINUTE: True,
        IDocTagsToken.SECOND: True,
        IDocTagsToken.CENTISECOND: True,
        IDocTagsToken.BR: True,
        # OTSL structural tokens are emitted as self-closing markers
        IDocTagsToken.FCEL: True,
        IDocTagsToken.ECEL: True,
        IDocTagsToken.CHED: True,
        IDocTagsToken.RHED: True,
        IDocTagsToken.CORN: True,
        IDocTagsToken.SROW: True,
        IDocTagsToken.LCEL: True,
        IDocTagsToken.UCEL: True,
        IDocTagsToken.XCEL: True,
        IDocTagsToken.NL: True,
        # Continuation markers
        IDocTagsToken.THREAD: True,
        IDocTagsToken.H_THREAD: True,
    }

    @classmethod
    def create_closing_token(cls, *, token: str) -> str:
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
                IDocTagsToken(name)
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
            tok_enum = IDocTagsToken(name)
        except ValueError:
            raise ValueError(f"unknown token '{name}'")

        # Disallow explicit self-closing markup or inherently self-closing tokens
        if trailing_slash == "/":
            raise ValueError(f"token '{name}' is self-closing; no closing tag")
        if cls.IS_SELFCLOSING.get(tok_enum, False):
            raise ValueError(f"token '{name}' is self-closing; no closing tag")

        return f"</{name}>"

    @classmethod
    def create_doctag_root(
        cls, *, version: str = DOCTAGS_VERSION, closing: bool = False
    ) -> str:
        """Create the document root tag.

        - When `closing` is True, returns the closing root tag.
        - When a `version` is provided, includes it as an attribute.
        - Otherwise returns a bare opening root tag.
        """
        if closing:
            return f"</{IDocTagsToken.DOCUMENT.value}>"
        elif version:
            return f'<{IDocTagsToken.DOCUMENT.value} {IDocTagsAttributeKey.VERSION.value}="{version}">'
        else:
            # Version attribute is optional; emit bare root tag when not provided
            return f"<{IDocTagsToken.DOCUMENT.value}>"

    @classmethod
    def create_threading_token(cls, *, id: str, horizontal: bool = False) -> str:
        """Create a continuation threading token.

        Emits `<thread id="..."/>` or `<h_thread id="..."/>` depending on
        the `horizontal` flag. Validates required attributes against the
        class schema and basic value sanity.
        """
        token = IDocTagsToken.H_THREAD if horizontal else IDocTagsToken.THREAD
        # Ensure the required attribute is declared for this token
        assert IDocTagsAttributeKey.ID in cls.ALLOWED_ATTRIBUTES.get(token, set())

        # Validate id length if a range is specified
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token][IDocTagsAttributeKey.ID]
        length = len(id)
        if not (lo <= length <= hi):
            raise ValueError(f"id length must be in [{lo}, {hi}]")

        return f'<{token.value} {IDocTagsAttributeKey.ID.value}="{id}"/>'

    @classmethod
    def create_floating_group_token(
        cls, *, value: IDocTagsAttributeValue, closing: bool = False
    ) -> str:
        """Create a floating group tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag with a class attribute derived from `value`.
        """
        if closing:
            return f"</{IDocTagsToken.FLOATING_GROUP.value}>"
        else:
            return f'<{IDocTagsToken.FLOATING_GROUP.value} {IDocTagsAttributeKey.CLASS.value}="{value.value}">'

    @classmethod
    def create_list_token(cls, *, ordered: bool, closing: bool = False) -> str:
        """Create a list tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag with an `ordered` boolean attribute.
        """
        if closing:
            return f"</{IDocTagsToken.LIST.value}>"
        elif ordered:
            return (
                f"<{IDocTagsToken.LIST.value} "
                f'{IDocTagsAttributeKey.ORDERED.value}="{IDocTagsAttributeValue.TRUE.value}">'
            )
        else:
            return (
                f"<{IDocTagsToken.LIST.value} "
                f'{IDocTagsAttributeKey.ORDERED.value}="{IDocTagsAttributeValue.FALSE.value}">'
            )

    @classmethod
    def create_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a heading tag with validated level.

        When `closing` is False, emits an opening tag with level attribute.
        When `closing` is True, emits the corresponding closing tag.
        """
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[IDocTagsToken.HEADING][
            IDocTagsAttributeKey.LEVEL
        ]
        if not (lo <= level <= hi):
            raise ValueError(f"level must be in [{lo}, {hi}]")

        if closing:
            return f"</{IDocTagsToken.HEADING.value}>"
        return f'<{IDocTagsToken.HEADING.value} {IDocTagsAttributeKey.LEVEL.value}="{level}">'

    @classmethod
    def create_location_token(
        cls, *, value: int, resolution: int = DOCTAGS_RESOLUTION
    ) -> str:
        """Create a location token with value and resolution.

        Validates both attributes using the configured ranges and ensures
        `value` lies within [0, resolution]. Always emits the resolution
        attribute for explicitness.
        """
        range_map = cls.ALLOWED_ATTRIBUTE_RANGE[IDocTagsToken.LOCATION]
        # Validate resolution if a constraint exists
        r_lo, r_hi = range_map.get(
            IDocTagsAttributeKey.RESOLUTION, (resolution, resolution)
        )
        if not (r_lo <= resolution <= r_hi):
            raise ValueError(f"resolution: {resolution} must be in [{r_lo}, {r_hi}]")

        v_lo, v_hi = range_map[IDocTagsAttributeKey.VALUE]
        if not (v_lo <= value <= v_hi):
            raise ValueError(f"value: {value} must be in [{v_lo}, {v_hi}]")
        if not (0 <= value <= resolution):
            raise ValueError(f"value: {value} must be in [0, {resolution}]")

        return (
            f"<{IDocTagsToken.LOCATION.value} "
            f'{IDocTagsAttributeKey.VALUE.value}="{value}" '
            f'{IDocTagsAttributeKey.RESOLUTION.value}="{resolution}"/>'
        )

    @classmethod
    def get_special_tokens(
        cls,
        *,
        include_location_tokens: bool = True,
        include_temporal_tokens: bool = True,
    ) -> list[str]:
        """Return all DocTags special tokens.

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
            IDocTagsToken.HOUR,
            IDocTagsToken.MINUTE,
            IDocTagsToken.SECOND,
            IDocTagsToken.CENTISECOND,
        }

        for token in IDocTagsToken:
            # Optional gating for location/temporal tokens
            if not include_location_tokens and token is IDocTagsToken.LOCATION:
                continue
            if not include_temporal_tokens and token in temporal_tokens:
                continue

            name = token.value
            is_selfclosing = bool(cls.IS_SELFCLOSING.get(token, False))

            # Attribute-aware emission
            attrs = cls.ALLOWED_ATTRIBUTES.get(token, set())
            if attrs:
                # Enumerated attribute values
                enum_map = cls.ALLOWED_ATTRIBUTE_VALUES.get(token, {})
                for attr_name, allowed_vals in enum_map.items():
                    for v in sorted(allowed_vals, key=lambda x: x.value):
                        if is_selfclosing:
                            special_tokens.append(
                                f'<{name} {attr_name.value}="{v.value}"/>'
                            )
                        else:
                            special_tokens.append(
                                f'<{name} {attr_name.value}="{v.value}">'
                            )
                            special_tokens.append(f"</{name}>")

                # Ranged attribute values (emit a conservative, complete range)
                range_map = cls.ALLOWED_ATTRIBUTE_RANGE.get(token, {})
                for attr_name, (lo, hi) in range_map.items():
                    # Keep the list size reasonable by skipping optional resolution enumeration
                    if (
                        token is IDocTagsToken.LOCATION
                        and attr_name is IDocTagsAttributeKey.RESOLUTION
                    ):
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
    def create_selfclosing_token(
        cls,
        *,
        token: IDocTagsToken,
        attrs: Optional[dict["IDocTagsAttributeKey", Any]] = None,
    ) -> str:
        """Create a self-closing token with optional attributes (default None).

        - Validates the token is declared self-closing.
        - Validates provided attributes against ``ALLOWED_ATTRIBUTES`` and
          ``ALLOWED_ATTRIBUTE_VALUES`` or ``ALLOWED_ATTRIBUTE_RANGE`` when present.
        """
        if not cls.IS_SELFCLOSING.get(token, False):
            raise ValueError(f"token '{token.value}' is not self-closing")

        # No attributes requested
        if not attrs:
            return f"<{token.value}/>"

        # Validate attribute keys
        allowed_keys = cls.ALLOWED_ATTRIBUTES.get(token, set())
        for k in attrs.keys():
            if k not in allowed_keys:
                raise ValueError(
                    f"attribute '{getattr(k, 'value', str(k))}' not allowed on '{token.value}'"
                )

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
                    raise ValueError(
                        f"invalid value '{v_norm}' for '{k.value}' on '{token.value}'"
                    )
                parts.append(f'{k.value}="{v_norm}"')
                continue

            # Ranged numeric values
            if k in range_map:
                lo, hi = range_map[k]
                try:
                    v_num = int(v)
                except Exception:
                    raise ValueError(
                        f"attribute '{k.value}' on '{token.value}' must be an integer"
                    )
                if not (lo <= v_num <= hi):
                    raise ValueError(
                        f"attribute '{k.value}' must be in [{lo}, {hi}] for '{token.value}'"
                    )
                parts.append(f'{k.value}="{v_num}"')
                continue

            # Free-form attribute without specific constraints
            parts.append(f'{k.value}="{_coerce_value(v)}"')

        # Assemble tag
        attrs_text = " ".join(parts)
        return f"<{token.value} {attrs_text}/>"


class IDocTagsSerializationMode(str, Enum):
    """Serialization mode for IDocTags output."""

    HUMAN_FRIENDLY = "human_friendly"
    LLM_FRIENDLY = "llm_friendly"


class IDocTagsParams(CommonParams):
    """IDocTags-specific serialization parameters independent of DocTags."""

    # Geometry & content controls (aligned with DocTags defaults)
    xsize: int = DOCTAGS_RESOLUTION
    ysize: int = DOCTAGS_RESOLUTION
    add_location: bool = True
    add_caption: bool = True
    add_content: bool = True
    add_table_cell_location: bool = False
    add_page_break: bool = True

    mode: IDocTagsSerializationMode = IDocTagsSerializationMode.HUMAN_FRIENDLY

    # IDocTags formatting
    do_self_closing: bool = True
    pretty_indentation: Optional[str] = 2 * " "
    # Expand self-closing forms of non-self-closing tokens after pretty-printing
    preserve_empty_non_selfclosing: bool = True
    # XML compliance: escape special characters in text content
    xml_compliant: bool = False


def _get_delim(*, params: IDocTagsParams) -> str:
    """Return record delimiter based on IDocTagsSerializationMode."""
    if params.mode == IDocTagsSerializationMode.HUMAN_FRIENDLY:
        return "\n"
    if params.mode == IDocTagsSerializationMode.LLM_FRIENDLY:
        return ""
    raise RuntimeError(f"Unknown IDocTags mode: {params.mode}")


class _WhitelistHTMLParser(HTMLParser):
    """XML-safe sanitizer that preserves only specific IDocTags formatting and content tags.

    Preserves these tags (attributes are stripped):
    bold, italic, strikethrough, superscript, subscript, inline, text, code, formula, facets.
    All other tags are escaped literally.
    """

    # Allowed formatting and content tags
    _ALLOWED = {
        IDocTagsToken.BOLD.value,
        IDocTagsToken.ITALIC.value,
        IDocTagsToken.STRIKETHROUGH.value,
        IDocTagsToken.SUPERSCRIPT.value,
        IDocTagsToken.SUBSCRIPT.value,
        IDocTagsToken.INLINE.value,
        IDocTagsToken.TEXT.value,
        IDocTagsToken.CODE.value,
        IDocTagsToken.FORMULA.value,
        IDocTagsToken.FACETS.value,
    }

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in self._ALLOWED:
            self.out.append(f"<{tag}>")
        else:
            # Escape disallowed tags literally
            self.out.append(html.escape(self.get_starttag_text(), quote=False))

    def handle_endtag(self, tag):
        if tag in self._ALLOWED:
            self.out.append(f"</{tag}>")
        else:
            self.out.append(html.escape(f"</{tag}>", quote=False))

    def handle_startendtag(self, tag, attrs):
        if tag in self._ALLOWED:
            self.out.append(f"<{tag}></{tag}>")
        else:
            self.out.append(html.escape(self.get_starttag_text(), quote=False))

    def handle_data(self, data):
        self.out.append(html.escape(data, quote=False))

    def handle_entityref(self, name):
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        self.out.append(f"&#{name};")

    def handle_comment(self, data):
        self.out.append(html.escape(f"<!--{data}-->", quote=False))


# def _escape_xml_text(text: str, xml_compliant: bool) -> str:
#     """Escape XML special characters if xml_compliant is enabled."""
#     if xml_compliant:
#         return html.escape(text, quote=False)
#     return text


def _escape_xml_text(text: str, xml_compliant: bool) -> str:
    """Escape text for XML while optionally preserving specific IDocTags formatting tags.

    If xml_compliant=True, preserves only these tags (attributes stripped):
    bold, italic, strikethrough, superscript, subscript, inline, text, code, formula, facets.
    All other tags are escaped. If xml_compliant=False, returns text unchanged.
    """
    if not xml_compliant:
        return text
    parser = _WhitelistHTMLParser()
    parser.feed(text)
    parser.close()
    return "".join(parser.out)


class IDocTagsListSerializer(BaseModel, BaseListSerializer):
    """DocTags-specific list serializer."""

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
        """Serialize a ``ListGroup`` into IDocTags markup.

        This emits list containers (``<ordered_list>``/``<unordered_list>``) and
        serializes children explicitly. Nested ``ListGroup`` items are emitted as
        siblings, and individual list items are not wrapped here. The text
        serializer is responsible for wrapping list item content (as
        ``<list_text>``), so this serializer remains agnostic of item types.

        Args:
            item: The list group to serialize.
            doc_serializer: The document-level serializer to delegate nested items.
            doc: The document that provides item resolution.
            list_level: Current nesting depth (0-based).
            is_inline_scope: Whether serialization happens in an inline context.
            visited: Set of already visited item refs to avoid cycles.
            **kwargs: Additional serializer parameters forwarded to ``IDocTagsParams``.

        Returns:
            A ``SerializationResult`` containing serialized text and metadata.
        """
        my_visited = visited if visited is not None else set()
        params = IDocTagsParams(**kwargs)

        # Build list children explicitly. Requirements:
        # 1) <list ordered="true|false"></list> can be children of lists.
        # 2) Do NOT wrap nested lists into <list_text>, even if they are
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
            # serializer (as <list_text>), not here.
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

            # After the <list_text>, append any nested lists (children of this ListItem)
            # as siblings at the same level (not wrapped in <list_text>).
            for subref in child.children:
                sub = subref.resolve(doc)
                if (
                    isinstance(sub, ListGroup)
                    and sub.self_ref not in my_visited
                    and sub.self_ref not in excluded
                ):
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
                IDocTagsVocabulary.create_list_token(ordered=True)
                if item.first_item_is_enumerated(doc)
                else IDocTagsVocabulary.create_list_token(ordered=False)
            )
            text_res = _wrap_token(text=text_res, open_token=open_token)
        else:
            text_res = ""
        return create_ser_result(text=text_res, span_source=item_results)


class IDocTagsTextSerializer(BaseModel, BaseTextSerializer):
    """IDocTags-specific text item serializer using `<location>` tokens."""

    @override
    def serialize(
        self,
        *,
        item: "TextItem",
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a text item to IDocTags format.

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
                item_.text = item.orig[
                    prov_.charspan[0] : prov_.charspan[1]
                ]  # it must be `orig`, not `text` here!
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
                **kwargs,
            )

    def _serialize_single_item(
        self,
        *,
        item: "TextItem",
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a ``TextItem`` into IDocTags markup.

        Depending on parameters, emits meta blocks, location tokens, and the
        item's textual content (prefixing code language for ``CodeItem``). For
        floating items, captions may be appended. The result can be wrapped in a
        tag derived from the item's label when applicable.

        Args:
            item: The text-like item to serialize.
            doc_serializer: The document-level serializer for delegating nested items.
            doc: The document used to resolve references and children.
            visited: Set of already visited item refs to avoid cycles.
            **kwargs: Additional serializer parameters forwarded to ``IDocTagsParams``.

        Returns:
            A ``SerializationResult`` with the serialized text and span source.
        """
        my_visited = visited if visited is not None else set()
        params = IDocTagsParams(**kwargs)

        # Determine wrapper open-token for this item using IDocTags vocabulary.
        # - SectionHeaderItem: use <heading level="N"> ... </heading>.
        # - Other text-like items: map the label to an IDocTagsToken; for
        #   list items, this maps to <list_text> and keeps the text serializer
        #   free of type-based special casing.
        wrap_open_token: Optional[str]
        selected_token: str = ""
        if isinstance(item, SectionHeaderItem):
            wrap_open_token = IDocTagsVocabulary.create_heading_token(level=item.level)
        elif isinstance(item, ListItem):
            tok = IDocTagsToken.LIST_TEXT
            wrap_open_token = f"<{tok.value}>"
        elif (
            isinstance(item, TextItem) and item.label == DocItemLabel.CHECKBOX_SELECTED
        ):
            tok = IDocTagsToken.TEXT
            # FIXME: make a dedicated create_selected_token in IDocTagsVocabulary
            wrap_open_token = f"<{tok.value}>"
            selected_token = '<selected value="true"/>'
        elif (
            isinstance(item, TextItem)
            and item.label == DocItemLabel.CHECKBOX_UNSELECTED
        ):
            tok = IDocTagsToken.TEXT
            # FIXME: make a dedicated create_selected_token in IDocTagsVocabulary
            wrap_open_token = f"<{tok.value}>"
            selected_token = '<selected value="false"/>'
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
            tok = IDocTagsToken.TEXT
            wrap_open_token = f"<{tok.value}>"
        else:
            label_value = str(item.label)
            try:
                tok = IDocTagsToken(label_value)
                wrap_open_token = f"<{tok.value}>"
            except ValueError:
                raise ValueError(
                    f"Unsupported IDocTags token for label '{label_value}'"
                )

        parts: list[str] = []

        if params.add_location:
            # Use IDocTags `<location>` tokens instead of `<loc_.../>`
            loc = _create_location_tokens_for_item(item=item, doc=doc)
            if loc:
                parts.append(loc)

        if selected_token:
            parts.append(selected_token)

        if item.meta:
            meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
            if meta_res.text:
                parts.append(meta_res.text)

        if params.add_content:
            # Check if we should serialize a single inline group child instead of text
            if not item.text and len(item.children) == 1:
                child = item.children[0].resolve(doc)
                if isinstance(child, InlineGroup):
                    ser_res = doc_serializer.serialize(
                        item=child, visited=my_visited, **kwargs
                    )
                    text_part = ser_res.text
                else:
                    text_part = doc_serializer.post_process(
                        text=item.text,
                        formatting=item.formatting,
                        hyperlink=item.hyperlink,
                    )
            else:
                text_part = doc_serializer.post_process(
                    text=item.text,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
                )

            # For code blocks, preserve language using a lightweight facets marker
            # e.g., <facets>language=python</facets> before the code content.
            if isinstance(item, CodeItem):
                lang = getattr(item.code_language, "value", str(item.code_language))
                if lang:
                    parts.append(
                        _wrap(
                            text=f"language={lang.lower()}",
                            wrap_tag=IDocTagsToken.FACETS.value,
                        )
                    )
                # Keep the textual code content as-is (no stripping)
            else:
                text_part = text_part.strip()

            # Apply XML escaping if xml_compliant is enabled
            text_part = _escape_xml_text(text_part, params.xml_compliant)

            if text_part:
                parts.append(text_part)

        if params.add_caption and isinstance(item, FloatingItem):
            cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
            if cap_text:
                cap_text = _escape_xml_text(cap_text, params.xml_compliant)
                parts.append(cap_text)

        text_res = "".join(parts)
        if wrap_open_token is not None:
            text_res = _wrap_token(text=text_res, open_token=wrap_open_token)
        return create_ser_result(text=text_res, span_source=item)


class IDocTagsMetaSerializer(BaseModel, BaseMetaSerializer):
    """DocTags-specific meta serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """DocTags-specific meta serializer."""
        params = IDocTagsParams(**kwargs)

        elem_delim = ""
        texts = (
            [
                tmp
                for key in (
                    list(item.meta.__class__.model_fields)
                    + list(item.meta.get_custom_part())
                )
                if (
                    (
                        params.allowed_meta_names is None
                        or key in params.allowed_meta_names
                    )
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

    def _serialize_meta_field(
        self, meta: BaseMeta, name: str, params: IDocTagsParams
    ) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None:
            if name == MetaFieldName.SUMMARY and isinstance(
                field_val, SummaryMetaField
            ):
                escaped_text = _escape_xml_text(field_val.text, params.xml_compliant)
                txt = f"<summary>{escaped_text}</summary>"
            elif name == MetaFieldName.DESCRIPTION and isinstance(
                field_val, DescriptionMetaField
            ):
                escaped_text = _escape_xml_text(field_val.text, params.xml_compliant)
                txt = f"<description>{escaped_text}</description>"
            elif name == MetaFieldName.CLASSIFICATION and isinstance(
                field_val, PictureClassificationMetaField
            ):
                class_name = self._humanize_text(
                    field_val.get_main_prediction().class_name
                )
                escaped_class_name = _escape_xml_text(class_name, params.xml_compliant)
                txt = f"<classification>{escaped_class_name}</classification>"
            elif name == MetaFieldName.MOLECULE and isinstance(
                field_val, MoleculeMetaField
            ):
                escaped_smi = _escape_xml_text(field_val.smi, params.xml_compliant)
                txt = f"<molecule>{escaped_smi}</molecule>"
            elif name == MetaFieldName.TABULAR_CHART and isinstance(
                field_val, TabularChartMetaField
            ):
                # suppressing tabular chart serialization
                return None
            # elif tmp := str(field_val or ""):
            #     txt = tmp
            elif name not in {v.value for v in MetaFieldName}:
                escaped_text = _escape_xml_text(
                    str(field_val or ""), params.xml_compliant
                )
                txt = _wrap(text=escaped_text, wrap_tag=name)
            return txt
        return None


class IDocTagsPictureSerializer(BasePictureSerializer):
    """DocTags-specific picture item serializer."""

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
        params = IDocTagsParams(**kwargs)

        open_token: str = IDocTagsVocabulary.create_floating_group_token(
            value=IDocTagsAttributeValue.PICTURE
        )
        close_token: str = IDocTagsVocabulary.create_floating_group_token(
            value=IDocTagsAttributeValue.PICTURE, closing=True
        )

        # Build caption (as a sibling of the picture within the floating_group)
        res_parts: list[SerializationResult] = []
        caption_text = ""
        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                caption_text = cap_res.text
                res_parts.append(cap_res)

        # Build picture inner content (meta + body) that will go inside <picture> ... </picture>
        picture_inner_parts: list[str] = []
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if item.meta:
                meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
                if meta_res.text:
                    picture_inner_parts.append(meta_res.text)
                    res_parts.append(meta_res)

            body = ""
            if params.add_location:
                body += _create_location_tokens_for_item(item=item, doc=doc)

            # handle tabular chart data
            chart_data: Optional[TableData] = None
            if item.meta and item.meta.tabular_chart:
                chart_data = item.meta.tabular_chart.chart_data
            if chart_data and chart_data.table_cells:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(data=chart_data)
                # Reuse the IDocTags table emission for chart data
                params_chart = IDocTagsParams(
                    **{
                        **params.model_dump(),
                        "add_table_cell_location": False,
                    }
                )
                otsl_content = IDocTagsTableSerializer()._emit_otsl(
                    item=temp_table,  # type: ignore[arg-type]
                    doc_serializer=doc_serializer,
                    doc=temp_doc,
                    params=params_chart,
                    **kwargs,
                )
                body += otsl_content

            if body:
                picture_inner_parts.append(body)
                res_parts.append(create_ser_result(text=body, span_source=item))

        picture_text = "".join(picture_inner_parts)
        if picture_text:
            picture_text = _wrap(
                text=picture_text, wrap_tag=IDocTagsToken.PICTURE.value
            )

        # Compose final structure for picture group:
        # <floating_group class="picture"> [<caption>] <picture>...</picture> </floating_group>
        composed_inner = f"{caption_text}{picture_text}"
        text_res = ""
        if composed_inner:
            text_res = f"{open_token}{composed_inner}{close_token}"

        return create_ser_result(text=text_res, span_source=res_parts)


class IDocTagsTableSerializer(BaseTableSerializer):
    """DocTags-specific table item serializer."""

    # _get_table_token no longer needed; OTSL tokens are emitted via vocabulary

    def _emit_otsl(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        params: "IDocTagsParams",
        **kwargs: Any,
    ) -> str:
        """Emit OTSL payload using IDocTags tokens and location semantics.

        Raises ValueError when per-cell location is requested but the
        document/page context is unavailable.
        """
        if not item.data or not item.data.table_cells:
            return ""

        nrows, ncols = item.data.num_rows, item.data.num_cols

        # Check if any cells have bounding boxes available
        has_any_cell_bbox = any(
            (c.bbox is not None) for row in item.data.grid for c in row
        )

        # If per-cell location is requested but no cells have bboxes, fail early
        if params.add_table_cell_location and not has_any_cell_bbox:
            raise ValueError(
                "Cell locations requested but no cells have bounding boxes in the table."
            )

        # Determine if we need page context for location serialization
        need_cell_loc = params.add_table_cell_location and has_any_cell_bbox

        page_no = 0
        if need_cell_loc:
            if not item.prov:
                raise ValueError(
                    "Per-cell location requested but table has no provenance (page_no)."
                )
            page_no = item.prov[0].page_no
            if not doc.pages or page_no not in doc.pages:
                raise ValueError(
                    f"Per-cell location requested but no page info for page {page_no}."
                )
            page_w, page_h = doc.pages[page_no].size.as_tuple()
            if page_w <= 0 or page_h <= 0:
                raise ValueError(
                    f"Invalid page size ({page_w}, {page_h}) for page {page_no}."
                )
        else:
            # Defaults will not be used unless a bbox is present and
            # add_table_cell_location is true
            page_w, page_h = (1.0, 1.0)

        parts: list[str] = []
        for i in range(nrows):
            for j in range(ncols):
                cell = item.data.grid[i][j]
                content = cell._get_text(
                    doc=doc, doc_serializer=doc_serializer, **kwargs
                ).strip()

                rowspan, rowstart = cell.row_span, cell.start_row_offset_idx
                colspan, colstart = cell.col_span, cell.start_col_offset_idx

                # Optional per-cell location
                cell_loc = ""
                if params.add_table_cell_location:

                    if cell.bbox is None:
                        raise ValueError("Missing cell-bbox in the table-grid.")

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
                            parts.append(
                                IDocTagsVocabulary.create_selfclosing_token(
                                    token=IDocTagsToken.CHED
                                )
                            )
                        elif cell.row_header:
                            parts.append(
                                IDocTagsVocabulary.create_selfclosing_token(
                                    token=IDocTagsToken.RHED
                                )
                            )
                        elif cell.row_section:
                            parts.append(
                                IDocTagsVocabulary.create_selfclosing_token(
                                    token=IDocTagsToken.SROW
                                )
                            )
                        else:
                            parts.append(
                                IDocTagsVocabulary.create_selfclosing_token(
                                    token=IDocTagsToken.FCEL
                                )
                            )

                        if params.add_table_cell_location:
                            parts.append(cell_loc)
                        if params.add_content:
                            # Apply XML escaping to table cell content
                            escaped_content = _escape_xml_text(
                                content, params.xml_compliant
                            )
                            parts.append(escaped_content)
                    else:
                        parts.append(
                            IDocTagsVocabulary.create_selfclosing_token(
                                token=IDocTagsToken.ECEL
                            )
                        )
                elif (
                    rowstart != i and colspan == 1
                ):  # FIXME: I believe we should have colstart == j
                    parts.append(
                        IDocTagsVocabulary.create_selfclosing_token(
                            token=IDocTagsToken.UCEL
                        )
                    )
                elif (
                    colstart != j and rowspan == 1
                ):  # FIXME: I believe we should have rowstart == i
                    parts.append(
                        IDocTagsVocabulary.create_selfclosing_token(
                            token=IDocTagsToken.LCEL
                        )
                    )
                else:
                    parts.append(
                        IDocTagsVocabulary.create_selfclosing_token(
                            token=IDocTagsToken.XCEL
                        )
                    )

            parts.append(
                IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.NL)
            )

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
        params = IDocTagsParams(**kwargs)

        # FIXME: we might need to check the label to distinguish between TABLE and DOCUMENT_INDEX label
        open_token: str = IDocTagsVocabulary.create_floating_group_token(
            value=IDocTagsAttributeValue.TABLE
        )
        close_token: str = IDocTagsVocabulary.create_floating_group_token(
            value=IDocTagsAttributeValue.TABLE, closing=True
        )

        res_parts: list[SerializationResult] = []

        # Caption as sibling of the OTSL payload within the floating group
        caption_text = ""
        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                caption_text = cap_res.text
                res_parts.append(cap_res)

        # Build table payload: location (if any) + OTSL content inside <otsl> ... </otsl>
        otsl_payload = ""
        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""
            if params.add_location:
                body += _create_location_tokens_for_item(
                    item=item, doc=doc, xres=params.xsize, yres=params.ysize
                )

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
                otsl_payload = _wrap(text=body, wrap_tag=IDocTagsToken.OTSL.value)
                res_parts.append(create_ser_result(text=body, span_source=item))

        composed_inner = f"{caption_text}{otsl_payload}"
        text_res = ""
        if composed_inner:
            text_res = f"{open_token}{composed_inner}{close_token}"

        return create_ser_result(text=text_res, span_source=res_parts)


class IDocTagsInlineSerializer(BaseInlineSerializer):
    """Inline serializer emitting IDocTags `<inline>` and `<location>` tokens."""

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
        """Serialize inline content with optional location into IDocTags text."""
        my_visited = visited if visited is not None else set()
        params = IDocTagsParams(**kwargs)
        parts: list[SerializationResult] = []
        if params.add_location:
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
            text_res = _wrap(text=text_res, wrap_tag=IDocTagsToken.INLINE.value)
        return create_ser_result(text=text_res, span_source=parts)


class IDocTagsFallbackSerializer(BaseFallbackSerializer):
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
        if isinstance(item, (ListGroup, InlineGroup)):
            parts = doc_serializer.get_parts(item=item, **kwargs)
            text_res = "\n".join([p.text for p in parts if p.text])
            return create_ser_result(text=text_res, span_source=parts)
        return create_ser_result()


class IDocTagsKeyValueSerializer(BaseKeyValueSerializer):
    """No-op serializer for key/value items in IDocTags."""

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


class IDocTagsFormSerializer(BaseFormSerializer):
    """No-op serializer for form items in IDocTags."""

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


class IDocTagsAnnotationSerializer(BaseAnnotationSerializer):
    """No-op annotation serializer; IDocTags relies on meta instead."""

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


class IDocTagsDocSerializer(DocSerializer):
    """IDocTags document serializer."""

    text_serializer: BaseTextSerializer = IDocTagsTextSerializer()
    table_serializer: BaseTableSerializer = IDocTagsTableSerializer()
    picture_serializer: BasePictureSerializer = IDocTagsPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = IDocTagsKeyValueSerializer()
    form_serializer: BaseFormSerializer = IDocTagsFormSerializer()
    fallback_serializer: BaseFallbackSerializer = IDocTagsFallbackSerializer()

    list_serializer: BaseListSerializer = IDocTagsListSerializer()
    inline_serializer: BaseInlineSerializer = IDocTagsInlineSerializer()

    meta_serializer: BaseMetaSerializer = IDocTagsMetaSerializer()
    annotation_serializer: BaseAnnotationSerializer = IDocTagsAnnotationSerializer()

    params: IDocTagsParams = IDocTagsParams()

    @override
    def _meta_is_wrapped(self) -> bool:
        return True

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's captions with IDocTags location tokens."""
        params = IDocTagsParams(**kwargs)
        results: list[SerializationResult] = []
        if item.captions:
            cap_res = super().serialize_captions(item, **kwargs)
            if cap_res.text and params.add_location:
                for caption in item.captions:
                    if caption.cref not in self.get_excluded_refs(**kwargs):
                        if isinstance(cap := caption.resolve(self.doc), DocItem):
                            loc_txt = _create_location_tokens_for_item(
                                item=cap, doc=self.doc
                            )
                            results.append(create_ser_result(text=loc_txt))
            if cap_res.text and params.add_content:
                results.append(cap_res)
        text_res = "".join([r.text for r in results])
        if text_res:
            text_res = _wrap(text=text_res, wrap_tag=IDocTagsToken.CAPTION.value)
        return create_ser_result(text=text_res, span_source=results)

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Doc-level serialization with IDocTags root wrapper."""
        # Note: removed internal thread counting; not used.

        delim = _get_delim(params=self.params)

        open_token: str = IDocTagsVocabulary.create_doctag_root()
        close_token: str = IDocTagsVocabulary.create_doctag_root(closing=True)

        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            # Always emit well‑formed page breaks using the vocabulary
            page_sep = IDocTagsVocabulary.create_selfclosing_token(
                token=IDocTagsToken.PAGE_BREAK
            )
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        text_res = f"{open_token}{text_res}{close_token}"

        if self.params.pretty_indentation:
            try:
                my_root = parseString(text_res).documentElement
            except Exception as e:
                # print(text_res)

                ctx = _xml_error_context(text_res, e)
                raise ValueError(
                    f"XML pretty-print failed: {e}\n--- XML context ---\n{ctx}"
                ) from e
            if my_root is None:
                raise ValueError("XML pretty-print failed: documentElement is None")
            text_res = my_root.toprettyxml(indent=self.params.pretty_indentation)
            text_res = "\n".join(
                [line for line in text_res.split("\n") if line.strip()]
            )

            if self.params.preserve_empty_non_selfclosing:
                # Expand self-closing forms for tokens that are not allowed
                # to be self-closing according to the vocabulary.
                # Example: <list_text/> -> <list_text></list_text>
                non_selfclosing = [
                    tok
                    for tok in IDocTagsToken
                    if not IDocTagsVocabulary.IS_SELFCLOSING.get(tok, False)
                ]

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
        """Apply IDocTags-specific bold serialization."""
        return _wrap(text=text, wrap_tag=IDocTagsToken.BOLD.value)

    @override
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Apply IDocTags-specific italic serialization."""
        return _wrap(text=text, wrap_tag=IDocTagsToken.ITALIC.value)

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Apply IDocTags-specific strikethrough serialization."""
        return _wrap(text=text, wrap_tag=IDocTagsToken.STRIKETHROUGH.value)

    @override
    def serialize_subscript(self, text: str, **kwargs: Any) -> str:
        """Apply IDocTags-specific subscript serialization."""
        return _wrap(text=text, wrap_tag=IDocTagsToken.SUBSCRIPT.value)

    @override
    def serialize_superscript(self, text: str, **kwargs: Any) -> str:
        """Apply IDocTags-specific superscript serialization."""
        return _wrap(text=text, wrap_tag=IDocTagsToken.SUPERSCRIPT.value)


class IDocTagsDocDeserializer(BaseModel):
    """IDocTags document deserializer."""

    # Internal state used while walking the tree (private instance attributes)
    _page_no: int = PrivateAttr(default=0)
    _default_resolution: int = PrivateAttr(default=DOCTAGS_RESOLUTION)

    def deserialize(
        self,
        *,
        doctags: str,
    ) -> DoclingDocument:
        """Deserialize DocTags XML into a DoclingDocument.

        Args:
            doctags: DocTags XML string to parse.

        Returns:
            A populated `DoclingDocument` parsed from the input.
        """
        try:
            root_node = parseString(doctags).documentElement
        except Exception as e:
            ctx = _xml_error_context(doctags, e)
            raise ValueError(
                f"Invalid DocTags XML: {e}\n--- XML context ---\n{ctx}"
            ) from e
        if root_node is None:
            raise ValueError("Invalid DocTags XML: missing documentElement")
        root: Element = cast(Element, root_node)
        if root.tagName != IDocTagsToken.DOCUMENT.value:
            candidates = root.getElementsByTagName(IDocTagsToken.DOCUMENT.value)
            if candidates:
                root = cast(Element, candidates[0])

        doc = DoclingDocument(name="Document")
        # Initialize with a default page so location tokens can be re‑emitted
        self._page_no = 0
        self._default_resolution = DOCTAGS_RESOLUTION
        self._ensure_page_exists(
            doc=doc, page_no=self._page_no, resolution=self._default_resolution
        )
        self._parse_document_root(doc=doc, root=root)
        return doc

    # ------------- Core walkers -------------
    def _parse_document_root(self, *, doc: DoclingDocument, root: Element) -> None:
        for node in root.childNodes:
            if isinstance(node, Element):
                self._dispatch_element(doc=doc, el=node, parent=None)

    def _dispatch_element(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        name = el.tagName
        if name in {
            IDocTagsToken.TITLE.value,
            IDocTagsToken.TEXT.value,
            IDocTagsToken.CAPTION.value,
            IDocTagsToken.FOOTNOTE.value,
            IDocTagsToken.PAGE_HEADER.value,
            IDocTagsToken.PAGE_FOOTER.value,
            IDocTagsToken.CODE.value,
            IDocTagsToken.FORMULA.value,
        }:
            self._parse_text_like(doc=doc, el=el, parent=parent)
        elif name == IDocTagsToken.PAGE_BREAK.value:
            # Start a new page; keep a default square page using the configured resolution
            self._page_no += 1
            self._ensure_page_exists(
                doc=doc, page_no=self._page_no, resolution=self._default_resolution
            )
        elif name == IDocTagsToken.HEADING.value:
            self._parse_heading(doc=doc, el=el, parent=parent)
        elif name == IDocTagsToken.LIST.value:
            self._parse_list(doc=doc, el=el, parent=parent)
        elif name == IDocTagsToken.FLOATING_GROUP.value:
            self._parse_floating_group(doc=doc, el=el, parent=parent)
        elif name == IDocTagsToken.INLINE.value:
            self._parse_inline_group(doc=doc, el=el, parent=parent)
        else:
            self._walk_children(doc=doc, el=el, parent=parent)

    def _walk_children(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        for node in el.childNodes:
            if isinstance(node, Element):
                # Ignore geometry/meta containers at this level; pass through page breaks
                if node.tagName in {
                    IDocTagsToken.HEAD.value,
                    IDocTagsToken.META.value,
                    IDocTagsToken.LOCATION.value,
                }:
                    continue
                self._dispatch_element(doc=doc, el=node, parent=parent)

    # ------------- Text blocks -------------
    def _parse_text_like(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        """Parse text-like tokens (title, text, caption, footnotes, code, formula)."""
        prov_list = self._extract_provenance(doc=doc, el=el)
        text, formatting = self._extract_text_with_formatting(el)
        text = text.strip()
        if not text:
            return

        nm = el.tagName

        # Handle code separately (language + content extraction)
        if nm == IDocTagsToken.CODE.value:
            code_text, lang_label = self._extract_code_content_and_language(el)
            if not code_text.strip():
                return
            item = doc.add_code(
                text=code_text,
                code_language=lang_label,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
            )
            for p in prov_list[1:]:
                item.prov.append(p)
            return

        # Map text-like tokens to text item labels
        text_label_map = {
            IDocTagsToken.TEXT.value: DocItemLabel.TEXT,
            IDocTagsToken.CAPTION.value: DocItemLabel.CAPTION,
            IDocTagsToken.FOOTNOTE.value: DocItemLabel.FOOTNOTE,
            IDocTagsToken.PAGE_HEADER.value: DocItemLabel.PAGE_HEADER,
            IDocTagsToken.PAGE_FOOTER.value: DocItemLabel.PAGE_FOOTER,
        }

        if nm in text_label_map:
            item = doc.add_text(
                label=text_label_map[nm],
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            for p in prov_list[1:]:
                item.prov.append(p)
            return

        if nm == IDocTagsToken.TITLE.value:
            item = doc.add_title(
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            for p in prov_list[1:]:
                item.prov.append(p)
            return

        if nm == IDocTagsToken.FORMULA.value:
            item = doc.add_formula(
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            for p in prov_list[1:]:
                item.prov.append(p)
            return

    def _extract_code_content_and_language(
        self, el: Element
    ) -> tuple[str, CodeLanguageLabel]:
        """Extract code content and language from a <code> element."""
        lang_label = CodeLanguageLabel.UNKNOWN
        parts: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                if node.data.strip():
                    parts.append(node.data)
            elif isinstance(node, Element):
                nm_child = node.tagName
                if nm_child == IDocTagsToken.FACETS.value:
                    facets_text = self._get_text(node).strip()
                    if "=" in facets_text:
                        key, val = facets_text.split("=", 1)
                        if key.strip().lower() == "language":
                            val_norm = val.strip().lower()
                            try:
                                lang_label = next(
                                    lbl
                                    for lbl in CodeLanguageLabel
                                    if lbl.value.lower() == val_norm
                                )
                            except StopIteration:
                                lang_label = CodeLanguageLabel.UNKNOWN
                    continue
                if nm_child == IDocTagsToken.LOCATION.value:
                    continue
                if nm_child == IDocTagsToken.BR.value:
                    parts.append("\n")
                else:
                    parts.append(self._get_text(node))

        return "".join(parts), lang_label

    def _parse_heading(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        lvl_txt = el.getAttribute(IDocTagsAttributeKey.LEVEL.value) or "1"
        try:
            level = int(lvl_txt)
        except Exception:
            level = 1
        # Extract provenance from heading token (if any)
        prov_list = self._extract_provenance(doc=doc, el=el)
        text = self._get_text(el)
        text_stripped = text.strip()
        if text_stripped:
            item = doc.add_heading(
                text=text_stripped,
                level=level,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
            )
            for p in prov_list[1:]:
                item.prov.append(p)

    def _parse_list(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        ordered = (
            el.getAttribute(IDocTagsAttributeKey.ORDERED.value)
            == IDocTagsAttributeValue.TRUE.value
        )
        group = doc.add_list_group(parent=parent)
        for node in el.childNodes:
            if not isinstance(node, Element):
                continue
            if node.tagName == IDocTagsToken.LIST_TEXT.value:
                # Collect provenance from the <list_text> wrapper
                prov_list = self._extract_provenance(doc=doc, el=node)
                text = self._get_text(node).strip()
                if text:
                    item = doc.add_list_item(
                        text=text,
                        parent=group,
                        enumerated=ordered,
                        prov=(prov_list[0] if prov_list else None),
                    )
                    for p in prov_list[1:]:
                        item.prov.append(p)
            elif node.tagName == IDocTagsToken.LIST.value:
                # Recursively handle nested lists
                self._parse_list(doc=doc, el=node, parent=group)

    # ------------- Inline groups -------------
    def _parse_inline_group(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        """Parse <inline> elements into InlineGroup objects."""
        # Create the inline group
        inline_group = doc.add_inline_group(parent=parent)

        # Process all child elements, adding them as children of the inline group
        for node in el.childNodes:
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

    # ------------- Floating groups -------------
    def _parse_floating_group(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        cls_val = el.getAttribute(IDocTagsAttributeKey.CLASS.value)
        if cls_val == IDocTagsAttributeValue.TABLE.value:
            self._parse_table_group(doc=doc, el=el, parent=parent)
        elif cls_val == IDocTagsAttributeValue.PICTURE.value:
            self._parse_picture_group(doc=doc, el=el, parent=parent)
        else:
            self._walk_children(doc=doc, el=el, parent=parent)

    def _parse_table_group(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        caption = self._extract_caption(doc=doc, el=el)
        otsl_el = self._first_child(el, IDocTagsToken.OTSL.value)
        if otsl_el is None:
            doc.add_table(data=TableData(), caption=caption, parent=parent)
            return
        # Extract table provenance from <otsl> leading <location/> tokens
        tbl_provs = self._extract_provenance(doc=doc, el=otsl_el)
        inner = self._inner_xml(otsl_el)
        # Remove any location tokens from the OTSL content before parsing
        inner = re.sub(r"<\s*location\b[^>]*/\s*>", "", inner)
        td = self._parse_otsl_table_content(f"<otsl>{inner}</otsl>")
        tbl = doc.add_table(
            data=td,
            caption=caption,
            parent=parent,
            prov=(tbl_provs[0] if tbl_provs else None),
        )
        for p in tbl_provs[1:]:
            tbl.prov.append(p)

    def _parse_picture_group(
        self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]
    ) -> None:
        # Extract caption from the floating group
        caption = self._extract_caption(doc=doc, el=el)

        # Extract provenance from the <picture> block (locations appear inside it)
        prov_list: list[ProvenanceItem] = []
        picture_el = self._first_child(el, IDocTagsToken.PICTURE.value)
        if picture_el is not None:
            prov_list = self._extract_provenance(doc=doc, el=picture_el)

        # Create the picture item first, attach caption and provenance
        pic = doc.add_picture(
            caption=caption,
            parent=parent,
            prov=(prov_list[0] if prov_list else None),
        )
        for p in prov_list[1:]:
            pic.prov.append(p)

        # If there is a <picture> child and it contains an <otsl>,
        # parse it as TabularChartMetaField and attach to picture.meta
        if picture_el is not None:
            otsl_el = self._first_child(picture_el, IDocTagsToken.OTSL.value)
            if otsl_el is not None:
                inner = self._inner_xml(otsl_el)
                td = self._parse_otsl_table_content(f"<otsl>{inner}</otsl>")
                if pic.meta is None:
                    pic.meta = PictureMeta()
                pic.meta.tabular_chart = TabularChartMetaField(chart_data=td)

    # ------------- Helpers -------------
    def _extract_caption(
        self, *, doc: DoclingDocument, el: Element
    ) -> Optional[TextItem]:
        cap_el = self._first_child(el, IDocTagsToken.CAPTION.value)
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

    def _first_child(self, el: Element, tag_name: str) -> Optional[Element]:
        for node in el.childNodes:
            if isinstance(node, Element) and node.tagName == tag_name:
                return node
        return None

    def _inner_xml(self, el: Element) -> str:
        parts: list[str] = []
        for node in el.childNodes:
            if isinstance(node, Text):
                parts.append(node.data)
            elif isinstance(node, Element):
                parts.append(node.toxml())
        return "".join(parts)

    # --------- OTSL table parsing (inlined) ---------
    def _otsl_extract_tokens_and_text(self, s: str) -> tuple[list[str], list[str]]:
        """Extract OTSL structural tokens and interleaved text.

        Strips the outer <otsl> wrapper and ignores location tokens (expected
        to be removed before).
        """
        pattern = r"(<[^>]+>)"
        tokens = re.findall(pattern, s)
        # Drop the <otsl> wrapper tags
        tokens = [
            t
            for t in tokens
            if t
            not in [
                f"<{IDocTagsToken.OTSL.value}>",
                f"</{IDocTagsToken.OTSL.value}>",
            ]
        ]

        parts = re.split(pattern, s)
        parts = [
            p
            for p in parts
            if p.strip()
            and p
            not in [
                f"<{IDocTagsToken.OTSL.value}>",
                f"</{IDocTagsToken.OTSL.value}>",
            ]
        ]
        return tokens, parts

    def _otsl_parse_texts(
        self, texts: list[str], tokens: list[str]
    ) -> tuple[list[TableCell], list[list[str]]]:
        """Parse OTSL interleaved texts+tokens into TableCell list and row tokens."""
        # Token strings used in the stream (normalized to <name>)

        fcel = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.FCEL)
        ecel = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.ECEL)
        lcel = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.LCEL)
        ucel = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.UCEL)
        xcel = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.XCEL)
        nl = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.NL)
        ched = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.CHED)
        rhed = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.RHED)
        srow = IDocTagsVocabulary.create_selfclosing_token(token=IDocTagsToken.SROW)

        # Clean tokens to only structural OTSL markers
        clean_tokens: list[str] = []
        for t in tokens:
            if t in [ecel, fcel, lcel, ucel, xcel, nl, ched, rhed, srow]:
                clean_tokens.append(t)
        tokens = clean_tokens

        # Split into rows by NL markers while keeping segments
        split_row_tokens = [
            list(group)
            for is_sep, group in groupby(tokens, key=lambda z: z == nl)
            if not is_sep
        ]

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

                next_right = (
                    texts[i + right_offset] if i + right_offset < len(texts) else ""
                )
                next_bottom = (
                    split_row_tokens[r_idx + 1][c_idx]
                    if (r_idx + 1) < len(split_row_tokens)
                    and c_idx < len(split_row_tokens[r_idx + 1])
                    else ""
                )

                if next_right in [lcel, xcel]:
                    col_span += count_right(
                        split_row_tokens, c_idx + 1, r_idx, [lcel, xcel]
                    )
                if next_bottom in [ucel, xcel]:
                    row_span += count_down(
                        split_row_tokens, c_idx, r_idx + 1, [ucel, xcel]
                    )

                table_cells.append(
                    TableCell(
                        text=cell_text.strip(),
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

    def _parse_otsl_table_content(self, otsl_content: str) -> TableData:
        """Parse OTSL content into TableData (inlined from utils)."""
        tokens, mixed = self._otsl_extract_tokens_and_text(otsl_content)
        table_cells, split_rows = self._otsl_parse_texts(mixed, tokens)
        return TableData(
            num_rows=len(split_rows),
            num_cols=(max(len(r) for r in split_rows) if split_rows else 0),
            table_cells=table_cells,
        )

    def _extract_text_with_formatting(
        self, el: Element
    ) -> tuple[str, Optional[Formatting]]:
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
            if isinstance(node, Element)
            and node.tagName not in {IDocTagsToken.LOCATION.value}
        ]

        # Check if we have a single child that is a formatting tag
        if len(child_elements) == 1:
            child = child_elements[0]
            tag_name = child.tagName

            # Mapping of format tags to Formatting attributes
            format_tags = {
                IDocTagsToken.BOLD.value: "bold",
                IDocTagsToken.ITALIC.value: "italic",
                IDocTagsToken.STRIKETHROUGH.value: "strikethrough",
                IDocTagsToken.SUPERSCRIPT.value: "superscript",
                IDocTagsToken.SUBSCRIPT.value: "subscript",
            }

            if tag_name in format_tags:
                # Recursively extract text and formatting from the child
                text, child_formatting = self._extract_text_with_formatting(child)

                # Build up the formatting object
                if child_formatting is None:
                    child_formatting = Formatting()

                # Apply the current formatting tag
                if tag_name == IDocTagsToken.BOLD.value:
                    child_formatting.bold = True
                elif tag_name == IDocTagsToken.ITALIC.value:
                    child_formatting.italic = True
                elif tag_name == IDocTagsToken.STRIKETHROUGH.value:
                    child_formatting.strikethrough = True
                elif tag_name == IDocTagsToken.SUPERSCRIPT.value:
                    child_formatting.script = Script.SUPER
                elif tag_name == IDocTagsToken.SUBSCRIPT.value:
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
                    out.append(node.data)
            elif isinstance(node, Element):
                nm = node.tagName
                if nm in {IDocTagsToken.LOCATION.value}:
                    continue
                if nm == IDocTagsToken.BR.value:
                    out.append("\n")
                else:
                    out.append(self._get_text(node))
        return "".join(out)

    # --------- Location helpers ---------
    def _ensure_page_exists(
        self, *, doc: DoclingDocument, page_no: int, resolution: int
    ) -> None:
        # If the page already exists, do nothing; otherwise add with a square size based on resolution
        if page_no not in doc.pages:
            doc.add_page(
                page_no=page_no, size=Size(width=resolution, height=resolution)
            )

    def _extract_provenance(
        self, *, doc: DoclingDocument, el: Element
    ) -> list[ProvenanceItem]:
        # Collect immediate child <location value=.. resolution=.. /> tokens in groups of 4
        values: list[int] = []
        res_for_group: Optional[int] = None
        provs: list[ProvenanceItem] = []

        for node in el.childNodes:
            if not isinstance(node, Element):
                continue
            if node.tagName != IDocTagsToken.LOCATION.value:
                continue
            try:
                v = int(node.getAttribute(IDocTagsAttributeKey.VALUE.value) or "0")
            except Exception:
                v = 0
            try:
                r = int(
                    node.getAttribute(IDocTagsAttributeKey.RESOLUTION.value)
                    or str(self._default_resolution)
                )
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
                bbox = BoundingBox.from_tuple(
                    (l, t, rgt, btm), origin=CoordOrigin.TOPLEFT
                )
                provs.append(
                    ProvenanceItem(page_no=self._page_no, bbox=bbox, charspan=(0, 0))
                )
                values = []
                res_for_group = None

        return provs
