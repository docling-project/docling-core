"""Define classes for Doclang serialization."""

import copy
import re
import warnings
from enum import Enum
from itertools import groupby
from typing import Any, ClassVar, Final, Optional, cast
from xml.dom.minidom import Element, Node, Text, parseString

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
from docling_core.types.doc.document import FormulaItem, RichTableCell
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    PictureClassificationLabel,
)

# Note: Intentionally avoid importing DocumentToken here to ensure
# Doclang uses only its own token vocabulary.

DOCLANG_VERSION: Final = "1.0.0"
DOCLANG_DFLT_RESOLUTION: int = 512


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


def _wrap_token(*, text: str, open_token: str) -> str:
    close_token = DoclangVocabulary.create_closing_token(token=open_token)
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
        warnings.warn(f"Normalized {value=} greater or equal to 1; returning {resolution-1=}", stacklevel=2)
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
        DoclangVocabulary.create_location_token(value=x0v, resolution=xres)
        + DoclangVocabulary.create_location_token(value=y0v, resolution=yres)
        + DoclangVocabulary.create_location_token(value=x1v, resolution=xres)
        + DoclangVocabulary.create_location_token(value=y1v, resolution=yres)
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
    CONTINUATION = "continuation"


class DoclangToken(str, Enum):
    """DoclangToken.

    This class implements the tokens from the Token table,

    | # | Category | Token | Self-Closing [Yes/No] | Parametrized [Yes/No] | Attributes | Description |
    |---|----------|-------|-----------------------|-----------------------|------------|-------------|
    | 1 | Root Elements | `doclang` | No | Yes | `version` | Root container; optional semantic version `version`. |
    | 2 | Special Elements | `page_break` | Yes | No | — | Page delimiter. |
    | 3 |  | `time_break` | Yes | No | — | Temporal segment delimiter. |
    | 4 | Metadata Containers | `head` | No | No | — | Document-level metadata container. |
    | 5 |  | `meta` | No | No | — | Component-level metadata container. |
    | 6 | Geometric Tokens | `location` | Yes | Yes | `value`, `resolution?` |
        Geometric coordinate; `value` in [0, res]; optional `resolution`. |
    | 7 | Temporal Tokens | `hour` | Yes | Yes | `value` | Hours component; `value` in [0, 99]. |
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
    | 19 |  | `picture` | No | No | — | Block image/graphic; at most one of
        `base64`/`uri`; may include `meta` for classification; `otsl` may encode chart data. |
    | 20 |  | `form` | No | No | — | Form structure container. |
    | 21 |  | `formula` | No | No | — | Mathematical expression block. |
    | 22 |  | `code` | No | No | — | Code block. |
    | 23 |  | `list_text` | No | No | — | Leading text of a list item, including any
        available marker or checkbox information. |
    | 24 |  | `form_item` | No | No | — | Form item; exactly one `key`; one or more of
        `value`/`checkbox`/`marker`/`hint`. |
    | 25 |  | `form_heading` | No | Yes | `level?` | Form header; optional `level` (N ≥ 1). |
    | 26 |  | `form_text` | No | No | — | Form text block. |
    | 27 |  | `hint` | No | No | — | Hint for a fillable field (format/example/description). |
    | 28 | Grouping Tokens | `group` | No | Yes | `type?` | Generic group; no `location`
        tokens; associates composite content (e.g., captions/footnotes). |
    | 39 |  | `list` | No | Yes | `class` in {`unordered`, `ordered`}; defaults to `unordered` | List container. |
    | 30 |  | `floating_group` | No | Yes | `class` in {`table`,`picture`,`form`,`code`} |
        Floating container that groups a floating component with its caption, footnotes, and
        metadata; no `location` tokens. |
    | 31 | Formatting Tokens | `bold` | No | No | — | Bold text. |
    | 32 |  | `italic` | No | No | — | Italic text. |
    | 33 |  | `strikethrough` | No | No | — | Strike-through text. |
    | 34 |  | `superscript` | No | No | — | Superscript text. |
    | 35 |  | `subscript` | No | No | — | Subscript text. |
    | 36 |  | `rtl` | No | No | — | Right-to-left text direction. |
    | 37 |  | `inline` | No | Yes | `class` in {`formula`,`code`,`picture`} |
        Inline content; if `class="picture"`, may include one of `base64` or `uri`. |
    | 38 |  | `br` | Yes | No | — | Line break. |
    | 39 | Structural Tokens (OTSL) | `otsl` | No | No | — | Table structure container. |
    | 40 |  | `fcel` | Yes | No | — | New cell with content. |
    | 41 |  | `ecel` | Yes | No | — | New cell without content. |
    | 42 |  | `ched` | Yes | No | — | Column header cell. |
    | 43 |  | `rhed` | Yes | No | — | Row header cell. |
    | 44 |  | `corn` | Yes | No | — | Corner header cell. |
    | 45 |  | `srow` | Yes | No | — | Section row separator cell. |
    | 46 |  | `lcel` | Yes | No | — | Merge with left neighbor (horizontal span). |
    | 47 |  | `ucel` | Yes | No | — | Merge with upper neighbor (vertical span). |
    | 48 |  | `xcel` | Yes | No | — | Merge with left and upper neighbors (2D span). |
    | 49 |  | `nl` | Yes | No | — | New line (row separator). |
    | 50 | Continuation Tokens | `thread` | Yes | Yes | `id` |
        Continuation marker for split content; reuse same `id` across parts. |
    | 51 |  | `h_thread` | Yes | Yes | `id` | Horizontal stitching marker for split tables; reuse same `id`. |
    | 52 | Binary Data Tokens | `base64` | No | No | — | Embedded binary data (base64). |
    | 53 |  | `uri` | No | No | — | External resource reference. |
    | 54 | Content Tokens | `marker` | No | No | — | List/form marker content. |
    | 55 |  | `checkbox` | Yes | Yes | `class` in {`unselected`, `selected`};
        defaults to `unselected` | Checkbox status. |
    | 56 |  | `facets` | No | No | — | Container for application-specific derived properties. |
    | 57 | Structural Tokens (Form) | `key` | No | No | — | Form item key (child of `form_item`). |
    | 58 |  | `value` | No | No | — | Form item value (child of `form_item`). |
    """

    # Root and metadata
    DOCUMENT = "doclang"
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
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
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
    URI = "uri"
    MARKER = "marker"
    FACETS = "facets"
    CONTENT = "content"  # TODO: review element name


class DoclangAttributeKey(str, Enum):
    """Attribute keys allowed on Doclang tokens."""

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

    # Checkbox classes
    SELECTED = "selected"
    UNSELECTED = "unselected"

    # Inline class values
    FORMULA = "formula"
    CODE = "code"
    PICTURE = "picture"

    # Floating group class values
    DOCUMENT_INDEX = "document_index"
    TABLE = "table"
    FORM = "form"


class DoclangVocabulary(BaseModel):
    """DoclangVocabulary."""

    # Allowed attributes per token (defined outside the Enum to satisfy mypy)
    ALLOWED_ATTRIBUTES: ClassVar[dict[DoclangToken, set["DoclangAttributeKey"]]] = {
        DoclangToken.DOCUMENT: {
            DoclangAttributeKey.VERSION,
        },
        DoclangToken.LOCATION: {
            DoclangAttributeKey.VALUE,
            DoclangAttributeKey.RESOLUTION,
        },
        DoclangToken.HOUR: {DoclangAttributeKey.VALUE},
        DoclangToken.MINUTE: {DoclangAttributeKey.VALUE},
        DoclangToken.SECOND: {DoclangAttributeKey.VALUE},
        DoclangToken.CENTISECOND: {DoclangAttributeKey.VALUE},
        DoclangToken.HEADING: {DoclangAttributeKey.LEVEL},
        DoclangToken.FORM_HEADING: {DoclangAttributeKey.LEVEL},
        DoclangToken.CHECKBOX: {DoclangAttributeKey.CLASS},
        DoclangToken.SECTION: {DoclangAttributeKey.LEVEL},
        DoclangToken.LIST: {DoclangAttributeKey.ORDERED},
        DoclangToken.GROUP: {DoclangAttributeKey.TYPE},
        DoclangToken.FLOATING_GROUP: {DoclangAttributeKey.CLASS},
        DoclangToken.INLINE: {DoclangAttributeKey.CLASS},
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
            DoclangAttributeKey.ORDERED: {
                DoclangAttributeValue.TRUE,
                DoclangAttributeValue.FALSE,
            }
        },
        DoclangToken.CHECKBOX: {
            DoclangAttributeKey.CLASS: {
                DoclangAttributeValue.SELECTED,
                DoclangAttributeValue.UNSELECTED,
            }
        },
        DoclangToken.INLINE: {
            DoclangAttributeKey.CLASS: {
                DoclangAttributeValue.FORMULA,
                DoclangAttributeValue.CODE,
                DoclangAttributeValue.PICTURE,
            }
        },
        DoclangToken.FLOATING_GROUP: {
            DoclangAttributeKey.CLASS: {
                DoclangAttributeValue.DOCUMENT_INDEX,
                DoclangAttributeValue.TABLE,
                DoclangAttributeValue.PICTURE,
                DoclangAttributeValue.FORM,
                DoclangAttributeValue.CODE,
            }
        },
        # Other attributes (e.g., level, type, id) are not enumerated here
    }

    ALLOWED_ATTRIBUTE_RANGE: ClassVar[dict[DoclangToken, dict["DoclangAttributeKey", tuple[int, int]]]] = {
        # Geometric: value in [0, res]; resolution optional.
        # Keep conservative defaults aligned with existing usage.
        DoclangToken.LOCATION: {
            DoclangAttributeKey.VALUE: (0, DOCLANG_DFLT_RESOLUTION),  # TODO: review
            DoclangAttributeKey.RESOLUTION: (DOCLANG_DFLT_RESOLUTION, DOCLANG_DFLT_RESOLUTION),  # TODO: review
        },
        # Temporal components
        DoclangToken.HOUR: {DoclangAttributeKey.VALUE: (0, 99)},
        DoclangToken.MINUTE: {DoclangAttributeKey.VALUE: (0, 59)},
        DoclangToken.SECOND: {DoclangAttributeKey.VALUE: (0, 59)},
        DoclangToken.CENTISECOND: {DoclangAttributeKey.VALUE: (0, 99)},
        # Levels (N ≥ 1)
        DoclangToken.HEADING: {DoclangAttributeKey.LEVEL: (1, 6)},
        DoclangToken.FORM_HEADING: {DoclangAttributeKey.LEVEL: (1, 6)},
        DoclangToken.SECTION: {DoclangAttributeKey.LEVEL: (1, 6)},
        # Continuation markers (id length constraints)
        DoclangToken.THREAD: {DoclangAttributeKey.ID: (1, 10)},
        DoclangToken.H_THREAD: {DoclangAttributeKey.ID: (1, 10)},
    }

    # Self-closing tokens set
    IS_SELFCLOSING: ClassVar[set[DoclangToken]] = {
        DoclangToken.PAGE_BREAK,
        DoclangToken.TIME_BREAK,
        DoclangToken.LOCATION,
        DoclangToken.HOUR,
        DoclangToken.MINUTE,
        DoclangToken.SECOND,
        DoclangToken.CENTISECOND,
        DoclangToken.BR,
        DoclangToken.CHECKBOX,
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
        # Temporal
        DoclangToken.HOUR: DoclangCategory.TEMPORAL,
        DoclangToken.MINUTE: DoclangCategory.TEMPORAL,
        DoclangToken.SECOND: DoclangCategory.TEMPORAL,
        DoclangToken.CENTISECOND: DoclangCategory.TEMPORAL,
        # Semantic
        DoclangToken.TITLE: DoclangCategory.SEMANTIC,
        DoclangToken.HEADING: DoclangCategory.SEMANTIC,
        DoclangToken.TEXT: DoclangCategory.SEMANTIC,
        DoclangToken.CAPTION: DoclangCategory.SEMANTIC,
        DoclangToken.FOOTNOTE: DoclangCategory.SEMANTIC,
        DoclangToken.PAGE_HEADER: DoclangCategory.SEMANTIC,
        DoclangToken.PAGE_FOOTER: DoclangCategory.SEMANTIC,
        DoclangToken.WATERMARK: DoclangCategory.SEMANTIC,
        DoclangToken.PICTURE: DoclangCategory.SEMANTIC,
        DoclangToken.FORM: DoclangCategory.SEMANTIC,
        DoclangToken.FORM_ITEM: DoclangCategory.SEMANTIC,
        DoclangToken.FORM_HEADING: DoclangCategory.SEMANTIC,
        DoclangToken.FORM_TEXT: DoclangCategory.SEMANTIC,
        DoclangToken.HINT: DoclangCategory.SEMANTIC,
        DoclangToken.FORMULA: DoclangCategory.SEMANTIC,
        DoclangToken.CODE: DoclangCategory.SEMANTIC,
        DoclangToken.LIST_TEXT: DoclangCategory.SEMANTIC,
        DoclangToken.CHECKBOX: DoclangCategory.SEMANTIC,
        DoclangToken.OTSL: DoclangCategory.SEMANTIC,
        # Grouping
        DoclangToken.SECTION: DoclangCategory.GROUPING,
        DoclangToken.LIST: DoclangCategory.GROUPING,
        DoclangToken.GROUP: DoclangCategory.GROUPING,
        DoclangToken.FLOATING_GROUP: DoclangCategory.GROUPING,
        DoclangToken.INLINE: DoclangCategory.GROUPING,
        # Formatting
        DoclangToken.BOLD: DoclangCategory.FORMATTING,
        DoclangToken.ITALIC: DoclangCategory.FORMATTING,
        DoclangToken.STRIKETHROUGH: DoclangCategory.FORMATTING,
        DoclangToken.SUPERSCRIPT: DoclangCategory.FORMATTING,
        DoclangToken.SUBSCRIPT: DoclangCategory.FORMATTING,
        DoclangToken.RTL: DoclangCategory.FORMATTING,
        DoclangToken.BR: DoclangCategory.FORMATTING,
        # Structural
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
        DoclangToken.KEY: DoclangCategory.STRUCTURAL,
        DoclangToken.IMPLICIT_KEY: DoclangCategory.STRUCTURAL,
        DoclangToken.VALUE: DoclangCategory.STRUCTURAL,
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
    def get_category(cls, token: DoclangToken) -> DoclangCategory:
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
    def create_doclang_root(cls, *, version: str = DOCLANG_VERSION, closing: bool = False) -> str:
        """Create the document root tag.

        - When `closing` is True, returns the closing root tag.
        - When a `version` is provided, includes it as an attribute.
        - Otherwise returns a bare opening root tag.
        """
        if closing:
            return f"</{DoclangToken.DOCUMENT.value}>"
        elif version:
            return f'<{DoclangToken.DOCUMENT.value} {DoclangAttributeKey.VERSION.value}="{version}">'
        else:
            # Version attribute is optional; emit bare root tag when not provided
            return f"<{DoclangToken.DOCUMENT.value}>"

    @classmethod
    def create_threading_token(cls, *, id: str, horizontal: bool = False) -> str:
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
    def create_floating_group_token(cls, *, value: DoclangAttributeValue, closing: bool = False) -> str:
        """Create a floating group tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag with a class attribute derived from `value`.
        """
        if closing:
            return f"</{DoclangToken.FLOATING_GROUP.value}>"
        else:
            return f'<{DoclangToken.FLOATING_GROUP.value} {DoclangAttributeKey.CLASS.value}="{value.value}">'

    @classmethod
    def create_list_token(cls, *, ordered: bool, closing: bool = False) -> str:
        """Create a list tag.

        - When `closing` is True, returns the closing tag.
        - Otherwise returns an opening tag with an `ordered` boolean attribute.
        """
        if closing:
            return f"</{DoclangToken.LIST.value}>"
        elif ordered:
            return (
                f'<{DoclangToken.LIST.value} {DoclangAttributeKey.ORDERED.value}="{DoclangAttributeValue.TRUE.value}">'
            )
        else:
            return (
                f'<{DoclangToken.LIST.value} {DoclangAttributeKey.ORDERED.value}="{DoclangAttributeValue.FALSE.value}">'
            )

    @classmethod
    def create_heading_token(cls, *, level: int, closing: bool = False) -> str:
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
    def create_location_token(cls, *, value: int, resolution: int) -> str:
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
    def create_selfclosing_token(
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
    def create_checkbox_token(cls, selected: bool) -> str:
        """Create a checkbox token."""
        return cls.create_selfclosing_token(
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


_DEFAULT_CONTENT_TYPES: set[ContentType] = set(ContentType)


class DoclangParams(CommonParams):
    """Doclang-specific serialization parameters independent of Doclang."""

    # Geometry & content controls (aligned with Doclang defaults)
    xsize: int = DOCLANG_DFLT_RESOLUTION
    ysize: int = DOCLANG_DFLT_RESOLUTION
    add_location: bool = True
    add_table_cell_location: bool = False

    add_referenced_caption: bool = True
    add_referenced_footnote: bool = True

    add_page_break: bool = True

    # types of content to serialize:
    content_types: set[ContentType] = _DEFAULT_CONTENT_TYPES

    # Doclang formatting
    do_self_closing: bool = True
    pretty_indentation: Optional[str] = 2 * " "  # None means minimized serialization, "" means no indentation

    preserve_empty_non_selfclosing: bool = True
    # XML compliance: escape special characters in text content
    escape_mode: EscapeMode = EscapeMode.CDATA_WHEN_NEEDED
    content_wrapping_mode: WrapMode = WrapMode.WRAP_WHEN_NEEDED


def _get_delim(*, params: DoclangParams) -> str:
    """Return record delimiter based on DoclangSerializationMode."""
    return "" if params.pretty_indentation is None else "\n"


def _escape_text(text: str, params: DoclangParams) -> str:
    do_wrap = params.content_wrapping_mode == WrapMode.WRAP_ALWAYS or (
        params.content_wrapping_mode == WrapMode.WRAP_WHEN_NEEDED and text != text.strip()
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
        ``<list_text>``), so this serializer remains agnostic of item types.

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
                DoclangVocabulary.create_list_token(ordered=True)
                if item.first_item_is_enumerated(doc)
                else DoclangVocabulary.create_list_token(ordered=False)
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

    def _serialize_single_item(
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
        # - SectionHeaderItem: use <heading level="N"> ... </heading>.
        # - Other text-like items: map the label to an DoclangToken; for
        #   list items, this maps to <list_text> and keeps the text serializer
        #   free of type-based special casing.
        wrap_open_token: Optional[str]
        selected_token: str = ""
        if isinstance(item, SectionHeaderItem):
            wrap_open_token = DoclangVocabulary.create_heading_token(level=item.level)
        elif isinstance(item, ListItem):
            tok = DoclangToken.LIST_TEXT
            wrap_open_token = f"<{tok.value}>"
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
            tok = DoclangToken.TEXT
            wrap_open_token = None
            selected_token = DoclangVocabulary.create_checkbox_token(
                selected=(item.label == DocItemLabel.CHECKBOX_SELECTED)
            )
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

        if params.add_location:
            # Use Doclang `<location>` tokens instead of `<loc_.../>`
            loc = _create_location_tokens_for_item(item=item, doc=doc, xres=params.xsize, yres=params.ysize)
            if loc:
                parts.append(loc)

        if selected_token:
            parts.append(selected_token)

        if item.meta:
            meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
            if meta_res.text:
                parts.append(meta_res.text)

        if (
            (isinstance(item, CodeItem) and ContentType.TEXT_CODE in params.content_types)
            or (isinstance(item, FormulaItem) and ContentType.TEXT_FORMULA in params.content_types)
            or (not isinstance(item, CodeItem | FormulaItem) and ContentType.TEXT_OTHER in params.content_types)
        ):
            # Check if we should serialize a single inline group child instead of text
            if len(item.children) > 0 and isinstance((first_child := item.children[0].resolve(doc)), InlineGroup):
                ser_res = doc_serializer.serialize(item=first_child, visited=my_visited, **kwargs)
                text_part = ser_res.text
            else:
                text_part = _escape_text(item.text, params)
                text_part = doc_serializer.post_process(
                    text=text_part,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
                )

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
        if wrap_open_token is not None and not (is_inline_scope and item.label == DocItemLabel.TEXT):
            text_res = _wrap_token(text=text_res, open_token=wrap_open_token)
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

        open_token: str = DoclangVocabulary.create_floating_group_token(value=DoclangAttributeValue.PICTURE)
        close_token: str = DoclangVocabulary.create_floating_group_token(
            value=DoclangAttributeValue.PICTURE, closing=True
        )

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

            if item.image and item.image.uri:
                body += _wrap(text=str(item.image.uri), wrap_tag=DoclangToken.URI.value)

            is_chart = self._picture_is_chart(item)
            if ((not is_chart) and ContentType.PICTURE in params.content_types) or (
                is_chart and ContentType.CHART in params.content_types
            ):
                if item.meta:
                    meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
                    if meta_res.text:
                        picture_inner_parts.append(meta_res.text)
                        res_parts.append(meta_res)

                # handle tabular chart data
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
                    otsl_payload = _wrap(text=otsl_content, wrap_tag=DoclangToken.OTSL.value)
                    body += otsl_payload

            if body:
                picture_inner_parts.append(body)
                res_parts.append(create_ser_result(text=body, span_source=item))

        picture_text = "".join(picture_inner_parts)
        if picture_text:
            picture_text = _wrap(text=picture_text, wrap_tag=DoclangToken.PICTURE.value)

        # Build footnotes (as siblings of the picture within the floating_group)
        footnote_text = ""
        if params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        # Compose final structure for picture group:
        # <floating_group class="picture"> [<caption>] <picture>...</picture> [<footnote>...] </floating_group>
        composed_inner = f"{caption_text}{picture_text}{footnote_text}"
        text_res = f"{open_token}{composed_inner}{close_token}"

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
                            parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.CHED))
                        elif cell.row_header:
                            parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.RHED))
                        elif cell.row_section:
                            parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.SROW))
                        else:
                            parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.FCEL))

                        if cell_loc:
                            parts.append(cell_loc)
                        if ContentType.TABLE_CELL in params.content_types:
                            # Apply XML escaping to table cell content
                            if not isinstance(cell, RichTableCell):
                                content = _escape_text(content, params)
                            parts.append(content)
                    else:
                        parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.ECEL))
                elif rowstart != i and colspan == 1:  # FIXME: I believe we should have colstart == j
                    parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.UCEL))
                elif colstart != j and rowspan == 1:  # FIXME: I believe we should have rowstart == i
                    parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.LCEL))
                else:
                    parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.XCEL))

            parts.append(DoclangVocabulary.create_selfclosing_token(token=DoclangToken.NL))

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

        # FIXME: we might need to check the label to distinguish between TABLE and DOCUMENT_INDEX label
        open_token: str = DoclangVocabulary.create_floating_group_token(value=DoclangAttributeValue.TABLE)
        close_token: str = DoclangVocabulary.create_floating_group_token(
            value=DoclangAttributeValue.TABLE, closing=True
        )

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
                otsl_payload = _wrap(text=body, wrap_tag=DoclangToken.OTSL.value)
                res_parts.append(create_ser_result(text=body, span_source=item))

        # Footnote as sibling of the OTSL payload within the floating group
        footnote_text = ""
        if params.add_referenced_footnote:
            ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
            if ftn_res.text:
                footnote_text = ftn_res.text
                res_parts.append(ftn_res)

        composed_inner = f"{caption_text}{otsl_payload}{footnote_text}"
        text_res = f"{open_token}{composed_inner}{close_token}"

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

        if item.parent is None or not isinstance(item.parent.resolve(doc), TextItem):
            # if "unwrapped", wrap in <text>...</text>
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
        if isinstance(item, ListGroup | InlineGroup):
            parts = doc_serializer.get_parts(item=item, **kwargs)
            text_res = "\n".join([p.text for p in parts if p.text])
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
                                item=cap, doc=self.doc, xres=params.xsize, yres=params.ysize
                            )
                            results.append(create_ser_result(text=loc_txt))
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

                    content = ""
                    if ftn.text and ContentType.REF_FOOTNOTE in params.content_types:
                        content = _escape_text(ftn.text, params)

                    text_res = f"{location}{content}"
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

        open_token: str = DoclangVocabulary.create_doclang_root()
        head = self._create_head()
        close_token: str = DoclangVocabulary.create_doclang_root(closing=True)

        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            # Always emit well-formed page breaks using the vocabulary
            page_sep = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.PAGE_BREAK)
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        text_res = f"{open_token}{head}{text_res}{close_token}"

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
            text_res = "\n".join([line for line in text_res.split("\n") if line.strip()])

            if self.params.preserve_empty_non_selfclosing:
                # Expand self-closing forms for tokens that are not allowed
                # to be self-closing according to the vocabulary.
                # Example: <list_text/> -> <list_text></list_text>
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
    _page_no: int = PrivateAttr(default=0)
    _default_resolution: int = PrivateAttr(default=DOCLANG_DFLT_RESOLUTION)

    def deserialize(
        self,
        *,
        text: str,
    ) -> DoclingDocument:
        """Deserialize Doclang XML into a DoclingDocument.

        Args:
            text: Doclang XML string to parse.

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
        # TODO revise need for default page & resolution
        # Initialize with a default page so location tokens can be re-emitted
        self._page_no = 0
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
            DoclangToken.TITLE.value,
            DoclangToken.TEXT.value,
            DoclangToken.CAPTION.value,
            DoclangToken.FOOTNOTE.value,
            DoclangToken.PAGE_HEADER.value,
            DoclangToken.PAGE_FOOTER.value,
            DoclangToken.CODE.value,
            DoclangToken.FORMULA.value,
            DoclangToken.LIST_TEXT.value,
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
        elif name == DoclangToken.FLOATING_GROUP.value:
            self._parse_floating_group(doc=doc, el=el, parent=parent)
        elif name == DoclangToken.INLINE.value:
            self._parse_inline_group(doc=doc, el=el, parent=parent)
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
                    DoclangToken.BR.value,
                    DoclangToken.BOLD.value,
                    DoclangToken.ITALIC.value,
                    DoclangToken.UNDERLINE.value,
                    DoclangToken.STRIKETHROUGH.value,
                    DoclangToken.SUBSCRIPT.value,
                    DoclangToken.SUPERSCRIPT.value,
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
        """Parse text-like tokens (title, text, caption, footnotes, code, formula)."""
        element_children = [
            node for node in el.childNodes if isinstance(node, Element) and node.tagName != DoclangToken.LOCATION.value
        ]

        if len(element_children) > 1 or self._get_children_simple_text_block(el) is None:
            self._parse_inline_group(doc=doc, el=el, parent=parent)
            return

        prov_list = self._extract_provenance(doc=doc, el=el)
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
                DoclangToken.LIST_TEXT.value: DocItemLabel.TEXT,
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
            item = doc.add_text(
                label=text_label_map[nm],
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
            )
            for p in prov_list[1:]:
                item.prov.append(p)

        elif nm == DoclangToken.TITLE.value:
            item = doc.add_title(
                text=text,
                parent=parent,
                prov=(prov_list[0] if prov_list else None),
                formatting=formatting,
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

    def _parse_list(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        ordered = el.getAttribute(DoclangAttributeKey.ORDERED.value) == DoclangAttributeValue.TRUE.value
        li_group = doc.add_list_group(parent=parent)
        actual_children = [
            ch for ch in el.childNodes if isinstance(ch, Element) and ch.tagName not in {DoclangToken.LOCATION.value}
        ]
        boundaries = [
            i
            for i, n in enumerate(actual_children)
            if isinstance(n, Element) and n.tagName == DoclangToken.LIST_TEXT.value
        ]
        ranges = [
            (
                boundaries[i],
                (boundaries[i + 1] if i < len(boundaries) - 1 else len(actual_children)),
            )
            for i in range(len(boundaries))
        ]
        for start, end in ranges:
            if end - start == 1:
                child = actual_children[start]
                actual_grandchildren = [
                    ch
                    for ch in child.childNodes
                    if (isinstance(ch, Element) and ch.tagName != DoclangToken.LOCATION.value)
                    or (isinstance(ch, Text) and ch.data.strip())
                ]
                prov_list = self._extract_provenance(doc=doc, el=child)
                if len(actual_grandchildren) == 1 and isinstance(actual_grandchildren[0], Text):
                    doc.add_list_item(
                        text=self._get_text(child).strip(),
                        parent=li_group,
                        enumerated=ordered,
                        prov=(prov_list[0] if prov_list else None),
                    )
                else:
                    li = doc.add_list_item(
                        text="",
                        parent=li_group,
                        enumerated=ordered,
                        prov=(prov_list[0] if prov_list else None),
                    )
                    for el2 in actual_children[start:end]:
                        self._dispatch_element(doc=doc, el=el2, parent=li)
            else:
                if (
                    actual_children[start + 1].tagName == DoclangToken.LIST.value
                    and len(actual_children[start].childNodes) == 1
                    and isinstance(actual_children[start].childNodes[0], Text)
                ):
                    text = self._get_text(actual_children[start])
                    start_to_use = start + 1
                else:
                    text = ""
                    start_to_use = start

                # TODO add provenance
                wrapper = doc.add_list_item(text=text, parent=li_group, enumerated=ordered)
                for el in actual_children[start_to_use:end]:
                    self._dispatch_element(doc=doc, el=el, parent=wrapper)

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

    # ------------- Floating groups -------------
    def _parse_floating_group(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        cls_val = el.getAttribute(DoclangAttributeKey.CLASS.value)
        if cls_val == DoclangAttributeValue.TABLE.value:
            self._parse_table_group(doc=doc, el=el, parent=parent)
        elif cls_val == DoclangAttributeValue.PICTURE.value:
            self._parse_picture_group(doc=doc, el=el, parent=parent)
        else:
            self._walk_children(doc=doc, el=el, parent=parent)

    def _parse_table_group(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        caption = self._extract_caption(doc=doc, el=el)
        footnotes = self._extract_footnotes(doc=doc, el=el)
        otsl_el = self._first_child(el, DoclangToken.OTSL.value)
        if otsl_el is None:
            tbl = doc.add_table(data=TableData(), caption=caption, parent=parent)
            for ftn in footnotes:
                tbl.footnotes.append(ftn.get_ref())
            return
        # Extract table provenance from <otsl> leading <location/> tokens
        tbl_provs = self._extract_provenance(doc=doc, el=otsl_el)
        # Get inner XML excluding location tokens (work directly with parsed DOM)
        inner = self._inner_xml(otsl_el, exclude_tags={"location"})
        tbl = doc.add_table(
            data=TableData(),
            caption=caption,
            parent=parent,
            prov=(tbl_provs[0] if tbl_provs else None),
        )
        tbl_content = _wrap(text=inner, wrap_tag=DoclangToken.OTSL.value)
        td = self._parse_otsl_table_content(otsl_content=tbl_content, doc=doc, parent=tbl)
        tbl.data = td
        for p in tbl_provs[1:]:
            tbl.prov.append(p)
        for ftn in footnotes:
            tbl.footnotes.append(ftn.get_ref())

    def _parse_picture_group(self, *, doc: DoclingDocument, el: Element, parent: Optional[NodeItem]) -> None:
        # Extract caption from the floating group
        caption = self._extract_caption(doc=doc, el=el)
        footnotes = self._extract_footnotes(doc=doc, el=el)

        # Extract provenance from the <picture> block (locations appear inside it)
        prov_list: list[ProvenanceItem] = []
        picture_el = self._first_child(el, DoclangToken.PICTURE.value)
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
        for ftn in footnotes:
            pic.footnotes.append(ftn.get_ref())

        # If there is a <picture> child and it contains an <otsl>,
        # parse it as TabularChartMetaField and attach to picture.meta
        if picture_el is not None:
            otsl_el = self._first_child(picture_el, DoclangToken.OTSL.value)
            if otsl_el is not None:
                inner = self._inner_xml(otsl_el, exclude_tags={"location"})
                td = self._parse_otsl_table_content(_wrap(inner, DoclangToken.OTSL.value))
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
                if node.tagName not in exclude_tags:
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

        fcel = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.FCEL)
        ecel = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.ECEL)
        lcel = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.LCEL)
        ucel = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.UCEL)
        xcel = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.XCEL)
        nl = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.NL)
        ched = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.CHED)
        rhed = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.RHED)
        srow = DoclangVocabulary.create_selfclosing_token(token=DoclangToken.SROW)

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
                    for child_node in root_el.childNodes:
                        if isinstance(child_node, Element):
                            # Dispatch to parse this element (creates items as side effect)
                            self._dispatch_element(doc=doc, el=child_node, parent=parent)
                            break  # Only process first element

                    # Check if a new child was added
                    if len(parent.children) > children_before:
                        # Get the newly created item
                        child_item = parent.children[-1].resolve(doc=doc)
                        # Create a RichTableCell with reference to the parsed content
                        table_cells.append(
                            RichTableCell(
                                text=cell_text_stripped,
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
        self, otsl_content: str, doc: Optional["DoclingDocument"] = None, parent: Optional[NodeItem] = None
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
