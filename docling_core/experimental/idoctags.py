"""Define classes for DocTags serialization."""

import re
from enum import Enum
from typing import Any, ClassVar, Final, Optional
from xml.dom.minidom import parseString

from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseListSerializer,
    BaseMetaSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.doctags import (
    DocTagsDocSerializer,
    DocTagsParams,
    _get_delim,
)
from docling_core.types.doc import (
    BaseMeta,
    CodeItem,
    DescriptionMetaField,
    DocItem,
    DoclingDocument,
    FloatingItem,
    InlineGroup,
    ListGroup,
    ListItem,
    MetaFieldName,
    MoleculeMetaField,
    NodeItem,
    PictureClassificationMetaField,
    PictureItem,
    SectionHeaderItem,
    SummaryMetaField,
    TableData,
    TableItem,
    TabularChartMetaField,
    TextItem,
)

# Note: Intentionally avoid importing DocumentToken here to ensure
# IDocTags uses only its own token vocabulary.

DOCTAGS_VERSION: Final = "1.0.0"
DOCTAGS_RESOLUTION: int = 512


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


def _wrap_token(*, text: str, open_token: str) -> str:
    close_token = IDocTagsVocabulary.create_closing_token(token=open_token)
    return f"{open_token}{text}{close_token}"


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
    *, item: "DocItem", doc: "DoclingDocument", xres: int = 512, yres: int = 512
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
    # LIST_ITEM = "list_item"
    CHECKBOX = "checkbox"
    OTSL = "otsl"  # this will take care of the structure in the table.

    # Grouping
    SECTION = "section"
    LIST = "list"
    GROUP = "group"
    FLOATING_GROUP = "floating_group"
    # ORDERED_LIST = "ordered_list"
    # UNORDERED_LIST = "unordered_list"

    # Formatting
    BOLD = "bold"
    ITALIC = "italic"
    STRIKETHROUGH = "strikethrough"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"
    RTL = "rtl"
    INLINE = "inline"
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


class IDocTagsTableToken(str, Enum):
    """Serialized OTSL table tokens used in DocTags output."""

    OTSL_ECEL = f"<{IDocTagsToken.ECEL.value}/>"  # empty cell
    OTSL_FCEL = f"<{IDocTagsToken.FCEL.value}/>"  # cell with content
    OTSL_LCEL = f"<{IDocTagsToken.LCEL.value}/>"  # left looking cell,
    OTSL_UCEL = f"<{IDocTagsToken.UCEL.value}/>"  # up looking cell,
    OTSL_XCEL = f"<{IDocTagsToken.XCEL.value}/>"  # 2d extension cell (cross cell),
    OTSL_NL = f"<{IDocTagsToken.NL.value}/>"  # new line,
    OTSL_CHED = f"<{IDocTagsToken.CHED.value}/>"  # - column header cell,
    OTSL_RHED = f"<{IDocTagsToken.RHED.value}/>"  # - row header cell,
    OTSL_SROW = f"<{IDocTagsToken.SROW.value}/>"  # - section row cell


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
            IDocTagsAttributeKey.VALUE: (0, 512),
            IDocTagsAttributeKey.RESOLUTION: (512, 512),
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
    def create_location_token(cls, *, value: int, resolution: int = 512) -> str:
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
            raise ValueError(f"resolution must be in [{r_lo}, {r_hi}]")

        v_lo, v_hi = range_map[IDocTagsAttributeKey.VALUE]
        if not (v_lo <= value <= v_hi):
            raise ValueError(f"value must be in [{v_lo}, {v_hi}]")
        if not (0 <= value <= resolution):
            raise ValueError("value must be in [0, resolution]")

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


class IDocTagsParams(DocTagsParams):
    """DocTags-specific serialization parameters."""

    do_self_closing: bool = True
    pretty_indentation: Optional[str] = 2 * " "
    # When True, convert any self-closing form of non-self-closing tokens
    # (e.g., <list_text/>) into expanded pairs (<list_text></list_text>)
    # after pretty-printing. This prevents XML pretty-printers from
    # collapsing empty elements that must not be self-closing according
    # to the IDocTags vocabulary.
    preserve_empty_non_selfclosing: bool = True


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
        siblings without an enclosing ``<list_item>`` wrapper, while structural
        wrappers are still preserved even when content is suppressed.

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
        child_results_wrapped: list[str] = []

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
                    child_results_wrapped.append(sub_res.text)
                item_results.append(sub_res)
                continue

            # Normal case: ListItem under ListGroup
            if not isinstance(child, ListItem):
                continue
            if child.self_ref in my_visited or child.self_ref in excluded:
                continue

            my_visited.add(child.self_ref)

            # Serialize the list item content (DocTagsTextSerializer will not wrap it)
            child_res = doc_serializer.serialize(
                item=child,
                list_level=list_level + 1,
                is_inline_scope=is_inline_scope,
                visited=my_visited,
                **kwargs,
            )
            item_results.append(child_res)
            # Wrap the content into <list_text>, without any nested list content.
            child_text_wrapped = _wrap(
                text=f"{child_res.text}",
                wrap_tag=IDocTagsToken.LIST_TEXT.value,
            )
            child_results_wrapped.append(child_text_wrapped)

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
                        child_results_wrapped.append(sub_res.text)
                    item_results.append(sub_res)

        delim = _get_delim(params=params)
        if child_results_wrapped:
            text_res = delim.join(child_results_wrapped)
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
        # - ListItem: do not wrap here (list serializer wraps children as <list_text>).
        # - SectionHeaderItem: use <heading level="N"> ... </heading>.
        # - Other text-like items: require a valid IDocTagsToken from the item's label.
        #   If the label is not a known IDocTags token, raise for explicitness.
        wrap_open_token: Optional[str]
        if isinstance(item, ListItem):
            wrap_open_token = None
        elif isinstance(item, SectionHeaderItem):
            wrap_open_token = IDocTagsVocabulary.create_heading_token(level=item.level)
        else:
            label_value = str(item.label)
            try:
                tok = IDocTagsToken(label_value)
            except ValueError:
                raise ValueError(
                    f"Unsupported IDocTags token for label '{label_value}'"
                )
            wrap_open_token = f"<{tok.value}>"
        parts: list[str] = []

        if item.meta:
            meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
            if meta_res.text:
                parts.append(meta_res.text)

        if params.add_location:
            # Use IDocTags `<location>` tokens instead of `<loc_.../>`
            loc = _create_location_tokens_for_item(item=item, doc=doc)
            if loc:
                parts.append(loc)

        if params.add_content:
            if (
                item.text == ""
                and len(item.children) == 1
                and isinstance(
                    (child_group := item.children[0].resolve(doc)), InlineGroup
                )
            ):
                ser_res = doc_serializer.serialize(item=child_group, visited=my_visited)
                text_part = ser_res.text
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
                # Keep the textual code content as-is
                text_part = text_part
            else:
                text_part = text_part.strip()

            if text_part:
                parts.append(text_part)

        if params.add_caption and isinstance(item, FloatingItem):
            cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
            if cap_text:
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
                    and (tmp := self._serialize_meta_field(item.meta, key))
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

    def _serialize_meta_field(self, meta: BaseMeta, name: str) -> Optional[str]:
        if (field_val := getattr(meta, name)) is not None:
            if name == MetaFieldName.SUMMARY and isinstance(
                field_val, SummaryMetaField
            ):
                txt = f"<summary>{field_val.text}</summary>"
            elif name == MetaFieldName.DESCRIPTION and isinstance(
                field_val, DescriptionMetaField
            ):
                txt = f"<description>{field_val.text}</description>"
            elif name == MetaFieldName.CLASSIFICATION and isinstance(
                field_val, PictureClassificationMetaField
            ):
                class_name = self._humanize_text(
                    field_val.get_main_prediction().class_name
                )
                txt = f"<classification>{class_name}</classification>"
            elif name == MetaFieldName.MOLECULE and isinstance(
                field_val, MoleculeMetaField
            ):
                txt = f"<molecule>{field_val.smi}</molecule>"
            elif name == MetaFieldName.TABULAR_CHART and isinstance(
                field_val, TabularChartMetaField
            ):
                # suppressing tabular chart serialization
                return None
            # elif tmp := str(field_val or ""):
            #     txt = tmp
            elif name not in {v.value for v in MetaFieldName}:
                txt = _wrap(text=str(field_val or ""), wrap_tag=name)
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

        res_parts: list[SerializationResult] = []

        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):

            if item.meta:
                meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
                if meta_res.text:
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
                otsl_content = temp_table.export_to_otsl(
                    temp_doc,
                    add_cell_location=False,
                    # Suppress chart cell text if global content is off
                    add_cell_text=params.add_content,
                    self_closing=params.do_self_closing,
                    table_token=IDocTagsTableToken,
                )
                body += otsl_content
            res_parts.append(create_ser_result(text=body, span_source=item))

        text_res = "".join([r.text for r in res_parts])
        if text_res:
            text_res = (
                open_token
                + _wrap(text=text_res, wrap_tag=IDocTagsToken.PICTURE.value)
                + close_token
            )

        return create_ser_result(text=text_res, span_source=res_parts)


class IDocTagsTableSerializer(BaseTableSerializer):
    """DocTags-specific table item serializer."""

    def _get_table_token(self) -> Any:
        return IDocTagsTableToken

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

        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):

            if params.add_location:
                loc_text = _create_location_tokens_for_item(item=item, doc=doc)
                res_parts.append(create_ser_result(text=loc_text, span_source=item))

            otsl_text = item.export_to_otsl(
                doc=doc,
                add_cell_location=params.add_table_cell_location,
                # Suppress cell text when global content is disabled
                add_cell_text=(params.add_table_cell_text and params.add_content),
                xsize=params.xsize,
                ysize=params.ysize,
                visited=visited,
                table_token=self._get_table_token(),
            )
            res_parts.append(create_ser_result(text=otsl_text, span_source=item))

        text_res = "".join([r.text for r in res_parts])
        if text_res:
            text_res = (
                open_token
                + _wrap(text=text_res, wrap_tag=IDocTagsToken.OTSL.value)
                + close_token
            )

        return create_ser_result(text=text_res, span_source=res_parts)


class IDocTagsDocSerializer(DocTagsDocSerializer):
    """DocTags document serializer."""

    text_serializer: BaseTextSerializer = IDocTagsTextSerializer()
    table_serializer: BaseTableSerializer = IDocTagsTableSerializer()
    picture_serializer: BasePictureSerializer = IDocTagsPictureSerializer()
    # key_value_serializer: BaseKeyValueSerializer = DocTagsKeyValueSerializer()
    # form_serializer: BaseFormSerializer = DocTagsFormSerializer()
    # fallback_serializer: BaseFallbackSerializer = DocTagsFallbackSerializer()

    list_serializer: BaseListSerializer = IDocTagsListSerializer()
    # inline_serializer: BaseInlineSerializer = DocTagsInlineSerializer()

    # annotation_serializer: BaseAnnotationSerializer = DocTagsAnnotationSerializer()

    # picture_serializer: BasePictureSerializer = IDocTagsPictureSerializer()
    meta_serializer: BaseMetaSerializer = IDocTagsMetaSerializer()
    # table_serializer: BaseTableSerializer = IDocTagsTableSerializer()

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
            cap_res = super(DocTagsDocSerializer, self).serialize_captions(
                item, **kwargs
            )
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
        """DocTags-specific document serializer."""
        delim = _get_delim(params=self.params)

        open_token: str = IDocTagsVocabulary.create_doctag_root()
        close_token: str = IDocTagsVocabulary.create_doctag_root(closing=True)

        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            page_sep = f"<{IDocTagsToken.PAGE_BREAK.value}{'/' if self.params.do_self_closing else ''}>"
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        """
        tmp = f"<{IDocTagsToken.DOCUMENT.value}>"
        tmp += f"<{IDocTagsToken.VERSION.value}>{DOCTAGS_VERSION}</{IDocTagsToken.VERSION.value}>"
        tmp += f"{text_res}"
        tmp += f"</{IDocTagsToken.DOCUMENT.value}>"
        """

        text_res = f"{open_token}{text_res}{close_token}"

        if self.params.pretty_indentation and (
            my_root := parseString(text_res).documentElement
        ):
            # Pretty-print using minidom. This may collapse empty elements
            # into self-closing tags. We optionally expand back non-self-closing
            # tokens to preserve the IDocTags contract.
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
