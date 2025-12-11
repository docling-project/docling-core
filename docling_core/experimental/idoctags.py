"""Define classes for DocTags serialization."""

from enum import Enum
from typing import Any, Final, Optional
from xml.dom.minidom import parseString

from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseListSerializer,
    BaseMetaSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.doctags import (
    DocTagsDocSerializer,
    DocTagsParams,
    DocTagsPictureSerializer,
    DocTagsTableSerializer,
    _get_delim,
    _wrap,
)
from docling_core.types.doc import (
    BaseMeta,
    DescriptionMetaField,
    DocItem,
    DoclingDocument,
    ListGroup,
    ListItem,
    MetaFieldName,
    MoleculeMetaField,
    NodeItem,
    PictureClassificationMetaField,
    PictureItem,
    SummaryMetaField,
    TableData,
    TabularChartMetaField,
)
from docling_core.types.doc.labels import DocItemLabel

DOCTAGS_VERSION: Final = "1.0.0"


class IDocTagsCategory(str, Enum):
    """IDocTagsCtegory.

    DocTags defines the following categories of elements:

    - **root**: Elements that establish document scope such as `doctag`
    - **special**: Elements that establish document pagination, such `page_break`, and `time_break`.
    - **geometric**: Elements that capture geometric position as normalized coordinates/bounding boxes (via repeated `location`) anchoring block-level content to the page.
    - **temporal**: Elements that capture temporal positions using `<hour value={integer}/><minute value={integer}/><second value={integer}/><centisecond value={integer}/>` for a timestamp and a double timestamp for time intervals.
    - **semantic**: Block-level elements that convey document meaning (e.g., titles, paragraphs, captions, lists, forms, tables, formulas, code, pictures), optionally preceded by location tokens.
    - **formatting**: Inline elements that modify textual presentation within semantic content (e.g., `bold`, `italic`, `strikethrough`, `superscript`, `subscript`, `rtl`, `inline class="formula|code|picture"`, `br`).
    - **grouping**: Elements that organize semantic blocks into logical hierarchies and composites (e.g., `section`, `list`, `group type=*`) and never carry location tokens.
    - **structural**: Sequence tokens that define internal structure for complex constructs (primarily OTSL table layout: `otsl`, `fcel`, `ecel`, `lcel`, `ucel`, `xcel`, `nl`, `ched`, `rhed`, `corn`, `srow`; and form parts like `key`/`value`).
    - **content**: Lightweight content helpers used inside semantic blocks for explicit payload and annotations (e.g., `marker`).
    - **binary data**: Elements that embed or reference non-text payloads for media—either inline as `base64` or via `uri`—allowed under `picture`, `inline class="picture"`, or at page level.
    - **metadata**: Elements that provide metadata about the document or its components, contained within `head` and `meta` respectively.
    - **continuation** tokens: Markers that indicate content spanning pages or table boundaries (e.g., `thread`, `h_thread`, each with a required `id` attribute) to stitch split content (e.g., across columns or pages).
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
    | 6 | Geometric Tokens | `location` | Yes | Yes | `value`, `resolution?` | Geometric coordinate; `value` in [0, res]; optional `resolution`. |
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
    | 19 |  | `picture` | No | No | — | Block image/graphic; at most one of `base64`/`uri`; may include `meta` for classification; `otsl` may encode chart data. |
    | 20 |  | `form` | No | No | — | Form structure container. |
    | 21 |  | `formula` | No | No | — | Mathematical expression block. |
    | 22 |  | `code` | No | No | — | Code block. |
    | 23 |  | `list_text` | No | No | — | List item content. |
    | 24 |  | `checkbox` | No | Yes | `selected` | Checkbox item; optional `selected` in {`true`,`false`} defaults to `false`. |
    | 25 |  | `form_item` | No | No | — | Form item; exactly one `key`; one or more of `value`/`checkbox`/`marker`/`hint`. |
    | 26 |  | `form_heading` | No | Yes | `level?` | Form header; optional `level` (N ≥ 1). |
    | 27 |  | `form_text` | No | No | — | Form text block. |
    | 28 |  | `hint` | No | No | — | Hint for a fillable field (format/example/description). |
    | 29 | Grouping Tokens | `section` | No | Yes | `level` | Document section; `level` (N ≥ 1). |
    | 30 |  | `list` | No | Yes | `ordered` | List container; optional `ordered` in {`true`,`false`} defaults to `false`. |
    | 31 |  | `group` | No | Yes | `type?` | Generic group; no `location` tokens; associates composite content (e.g., captions/footnotes). |
    | 32 |  | `floating_group` | No | Yes | `class` in {`table`,`picture`,`form`,`code`} | Floating container that groups a floating component with its caption, footnotes, and metadata; no `location` tokens. |
    | 33 | Formatting Tokens | `bold` | No | No | — | Bold text. |
    | 34 |  | `italic` | No | No | — | Italic text. |
    | 35 |  | `strikethrough` | No | No | — | Strike-through text. |
    | 36 |  | `superscript` | No | No | — | Superscript text. |
    | 37 |  | `subscript` | No | No | — | Subscript text. |
    | 38 |  | `rtl` | No | No | — | Right-to-left text direction. |
    | 39 |  | `inline` | No | Yes | `class` in {`formula`,`code`,`picture`} | Inline content; if `class="picture"`, may include one of `base64` or `uri`. |
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
    | 52 | Continuation Tokens | `thread` | Yes | Yes | `id` | Continuation marker for split content; reuse same `id` across parts. |
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
    VERSION = "version"
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
    LIST_ITEM = "list_item"
    CHECKBOX = "checkbox"
    OTSL = "otsl"  # this will take care of the structure in the table.

    # Grouping
    SECTION = "section"
    LIST = "list"
    GROUP = "group"
    FLOATING_GROUP = "floating_group"
    ORDERED_LIST = "ordered_list"
    UNORDERED_LIST = "unordered_list"

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

    # Allowed attributes per token
    ALLOWED_ATTRIBUTES: dict["IDocTagsToken", set[str]] = {
        LOCATION: {"value", "resolution"},
        HOUR: {"value"},
        MINUTE: {"value"},
        SECOND: {"value"},
        CENTISECOND: {"value"},
        HEADING: {"level"},
        FORM_HEADING: {"level"},
        CHECKBOX: {"selected"},
        SECTION: {"level"},
        LIST: {"ordered"},
        GROUP: {"type"},
        FLOATING_GROUP: {"class"},
        INLINE: {"class"},
        THREAD: {"id"},
        H_THREAD: {"id"},
    }

    # Allowed values for specific attributes (enumerations)
    # Structure: token -> attribute name -> set of allowed string values
    ALLOWED_ATTRIBUTE_VALUES: dict["IDocTagsToken", dict[str, set[str]]] = {
        # Grouping and inline enumerations
        LIST: {"ordered": {"true", "false"}},
        CHECKBOX: {"selected": {"true", "false"}},
        INLINE: {"class": {"formula", "code", "picture"}},
        FLOATING_GROUP: {
            "class": {"document_index", "table", "picture", "form", "code"}
        },
        # Other attributes (e.g., level, type, id) are not enumerated here
    }

    ALLOWED_ATTRIBUTE_RANGE: dict["IDocTagsToken", dict[str, tuple[int, int]]] = {
        # Geometric: value in [0, res]; resolution optional.
        # Keep conservative defaults aligned with existing usage.
        LOCATION: {
            "value": (0, 512),
            "resolution": (512, 512),
        },
        # Temporal components
        HOUR: {"value": (0, 99)},
        MINUTE: {"value": (0, 59)},
        SECOND: {"value": (0, 59)},
        CENTISECOND: {"value": (0, 99)},
        # Levels (N ≥ 1)
        HEADING: {"level": (1, 6)},
        FORM_HEADING: {"level": (1, 6)},
        SECTION: {"level": (1, 6)},
        # Continuation markers
        THREAD: {"id": (1, 10)},
        H_THREAD: {"id": (1, 10)},
    }

    # Self-closing tokens map
    IS_SELFCLOSING: dict["IDocTagsToken", bool] = {
        PAGE_BREAK: True,
        TIME_BREAK: True,
        LOCATION: True,
        HOUR: True,
        MINUTE: True,
        SECOND: True,
        CENTISECOND: True,
        BR: True,
        # OTSL structural tokens are emitted as self-closing markers
        FCEL: True,
        ECEL: True,
        CHED: True,
        RHED: True,
        CORN: True,
        SROW: True,
        LCEL: True,
        UCEL: True,
        XCEL: True,
        NL: True,
        # Continuation markers
        THREAD: True,
        H_THREAD: True,
    }

    @classmethod
    def create_threading_token(cls, *, id: str, horizontal: bool = False) -> str:
        """Create a continuation threading token.

        Emits `<thread id="..."/>` or `<h_thread id="..."/>` depending on
        the `horizontal` flag. Validates required attributes against the
        class schema and basic value sanity.
        """
        token = cls.H_THREAD if horizontal else cls.THREAD
        # Ensure the required attribute is declared for this token
        assert "id" in cls.ALLOWED_ATTRIBUTES.get(token, set())

        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[token]["id"]
        if not (lo <= level <= hi):
            raise ValueError(f"level must be in [{lo}, {hi}]")

        return f'<{token.value} id="{id}"/>'

    @classmethod
    def create_heading_token(cls, *, level: int, closing: bool = False) -> str:
        """Create a heading tag with validated level.

        When `closing` is False, emits an opening tag with level attribute.
        When `closing` is True, emits the corresponding closing tag.
        """
        lo, hi = cls.ALLOWED_ATTRIBUTE_RANGE[cls.HEADING]["level"]
        if not (lo <= level <= hi):
            raise ValueError(f"level must be in [{lo}, {hi}]")

        if closing:
            return f"</{cls.HEADING.value}>"
        return f'<{cls.HEADING.value} level="{level}">'

    @classmethod
    def create_location_token(cls, *, value: int, resolution: int = 512) -> str:
        """Create a location token with value and resolution.

        Validates both attributes using the configured ranges and ensures
        `value` lies within [0, resolution]. Always emits the resolution
        attribute for explicitness.
        """
        range_map = cls.ALLOWED_ATTRIBUTE_RANGE[cls.LOCATION]
        # Validate resolution if a constraint exists
        r_lo, r_hi = range_map.get("resolution", (resolution, resolution))
        if not (r_lo <= resolution <= r_hi):
            raise ValueError(f"resolution must be in [{r_lo}, {r_hi}]")

        v_lo, v_hi = range_map["value"]
        if not (v_lo <= value <= v_hi):
            raise ValueError(f"value must be in [{v_lo}, {v_hi}]")
        if not (0 <= value <= resolution):
            raise ValueError("value must be in [0, resolution]")

        return f'<{cls.LOCATION.value} value="{value}" resolution="{resolution}"/>'

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

        temporal_tokens = {cls.HOUR, cls.MINUTE, cls.SECOND, cls.CENTISECOND}

        for token in cls:
            # Optional gating for location/temporal tokens
            if not include_location_tokens and token is cls.LOCATION:
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
                    for v in sorted(allowed_vals):
                        if is_selfclosing:
                            special_tokens.append(f'<{name} {attr_name}="{v}"/>')
                        else:
                            special_tokens.append(f'<{name} {attr_name}="{v}">')
                            special_tokens.append(f"</{name}>")

                # Ranged attribute values (emit a conservative, complete range)
                range_map = cls.ALLOWED_ATTRIBUTE_RANGE.get(token, {})
                for attr_name, (lo, hi) in range_map.items():
                    # Keep the list size reasonable by skipping optional resolution enumeration
                    if token is cls.LOCATION and attr_name == "resolution":
                        continue
                    for v in range(lo, hi + 1):
                        if is_selfclosing:
                            special_tokens.append(f'<{name} {attr_name}="{v}"/>')
                        else:
                            special_tokens.append(f'<{name} {attr_name}="{v}">')
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
        # 1) <ordered_list>/<unordered_list> can be children of lists.
        # 2) Do NOT wrap nested lists into <list_item>, even if they are
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
            # Wrap the content into <list_item>, without any nested list content.
            child_text_wrapped = _wrap(
                text=f"{child_res.text}",
                wrap_tag=IDocTagsToken.LIST_ITEM.value,
            )
            child_results_wrapped.append(child_text_wrapped)

            # After the <list_item>, append any nested lists (children of this ListItem)
            # as siblings at the same level (not wrapped in <list_item>).
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
            wrap_tag = (
                IDocTagsToken.ORDERED_LIST.value
                if item.first_item_is_enumerated(doc)
                else IDocTagsToken.UNORDERED_LIST.value
            )
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        else:
            text_res = ""
        return create_ser_result(text=text_res, span_source=item_results)


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


class IDocTagsPictureSerializer(DocTagsPictureSerializer):
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
        params = DocTagsParams(**kwargs)
        res_parts: list[SerializationResult] = []
        is_chart = False

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):

            if item.meta:
                meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
                if meta_res.text:
                    res_parts.append(meta_res)

            body = ""
            if params.add_location:
                body += item.get_location_tokens(
                    doc=doc,
                    xsize=params.xsize,
                    ysize=params.ysize,
                    self_closing=params.do_self_closing,
                )

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

        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        text_res = "".join([r.text for r in res_parts])
        if text_res:
            token = IDocTagsToken.create_token_name_from_doc_item_label(
                label=DocItemLabel.CHART if is_chart else DocItemLabel.PICTURE,
            )
            text_res = _wrap(text=text_res, wrap_tag=token)
        return create_ser_result(text=text_res, span_source=res_parts)


class IDocTagsTableSerializer(DocTagsTableSerializer):
    """DocTags-specific table item serializer."""

    def _get_table_token(self) -> Any:
        return IDocTagsTableToken


class IDocTagsDocSerializer(DocTagsDocSerializer):
    """DocTags document serializer."""

    picture_serializer: BasePictureSerializer = IDocTagsPictureSerializer()
    meta_serializer: BaseMetaSerializer = IDocTagsMetaSerializer()
    table_serializer: BaseTableSerializer = IDocTagsTableSerializer()
    params: IDocTagsParams = IDocTagsParams()

    @override
    def _meta_is_wrapped(self) -> bool:
        return True

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """DocTags-specific document serializer."""
        delim = _get_delim(params=self.params)
        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            page_sep = f"<{IDocTagsToken.PAGE_BREAK.value}{'/' if self.params.do_self_closing else ''}>"
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        tmp = f"<{IDocTagsToken.DOCUMENT.value}>"
        tmp += f"<{IDocTagsToken.VERSION.value}>{DOCTAGS_VERSION}</{IDocTagsToken.VERSION.value}>"
        tmp += f"{text_res}"
        tmp += f"</{IDocTagsToken.DOCUMENT.value}>"

        text_res = tmp

        if self.params.pretty_indentation and (
            my_root := parseString(text_res).documentElement
        ):
            text_res = my_root.toprettyxml(indent=self.params.pretty_indentation)
            text_res = "\n".join(
                [line for line in text_res.split("\n") if line.strip()]
            )

        return create_ser_result(text=text_res, span_source=parts)
