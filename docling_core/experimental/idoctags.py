"""Define classes for DocTags serialization."""

from enum import Enum
from typing import Any, Final, Optional, Tuple
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
from docling_core.types.doc.tokens import (
    _CodeLanguageToken,
    _PictureClassificationToken,
)

DOCTAGS_VERSION: Final = "1.0.0"


class IDocTagsTableToken(str, Enum):
    """Class to represent an LLM friendly representation of a Table."""

    CELL_LABEL_COLUMN_HEADER = "<column_header/>"
    CELL_LABEL_ROW_HEADER = "<row_header/>"
    CELL_LABEL_SECTION_HEADER = "<shed/>"
    CELL_LABEL_DATA = "<data/>"

    OTSL_ECEL = "<ecel/>"  # empty cell
    OTSL_FCEL = "<fcel/>"  # cell with content
    OTSL_LCEL = "<lcel/>"  # left looking cell,
    OTSL_UCEL = "<ucel/>"  # up looking cell,
    OTSL_XCEL = "<xcel/>"  # 2d extension cell (cross cell),
    OTSL_NL = "<nl/>"  # new line,
    OTSL_CHED = "<ched/>"  # - column header cell,
    OTSL_RHED = "<rhed/>"  # - row header cell,
    OTSL_SROW = "<srow/>"  # - section row cell

    @classmethod
    def get_special_tokens(
        cls,
    ):
        """Return all table-related special tokens.

        Includes the opening/closing OTSL tags and each enum token value.
        """
        special_tokens: list[str] = ["<otsl>", "</otsl>"]
        for token in cls:
            special_tokens.append(f"{token.value}")

        return special_tokens


class IDocTagsToken(str, Enum):
    """IDocTagsToken."""

    _LOC_PREFIX = "loc_"
    _SECTION_HEADER_PREFIX = "section_header_level_"

    DOCUMENT = "doctag"
    VERSION = "version"

    OTSL = "otsl"
    ORDERED_LIST = "ordered_list"
    UNORDERED_LIST = "unordered_list"

    PAGE_BREAK = "page_break"

    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PICTURE = "picture"
    SECTION_HEADER = "section_header"
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    DOCUMENT_INDEX = "document_index"
    CODE = "code"
    CHECKBOX_SELECTED = "checkbox_selected"
    CHECKBOX_UNSELECTED = "checkbox_unselected"
    FORM = "form"
    EMPTY_VALUE = "empty_value"  # used for empty value fields in fillable forms

    @classmethod
    def get_special_tokens(
        cls,
        *,
        page_dimension: Tuple[int, int] = (500, 500),
        include_location_tokens: bool = True,
        include_code_class: bool = False,
        include_picture_class: bool = False,
    ):
        """Function to get all special document tokens."""
        special_tokens: list[str] = []
        for token in cls:
            if not token.value.endswith("_"):
                special_tokens.append(f"<{token.value}>")
                special_tokens.append(f"</{token.value}>")

        for i in range(6):
            special_tokens += [
                f"<{IDocTagsToken._SECTION_HEADER_PREFIX.value}{i}>",
                f"</{IDocTagsToken._SECTION_HEADER_PREFIX.value}{i}>",
            ]

        special_tokens.extend(IDocTagsTableToken.get_special_tokens())

        if include_picture_class:
            special_tokens.extend([t.value for t in _PictureClassificationToken])

        if include_code_class:
            special_tokens.extend([t.value for t in _CodeLanguageToken])

        if include_location_tokens:
            # Adding dynamically generated location-tokens
            for i in range(0, max(page_dimension[0], page_dimension[1])):
                special_tokens.append(f"<{IDocTagsToken._LOC_PREFIX.value}{i}/>")

        return special_tokens

    @classmethod
    def create_token_name_from_doc_item_label(cls, label: str, level: int = 1) -> str:
        """Get token corresponding to passed doc item label."""
        doc_token_by_item_label = {
            DocItemLabel.CAPTION: IDocTagsToken.CAPTION,
            DocItemLabel.FOOTNOTE: IDocTagsToken.FOOTNOTE,
            DocItemLabel.FORMULA: IDocTagsToken.FORMULA,
            DocItemLabel.LIST_ITEM: IDocTagsToken.LIST_ITEM,
            DocItemLabel.PAGE_FOOTER: IDocTagsToken.PAGE_FOOTER,
            DocItemLabel.PAGE_HEADER: IDocTagsToken.PAGE_HEADER,
            DocItemLabel.PICTURE: IDocTagsToken.PICTURE,
            DocItemLabel.TABLE: IDocTagsToken.TABLE,
            DocItemLabel.TEXT: IDocTagsToken.TEXT,
            DocItemLabel.TITLE: IDocTagsToken.TITLE,
            DocItemLabel.DOCUMENT_INDEX: IDocTagsToken.DOCUMENT_INDEX,
            DocItemLabel.CODE: IDocTagsToken.CODE,
            DocItemLabel.CHECKBOX_SELECTED: IDocTagsToken.CHECKBOX_SELECTED,
            DocItemLabel.CHECKBOX_UNSELECTED: IDocTagsToken.CHECKBOX_UNSELECTED,
            DocItemLabel.FORM: IDocTagsToken.FORM,
            # Fallback mappings for labels without dedicated tokens in IDocTagsToken
            DocItemLabel.KEY_VALUE_REGION: IDocTagsToken.TEXT,
            DocItemLabel.PARAGRAPH: IDocTagsToken.TEXT,
            DocItemLabel.REFERENCE: IDocTagsToken.TEXT,
            DocItemLabel.CHART: IDocTagsToken.PICTURE,
        }

        res: str
        if label == DocItemLabel.SECTION_HEADER:
            res = f"{IDocTagsToken._SECTION_HEADER_PREFIX}{level}"
        else:
            try:
                res = doc_token_by_item_label[DocItemLabel(label)].value
            except KeyError as e:
                raise RuntimeError(f"Unexpected DocItemLabel: {label}") from e
        return res


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
