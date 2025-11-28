"""Define classes for DocTags serialization."""

from enum import Enum
from typing import Any, Final, Optional
from xml.dom.minidom import parseString

from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
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
from docling_core.types.doc.tokens import DocumentToken

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


class IDocTagsParams(DocTagsParams):
    """DocTags-specific serialization parameters."""

    do_self_closing: bool = True
    pretty_indentation: Optional[str] = 2 * " "


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
            token = DocumentToken.create_token_name_from_doc_item_label(
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
            page_sep = f"<{DocumentToken.PAGE_BREAK.value}{'/' if self.params.do_self_closing else ''}>"
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        wrap_tag = DocumentToken.DOCUMENT.value
        text_res = f"<{wrap_tag}><version>{DOCTAGS_VERSION}</version>{text_res}{delim}</{wrap_tag}>"

        if self.params.pretty_indentation and (
            my_root := parseString(text_res).documentElement
        ):
            text_res = my_root.toprettyxml(indent=self.params.pretty_indentation)
            text_res = "\n".join(
                [line for line in text_res.split("\n") if line.strip()]
            )
        return create_ser_result(text=text_res, span_source=parts)
