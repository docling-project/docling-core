"""Define classes for Doctags serialization."""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from typing_extensions import override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import (
    CommonParams,
    DocSerializer,
    create_ser_result,
)
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    FloatingItem,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListItem,
    NodeItem,
    OrderedList,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
    PictureTabularChartData,
    TableItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel, PictureClassificationLabel
from docling_core.types.doc.tokens import DocumentToken


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


class DocTagsParams(CommonParams):
    """DocTags-specific serialization parameters."""

    class Mode(str, Enum):
        """DocTags serialization mode."""

        MINIFIED = "minified"
        HUMAN_FRIENDLY = "human_friendly"

    xsize: int = 500
    ysize: int = 500
    add_location: bool = True
    add_caption: bool = True
    add_content: bool = True
    add_table_cell_location: bool = False
    add_table_cell_text: bool = True
    add_page_break: bool = True

    mode: Mode = Mode.HUMAN_FRIENDLY


def _get_delim(params: DocTagsParams) -> str:
    if params.mode == DocTagsParams.Mode.HUMAN_FRIENDLY:
        delim = "\n"
    elif params.mode == DocTagsParams.Mode.MINIFIED:
        delim = ""
    else:
        raise RuntimeError(f"Unknown DocTags mode: {params.mode}")
    return delim


class DocTagsTextSerializer(BaseModel, BaseTextSerializer):
    """DocTags-specific text item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        from docling_core.types.doc.document import SectionHeaderItem

        params = DocTagsParams(**kwargs)
        wrap_tag: Optional[str] = DocumentToken.create_token_name_from_doc_item_label(
            label=item.label,
            **({"level": item.level} if isinstance(item, SectionHeaderItem) else {}),
        )
        parts: list[str] = []

        if params.add_location:
            location = item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
            )
            if location:
                parts.append(location)

        if params.add_content:
            text_part = item.text
            text_part = doc_serializer.post_process(
                text=text_part,
                formatting=item.formatting,
                hyperlink=item.hyperlink,
            )

            if isinstance(item, CodeItem):
                language_token = DocumentToken.get_code_language_token(
                    code_language=item.code_language,
                )
                text_part = f"{language_token}{text_part}"
            else:
                text_part = text_part.strip()
                if isinstance(item, ListItem):
                    wrap_tag = None  # deferring list item tags to list handling

            if text_part:
                parts.append(text_part)

        if params.add_caption and isinstance(item, FloatingItem):
            cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
            if cap_text:
                parts.append(cap_text)

        text_res = "".join(parts)
        if wrap_tag is not None:
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        return create_ser_result(text=text_res, span_source=item)


class DocTagsTableSerializer(BaseTableSerializer):
    """DocTags-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)

        res_parts: list[SerializationResult] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.add_location:
                loc_text = item.get_location_tokens(
                    doc=doc,
                    xsize=params.xsize,
                    ysize=params.ysize,
                )
                res_parts.append(create_ser_result(text=loc_text, span_source=item))

            otsl_text = item.export_to_otsl(
                doc=doc,
                add_cell_location=params.add_table_cell_location,
                add_cell_text=params.add_table_cell_text,
                xsize=params.xsize,
                ysize=params.ysize,
            )
            res_parts.append(create_ser_result(text=otsl_text, span_source=item))

        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        text_res = "".join([r.text for r in res_parts])
        if text_res:
            text_res = _wrap(text=text_res, wrap_tag=DocumentToken.OTSL.value)

        return create_ser_result(text=text_res, span_source=res_parts)


class DocTagsPictureSerializer(BasePictureSerializer):
    """DocTags-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)
        res_parts: list[SerializationResult] = []
        is_chart = False

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""
            if params.add_location:
                body += item.get_location_tokens(
                    doc=doc,
                    xsize=params.xsize,
                    ysize=params.ysize,
                )

            classifications = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureClassificationData)
            ]
            if len(classifications) > 0:
                predicted_class = classifications[0].predicted_classes[0].class_name
                if predicted_class in [
                    PictureClassificationLabel.PIE_CHART,
                    PictureClassificationLabel.BAR_CHART,
                    PictureClassificationLabel.STACKED_BAR_CHART,
                    PictureClassificationLabel.LINE_CHART,
                    PictureClassificationLabel.FLOW_CHART,
                    PictureClassificationLabel.SCATTER_CHART,
                    PictureClassificationLabel.HEATMAP,
                ]:
                    is_chart = True
                body += DocumentToken.get_picture_classification_token(predicted_class)

            smiles_annotations = [
                ann for ann in item.annotations if isinstance(ann, PictureMoleculeData)
            ]
            if len(smiles_annotations) > 0:
                body += _wrap(
                    text=smiles_annotations[0].smi, wrap_tag=DocumentToken.SMILES.value
                )

            tabular_chart_annotations = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureTabularChartData)
            ]
            if len(tabular_chart_annotations) > 0:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(
                    data=tabular_chart_annotations[0].chart_data
                )
                otsl_content = temp_table.export_to_otsl(
                    temp_doc, add_cell_location=False
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


class DocTagsKeyValueSerializer(BaseKeyValueSerializer):
    """DocTags-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)
        body = ""
        results: list[SerializationResult] = []

        page_no = 1
        if len(item.prov) > 0:
            page_no = item.prov[0].page_no

        if params.add_location:
            body += item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
            )

        # mapping from source_cell_id to a list of target_cell_ids
        source_to_targets: Dict[int, List[int]] = {}
        for link in item.graph.links:
            source_to_targets.setdefault(link.source_cell_id, []).append(
                link.target_cell_id
            )

        for cell in item.graph.cells:
            cell_txt = ""
            if cell.prov is not None:
                if len(doc.pages.keys()):
                    page_w, page_h = doc.pages[page_no].size.as_tuple()
                    cell_txt += DocumentToken.get_location(
                        bbox=cell.prov.bbox.to_top_left_origin(page_h).as_tuple(),
                        page_w=page_w,
                        page_h=page_h,
                        xsize=params.xsize,
                        ysize=params.ysize,
                    )
            if params.add_content:
                cell_txt += cell.text.strip()

            if cell.cell_id in source_to_targets:
                targets = source_to_targets[cell.cell_id]
                for target in targets:
                    # TODO centralize token creation
                    cell_txt += f"<link_{target}>"

            # TODO centralize token creation
            tok = f"{cell.label.value}_{cell.cell_id}"
            cell_txt = _wrap(text=cell_txt, wrap_tag=tok)
            body += cell_txt
        results.append(create_ser_result(text=body, span_source=item))

        if params.add_caption:
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                results.append(cap_res)

        body = "".join([r.text for r in results])
        body = _wrap(body, DocumentToken.KEY_VALUE_REGION.value)
        return create_ser_result(text=body, span_source=results)


class DocTagsFormSerializer(BaseFormSerializer):
    """DocTags-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        return create_ser_result()


class DocTagsListSerializer(BaseModel, BaseListSerializer):
    """DocTags-specific list serializer."""

    indent: int = 4

    @override
    def serialize(
        self,
        *,
        item: Union[UnorderedList, OrderedList],
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        params = DocTagsParams(**kwargs)
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        delim = _get_delim(params=params)
        if parts:
            text_res = delim.join(
                [
                    t
                    for p in parts
                    if (t := _wrap(text=p.text, wrap_tag=DocumentToken.LIST_ITEM.value))
                ]
            )
            text_res = f"{text_res}{delim}"
            wrap_tag = (
                DocumentToken.ORDERED_LIST.value
                if isinstance(item, OrderedList)
                else DocumentToken.UNORDERED_LIST.value
            )
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        else:
            text_res = ""
        return create_ser_result(text=text_res, span_source=parts)


class DocTagsInlineSerializer(BaseInlineSerializer):
    """DocTags-specific inline group serializer."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        params = DocTagsParams(**kwargs)
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )
        wrap_tag = DocumentToken.INLINE.value
        delim = _get_delim(params=params)
        text_res = delim.join([p.text for p in parts if p.text])
        if text_res:
            text_res = f"{text_res}{delim}"
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        return create_ser_result(text=text_res, span_source=parts)


class DocTagsFallbackSerializer(BaseFallbackSerializer):
    """DocTags-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        return create_ser_result()


class DocTagsDocSerializer(DocSerializer):
    """DocTags-specific document serializer."""

    text_serializer: BaseTextSerializer = DocTagsTextSerializer()
    table_serializer: BaseTableSerializer = DocTagsTableSerializer()
    picture_serializer: BasePictureSerializer = DocTagsPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = DocTagsKeyValueSerializer()
    form_serializer: BaseFormSerializer = DocTagsFormSerializer()
    fallback_serializer: BaseFallbackSerializer = DocTagsFallbackSerializer()

    list_serializer: BaseListSerializer = DocTagsListSerializer()
    inline_serializer: BaseInlineSerializer = DocTagsInlineSerializer()

    params: DocTagsParams = DocTagsParams()

    @override
    def serialize_doc(
        self, *, parts: list[SerializationResult], **kwargs
    ) -> SerializationResult:
        """Serialize a document out of its pages."""
        delim = _get_delim(params=self.params)
        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            page_sep = f"<{DocumentToken.PAGE_BREAK.value}>"
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        wrap_tag = DocumentToken.DOCUMENT.value
        text_res = f"<{wrap_tag}>{text_res}{delim}</{wrap_tag}>"
        return create_ser_result(text=text_res, span_source=parts)

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        params = DocTagsParams(**kwargs)
        results: list[SerializationResult] = []
        if item.captions:
            cap_res = super().serialize_captions(item, **kwargs)
            if cap_res.text:
                if params.add_location:
                    for caption in item.captions:
                        if caption.cref not in self.get_excluded_refs(**kwargs):
                            if isinstance(cap := caption.resolve(self.doc), DocItem):
                                loc_txt = cap.get_location_tokens(
                                    doc=self.doc,
                                    xsize=params.xsize,
                                    ysize=params.ysize,
                                )
                                results.append(create_ser_result(text=loc_txt))
                results.append(cap_res)
        text_res = "".join([r.text for r in results])
        if text_res:
            text_res = _wrap(text=text_res, wrap_tag=DocumentToken.CAPTION.value)
        return create_ser_result(text=text_res, span_source=results)

    @override
    def requires_page_break(self):
        """Whether to add page breaks."""
        return self.params.add_page_break
