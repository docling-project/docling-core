"""Define classes for Doctags serialization."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
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
    Span,
)
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    _iterate_items,
    _should_use_legacy_annotations,
    create_ser_result,
)
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    FloatingItem,
    FormItem,
    FormulaItem,
    GroupItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    ListItem,
    NodeItem,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
    PictureTabularChartData,
    ProvenanceItem,
    SectionHeaderItem,
    TableData,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel, PictureClassificationLabel
from docling_core.types.doc.tokens import DocumentToken, TableToken


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

    do_self_closing: bool = False

    # Task-specific parameters for filtering content
    include_ocr: bool = True
    include_layout: bool = True
    include_otsl: bool = True
    include_code: bool = True
    include_picture: bool = True
    include_chart: bool = True
    include_formula: bool = True

    # Layout mode: when True, only include structure with locations, no content
    layout_mode_only: bool = False


def _get_delim(params: DocTagsParams) -> str:
    if params.mode == DocTagsParams.Mode.HUMAN_FRIENDLY:
        delim = "\n"
    elif params.mode == DocTagsParams.Mode.MINIFIED:
        delim = ""
    else:
        raise RuntimeError(f"Unknown DocTags mode: {params.mode}")
    return delim


def create_task_filtered_params(task_list: list[str], **kwargs) -> DocTagsParams:
    """Create DocTagsParams with task-specific filtering enabled.

    Args:
        task_list: List of tasks to include (e.g., ['ocr', 'layout', 'otsl'])
        **kwargs: Additional parameters to override defaults

    Returns:
        DocTagsParams configured for the specified tasks
    """
    # Default: include all tasks
    params = {
        "include_ocr": True,
        "include_layout": True,
        "include_otsl": True,
        "include_code": True,
        "include_picture": True,
        "include_chart": True,
        "include_formula": True,
        "add_location": True,
        "add_content": True,
        "add_caption": True,
    }

    # Override based on task list
    if task_list:
        params["include_ocr"] = "ocr" in task_list
        params["include_layout"] = "layout" in task_list
        params["include_otsl"] = "otsl" in task_list
        params["include_code"] = "code" in task_list
        params["include_picture"] = "picture" in task_list
        params["include_chart"] = "chart" in task_list
        params["include_formula"] = "formula" in task_list

        # Special handling for layout mode
        if "layout" in task_list:
            # When layout is present, always include locations
            params["add_location"] = True
            # For layout mode, we want to show structure but strip content
            # The individual serializers will handle content stripping based on task parameters
            params["add_content"] = True  # Let serializers handle content filtering
        else:
            # When layout is NOT present, only include locations for specific tasks
            params["add_location"] = False
            # Only include content for explicitly requested tasks
            # But don't disable add_content globally - let individual serializers handle it
            if not params["include_otsl"]:
                params["add_table_cell_text"] = False

    # Apply any additional kwargs
    params.update(kwargs)

    return DocTagsParams(**params)


class DocTagsTextSerializer(BaseModel, BaseTextSerializer):
    """DocTags-specific text item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        params = DocTagsParams(**kwargs)
        wrap_tag: Optional[str] = DocumentToken.create_token_name_from_doc_item_label(
            label=item.label,
            **({"level": item.level} if isinstance(item, SectionHeaderItem) else {}),
        )
        parts: list[str] = []

        if item.meta:
            meta_res = doc_serializer.serialize_meta(item=item, **kwargs)
            if meta_res.text:
                parts.append(meta_res.text)

        # Check if this item type should be included based on task parameters
        should_include_content = True
        if isinstance(item, CodeItem) and not params.include_code:
            should_include_content = False
        elif isinstance(item, FormulaItem) and not params.include_formula:
            should_include_content = False
        elif (
            isinstance(item, (TextItem, SectionHeaderItem, ListItem))
            and not params.include_ocr
        ):
            should_include_content = False

        # In layout mode, skip content for items whose specific flags are disabled
        if params.include_layout:
            if isinstance(item, CodeItem) and not params.include_code:
                should_include_content = False
            elif isinstance(item, FormulaItem) and not params.include_formula:
                should_include_content = False

        # For code items, if include_code is False, return empty result
        # Exception: in layout mode, always include items (with location but no content)
        if not params.include_layout:
            if isinstance(item, CodeItem) and not params.include_code:
                return create_ser_result()

            # For formula items, if include_formula is False, return empty result
            if isinstance(item, FormulaItem) and not params.include_formula:
                return create_ser_result()

        if params.add_location and params.include_layout:
            location = item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
                self_closing=params.do_self_closing,
            )
            if location:
                parts.append(location)

        if params.add_content and should_include_content:
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

            if isinstance(item, CodeItem):
                language_token = DocumentToken.get_code_language_token(
                    code_language=item.code_language,
                    self_closing=params.do_self_closing,
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

    def _get_table_token(self) -> Any:
        return TableToken

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
        params = DocTagsParams(**kwargs)

        res_parts: list[SerializationResult] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.add_location and params.include_layout:
                loc_text = item.get_location_tokens(
                    doc=doc,
                    xsize=params.xsize,
                    ysize=params.ysize,
                    self_closing=params.do_self_closing,
                )
                res_parts.append(create_ser_result(text=loc_text, span_source=item))

            # Check if OTSL content should be included
            if params.include_otsl:
                otsl_text = item.export_to_otsl(
                    doc=doc,
                    add_cell_location=params.add_table_cell_location,
                    add_cell_text=params.add_table_cell_text,
                    xsize=params.xsize,
                    ysize=params.ysize,
                    visited=visited,
                    table_token=self._get_table_token(),
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

    def _get_predicted_class(
        self, item: PictureItem, params: DocTagsParams
    ) -> Optional[str]:
        """Get the predicted class from item metadata or annotations."""
        if item.meta:
            if item.meta.classification:
                return item.meta.classification.get_main_prediction().class_name
        elif _should_use_legacy_annotations(
            params=params,
            item=item,
            kind=PictureClassificationData.model_fields["kind"].default,
        ):
            if classifications := [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureClassificationData)
            ]:
                if classifications[0].predicted_classes:
                    return classifications[0].predicted_classes[0].class_name
        return None

    def _is_chart_type(self, predicted_class: Optional[str]) -> bool:
        """Check if predicted class indicates a chart."""
        if not predicted_class:
            return False
        return predicted_class in [
            PictureClassificationLabel.PIE_CHART,
            PictureClassificationLabel.BAR_CHART,
            PictureClassificationLabel.STACKED_BAR_CHART,
            PictureClassificationLabel.LINE_CHART,
            PictureClassificationLabel.FLOW_CHART,
            PictureClassificationLabel.SCATTER_CHART,
            PictureClassificationLabel.HEATMAP,
        ]

    def _get_molecule_smi(
        self, item: PictureItem, params: DocTagsParams
    ) -> Optional[str]:
        """Get SMILES string from item metadata or annotations."""
        if item.meta:
            if item.meta.molecule:
                return item.meta.molecule.smi
        elif _should_use_legacy_annotations(
            params=params,
            item=item,
            kind=PictureMoleculeData.model_fields["kind"].default,
        ):
            if smiles_annotations := [
                ann for ann in item.annotations if isinstance(ann, PictureMoleculeData)
            ]:
                return smiles_annotations[0].smi
        return None

    def _get_tabular_chart_data(
        self, item: PictureItem, params: DocTagsParams
    ) -> Optional[TableData]:
        """Get tabular chart data from item metadata or annotations."""
        if item.meta:
            if item.meta.tabular_chart:
                return item.meta.tabular_chart.chart_data
        elif _should_use_legacy_annotations(
            params=params,
            item=item,
            kind=PictureTabularChartData.model_fields["kind"].default,
        ):
            if tabular_chart_annotations := [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureTabularChartData)
            ]:
                return tabular_chart_annotations[0].chart_data
        return None

    def _build_body_content(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        params: DocTagsParams,
        predicted_class: Optional[str],
        is_chart: bool,
    ) -> str:
        """Build the body content for the picture item."""
        body = ""
        if params.add_location and params.include_layout:
            body += item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
                self_closing=params.do_self_closing,
            )

        should_include_content = True
        if params.include_layout:
            if is_chart and not params.include_chart:
                should_include_content = False
            elif not is_chart and not params.include_picture:
                should_include_content = False

        if should_include_content and predicted_class:
            body += DocumentToken.get_picture_classification_token(predicted_class)

        if should_include_content:
            smi = self._get_molecule_smi(item, params)
            if smi:
                body += _wrap(text=smi, wrap_tag=DocumentToken.SMILES.value)

            chart_data = self._get_tabular_chart_data(item, params)
            if chart_data and chart_data.table_cells:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(data=chart_data)
                otsl_content = temp_table.export_to_otsl(
                    temp_doc, add_cell_location=False
                )
                body += otsl_content
        return body

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

        predicted_class = self._get_predicted_class(item, params)
        is_chart = self._is_chart_type(predicted_class)

        if not params.include_layout:
            if is_chart and not params.include_chart:
                return create_ser_result()
            elif not is_chart and not params.include_picture:
                return create_ser_result()

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = self._build_body_content(
                item, doc, params, predicted_class, is_chart
            )
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
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)
        body = ""
        results: list[SerializationResult] = []

        page_no = 1
        if len(item.prov) > 0:
            page_no = item.prov[0].page_no

        if params.add_location and params.include_layout:
            body += item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
                self_closing=params.do_self_closing,
            )

        # mapping from source_cell_id to a list of target_cell_ids
        source_to_targets: Dict[int, List[int]] = {}
        for link in item.graph.links:
            source_to_targets.setdefault(link.source_cell_id, []).append(
                link.target_cell_id
            )

        for cell in item.graph.cells:
            cell_txt = ""
            if cell.prov is not None and params.add_location and params.include_layout:
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
        **kwargs: Any,
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
        item: ListGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
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
                if item.first_item_is_enumerated(doc)
                else DocumentToken.UNORDERED_LIST.value
            )
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        else:
            text_res = ""
        return create_ser_result(text=text_res, span_source=parts)


class DocTagsInlineSerializer(BaseInlineSerializer):
    """DocTags-specific inline group serializer."""

    def _get_inline_location_tags(
        self, doc: DoclingDocument, item: InlineGroup, params: DocTagsParams
    ) -> SerializationResult:

        prov: Optional[ProvenanceItem] = None
        boxes: list[BoundingBox] = []
        doc_items: list[DocItem] = []
        for it, _ in doc.iterate_items(root=item):
            if isinstance(it, DocItem):
                for prov in it.prov:
                    boxes.append(prov.bbox)
                    doc_items.append(it)
        if prov is None:
            return create_ser_result()

        bbox = BoundingBox.enclosing_bbox(boxes=boxes)

        # using last seen prov as reference for page dims
        page_w, page_h = doc.pages[prov.page_no].size.as_tuple()

        loc_str = DocumentToken.get_location(
            bbox=bbox.to_top_left_origin(page_h).as_tuple(),
            page_w=page_w,
            page_h=page_h,
            xsize=params.xsize,
            ysize=params.ysize,
            self_closing=params.do_self_closing,
        )

        return SerializationResult(
            text=loc_str,
            spans=[Span(item=it) for it in doc_items],
        )

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        params = DocTagsParams(**kwargs)
        parts: List[SerializationResult] = []
        if params.add_location and params.include_layout:
            inline_loc_tags_ser_res = self._get_inline_location_tags(
                doc=doc,
                item=item,
                params=params,
            )
            parts.append(inline_loc_tags_ser_res)
            params.add_location = False  # suppress children location serialization
        parts.extend(
            doc_serializer.get_parts(
                item=item,
                list_level=list_level,
                is_inline_scope=True,
                visited=my_visited,
                **{**kwargs, **params.model_dump()},
            )
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
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        if isinstance(item, GroupItem):
            parts = doc_serializer.get_parts(item=item, **kwargs)
            text_res = "\n".join([p.text for p in parts if p.text])
            return create_ser_result(text=text_res, span_source=parts)
        else:
            return create_ser_result()


class DocTagsAnnotationSerializer(BaseAnnotationSerializer):
    """DocTags-specific annotation serializer."""

    @override
    def serialize(self, *, item: DocItem, **kwargs: Any) -> SerializationResult:
        """Serializes the item's annotations."""
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

    annotation_serializer: BaseAnnotationSerializer = DocTagsAnnotationSerializer()

    params: DocTagsParams = DocTagsParams()

    @override
    def get_parts(
        self,
        item: Optional[NodeItem] = None,
        *,
        traverse_pictures: bool = False,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Get the components to be combined for serializing this node with task filtering."""
        parts: list[SerializationResult] = []
        my_visited: set[str] = visited if visited is not None else set()
        params = self.params.merge_with_patch(patch=kwargs)
        for node, lvl in _iterate_items(
            doc=self.doc,
            layers=params.layers,
            node=item,
            traverse_pictures=traverse_pictures,
            add_page_breaks=self.requires_page_break(),
        ):
            if node.self_ref in my_visited:
                continue
            else:
                my_visited.add(node.self_ref)

            # Task-based filtering: only process items that match the requested tasks
            should_process = self._should_process_item(node, params)
            if not should_process:
                continue

            part = self.serialize(
                item=node,
                list_level=list_level,
                is_inline_scope=is_inline_scope,
                visited=my_visited,
                **(dict(level=lvl) | kwargs),
            )
            if part.text:
                parts.append(part)
        return parts

    def _should_process_item(self, node: NodeItem, params: DocTagsParams) -> bool:
        """Determine if an item should be processed based on task parameters."""
        if not isinstance(node, DocItem):
            return True  # Process non-DocItem nodes (groups, etc.)

        # For layout mode, include all elements (they'll be processed with locations but no content)
        if params.include_layout:
            return True

        # For non-layout mode, only include elements for explicitly requested tasks
        # Tables: allow through if layout is enabled (for locations/captions) or if OTSL is enabled
        if isinstance(node, TableItem):
            if params.include_layout or params.include_otsl:
                return True
            return False
        elif isinstance(node, PictureItem) and not params.include_picture:
            return False
        elif isinstance(node, CodeItem) and not params.include_code:
            return False
        elif isinstance(node, FormulaItem) and not params.include_formula:
            return False
        elif (
            isinstance(node, (TextItem, SectionHeaderItem, ListItem))
            and not params.include_ocr
        ):
            return False
        elif isinstance(node, FormItem) and not params.include_ocr:
            return False

        return True

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        params = DocTagsParams(**kwargs)
        results: list[SerializationResult] = []
        if item.captions:
            # Always include caption structure when layout is present
            if params.include_layout:
                # For layout mode, include captions with locations but without content
                if params.add_location and params.include_layout:
                    for caption in item.captions:
                        if caption.cref not in self.get_excluded_refs(**kwargs):
                            if isinstance(cap := caption.resolve(self.doc), DocItem):
                                loc_txt = cap.get_location_tokens(
                                    doc=self.doc,
                                    xsize=params.xsize,
                                    ysize=params.ysize,
                                    self_closing=params.do_self_closing,
                                )
                                results.append(create_ser_result(text=loc_txt))
                # Don't include caption content when only layout is requested
                if not params.include_ocr:
                    pass  # Skip content, only include locations
                else:
                    # Include content when OCR is also requested
                    cap_res = super().serialize_captions(item, **kwargs)
                    if cap_res.text:
                        results.append(cap_res)
            else:
                # For non-layout mode, only include captions if OCR is requested
                if params.include_ocr:
                    cap_res = super().serialize_captions(item, **kwargs)
                    if cap_res.text:
                        if params.add_location and params.include_layout:
                            for caption in item.captions:
                                if caption.cref not in self.get_excluded_refs(**kwargs):
                                    if isinstance(
                                        cap := caption.resolve(self.doc), DocItem
                                    ):
                                        loc_txt = cap.get_location_tokens(
                                            doc=self.doc,
                                            xsize=params.xsize,
                                            ysize=params.ysize,
                                            self_closing=params.do_self_closing,
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
