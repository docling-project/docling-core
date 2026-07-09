"""Models for the Docling Document data type."""

import base64
import copy
import hashlib
import json
import logging
import mimetypes
import re
import sys
import tempfile
import typing
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from io import BytesIO, StringIO
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    Optional,
    Union,
)
from urllib.parse import unquote

import numpy as np
import pandas as pd
import yaml
from PIL import Image as PILImage
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    SerializerFunctionWrapHandler,
    StringConstraints,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
    validate_call,
)
from tabulate import _column_type, tabulate
from typing_extensions import Self, deprecated, override

from docling_core.types.base import _JSON_POINTER_REGEX, VERSION_PATTERN, UniqueList
from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.base import (
    CoordOrigin,
    ImageRefMode,
    PydanticSerCtxKey,
    round_pydantic_float,
)

# --- Re-exports from split modules (kept for backward compatibility) ---
from docling_core.types.doc.common.annotations import (
    BaseAnnotation,
    DescriptionAnnotation,
    MiscAnnotation,
)
from docling_core.types.doc.common.constants import (
    CURRENT_VERSION,
    DEFAULT_EXPORT_LABELS,
    DOCUMENT_TOKENS_EXPORT_LABELS,
)
from docling_core.types.doc.common.content_layer import DEFAULT_CONTENT_LAYERS, ContentLayer
from docling_core.types.doc.common.formatting import Formatting, Script
from docling_core.types.doc.common.meta import (
    BaseMeta,
    BasePrediction,
    CodeMetaField,
    DescriptionMetaField,
    EntitiesMetaField,
    EntityMention,
    FloatingMeta,
    KeywordsMetaField,
    LanguageMetaField,
    MetaFieldName,
    MetaUtils,
    SummaryMetaField,
    TopicsMetaField,
    _ExtraAllowingModel,
)
from docling_core.types.doc.common.origin import DocumentOrigin
from docling_core.types.doc.common.reference import (
    CV2_INSTALLED,
    FineRef,
    ImageRef,
    PageItem,
    ProvenanceItem,
    RefItem,
)
from docling_core.types.doc.common.scalars import CharSpan, LevelNumber, Uint64
from docling_core.types.doc.common.source import BaseSource, SourceType, TrackSource
from docling_core.types.doc.doctags import DocTagsDocument, DocTagsPage
from docling_core.types.doc.items.code import CodeItem
from docling_core.types.doc.items.content import ContentItem
from docling_core.types.doc.items.form import FieldHeadingItem, FieldItem, FieldRegionItem, FieldValueItem
from docling_core.types.doc.items.group import GroupItem, InlineGroup, ListGroup, OrderedList, UnorderedList
from docling_core.types.doc.items.key_value import FormItem, GraphCell, GraphData, GraphLink, KeyValueItem
from docling_core.types.doc.items.node import DocItem, FloatingItem, NodeItem
from docling_core.types.doc.items.picture.charts import (
    ChartBar,
    ChartLine,
    ChartPoint,
    ChartSlice,
    ChartStackedBar,
    PictureBarChartData,
    PictureChartData,
    PictureLineChartData,
    PicturePieChartData,
    PictureScatterChartData,
    PictureStackedBarChartData,
    PictureTabularChartData,
)
from docling_core.types.doc.items.picture.classification import PictureClassificationClass, PictureClassificationData
from docling_core.types.doc.items.picture.meta import (
    MoleculeMetaField,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    TabularChartMetaField,
)
from docling_core.types.doc.items.picture.molecule import PictureMoleculeData
from docling_core.types.doc.items.picture.picture import (
    BasePictureData,
    PictureDataType,
    PictureDescriptionData,
    PictureItem,
    PictureMiscData,
)
from docling_core.types.doc.items.table.table import TableAnnotationType, TableItem
from docling_core.types.doc.items.table.table_data import (
    AnyTableCell,
    Orientation,
    RichTableCell,
    TableCell,
    TableData,
)
from docling_core.types.doc.items.text import FormulaItem, ListItem, SectionHeaderItem, TextItem, TitleItem
from docling_core.types.doc.labels import (
    CodeLanguageLabel,
    DocItemLabel,
    GraphCellLabel,
    GraphLinkLabel,
    GroupLabel,
    HumanLanguageLabel,
    PictureClassificationLabel,
)
from docling_core.types.doc.tokens import DocumentToken, TableToken
from docling_core.types.doc.utils import (
    is_remote_path,
    parse_otsl_table_content,
    relative_path,
    resolve_archive_path,
    safe_extract_zip_archive,
    validate_archive_relative_path,
)
from docling_core.utils.settings import settings

# --- end re-exports ---


_logger = logging.getLogger(__name__)


class DoclingDocument(BaseModel):
    """DoclingDocument."""

    schema_name: typing.Literal["DoclingDocument"] = "DoclingDocument"
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = CURRENT_VERSION
    name: str  # The working name of this document, without extensions
    # (could be taken from originating doc, or just "Untitled 1")
    origin: Optional[DocumentOrigin] = (
        None  # DoclingDocuments may specify an origin (converted to DoclingDocument).
        # This is optional, e.g. a DoclingDocument could also be entirely
        # generated from synthetic data.
    )
    furniture: Annotated[GroupItem, Field(deprecated=True)] = GroupItem(
        name="_root_",
        self_ref="#/furniture",
        content_layer=ContentLayer.FURNITURE,
    )  # List[RefItem] = []
    body: GroupItem = GroupItem(name="_root_", self_ref="#/body")  # List[RefItem] = []

    groups: list[Union[ListGroup, InlineGroup, GroupItem]] = []
    texts: list[
        Union[
            TitleItem,
            SectionHeaderItem,
            ListItem,
            CodeItem,
            FormulaItem,
            FieldHeadingItem,
            FieldValueItem,
            TextItem,
        ]
    ] = []
    pictures: list[PictureItem] = []
    tables: list[TableItem] = []
    key_value_items: list[KeyValueItem] = []
    form_items: list[FormItem] = []
    field_regions: list[FieldRegionItem] = []
    field_items: list[FieldItem] = []

    pages: dict[int, PageItem] = {}  # empty as default

    @model_serializer(mode="wrap")
    def _custom_pydantic_serialize(self, handler: SerializerFunctionWrapHandler) -> dict:
        dumped = handler(self)

        # suppress serializing certain fields when empty:
        for field in {"field_regions", "field_items"}:
            if dumped.get(field) == []:
                del dumped[field]

        return dumped

    @staticmethod
    def _clamp_location_coordinate(
        *,
        value: float,
        lo: float,
        hi: float,
        tolerance: float,
        page_no: int,
        bbox_label: str,
        coord_name: str,
    ) -> float:
        clamped = min(max(value, lo), hi)
        if clamped != value and abs(value - clamped) > tolerance:
            if value < lo:
                warnings.warn(
                    f"{bbox_label} coordinate {coord_name} on page {page_no} is outside page bounds: "
                    f"{value=} < {lo=}; clamping to {lo}",
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"{bbox_label} coordinate {coord_name} on page {page_no} is outside page bounds: "
                    f"{value=} > {hi=}; clamping to {hi}",
                    stacklevel=3,
                )
        return clamped

    @classmethod
    def _clamp_bbox_to_page(
        cls,
        *,
        bbox: BoundingBox,
        page_size: Size,
        page_no: int,
        bbox_label: str,
    ) -> BoundingBox:
        page_width = page_size.width
        page_height = page_size.height

        tolerance_factor = 1e-2

        return bbox.model_copy(
            update={
                "l": cls._clamp_location_coordinate(
                    value=bbox.l,
                    lo=0.0,
                    hi=page_width,
                    tolerance=page_width * tolerance_factor,
                    page_no=page_no,
                    bbox_label=bbox_label,
                    coord_name="l",
                ),
                "r": cls._clamp_location_coordinate(
                    value=bbox.r,
                    lo=0.0,
                    hi=page_width,
                    tolerance=page_width * tolerance_factor,
                    page_no=page_no,
                    bbox_label=bbox_label,
                    coord_name="r",
                ),
                "t": cls._clamp_location_coordinate(
                    value=bbox.t,
                    lo=0.0,
                    hi=page_height,
                    tolerance=page_height * tolerance_factor,
                    page_no=page_no,
                    bbox_label=bbox_label,
                    coord_name="t",
                ),
                "b": cls._clamp_location_coordinate(
                    value=bbox.b,
                    lo=0.0,
                    hi=page_height,
                    tolerance=page_height * tolerance_factor,
                    page_no=page_no,
                    bbox_label=bbox_label,
                    coord_name="b",
                ),
            }
        )

    @classmethod
    def _clamp_provenance_bbox_to_page(cls, *, prov: ProvenanceItem, page_size: Size) -> None:
        prov.bbox = cls._clamp_bbox_to_page(
            bbox=prov.bbox,
            page_size=page_size,
            page_no=prov.page_no,
            bbox_label="Provenance bbox",
        )

    def _clamp_provenance_bboxes_to_pages(self) -> None:
        def clamp_prov(prov: ProvenanceItem) -> None:
            page = self.pages.get(prov.page_no)
            if page is not None:
                self._clamp_provenance_bbox_to_page(prov=prov, page_size=page.size)

        def clamp_table_cell_bboxes(table: TableItem) -> None:
            page_nos = {prov.page_no for prov in table.prov}
            if len(page_nos) != 1:
                return
            page_no = next(iter(page_nos))
            page = self.pages.get(page_no)
            if page is None:
                return

            for cell in table.data.table_cells:
                if cell.bbox is not None:
                    cell.bbox = self._clamp_bbox_to_page(
                        bbox=cell.bbox,
                        page_size=page.size,
                        page_no=page_no,
                        bbox_label="Table cell bbox",
                    )

        item_lists: tuple[Iterable[NodeItem], ...] = (
            self.texts,
            self.pictures,
            self.tables,
            self.key_value_items,
            self.form_items,
            self.field_regions,
            self.field_items,
        )
        for item_list in item_lists:
            for item in item_list:
                if isinstance(item, DocItem):
                    for prov in item.prov:
                        clamp_prov(prov)
                if isinstance(item, TableItem):
                    clamp_table_cell_bboxes(item)
                if isinstance(item, KeyValueItem | FormItem):
                    for cell in item.graph.cells:
                        if cell.prov is not None:
                            clamp_prov(cell.prov)

    @model_validator(mode="before")
    @classmethod
    def transform_to_content_layer(cls, data: Any) -> Any:
        """transform_to_content_layer."""
        # Since version 1.1.0, all NodeItems carry content_layer property.
        # We must assign previous page_header and page_footer instances to furniture.
        # Note: model_validators which check on the version must use "before".
        if isinstance(data, dict) and data.get("version", "") == "1.0.0":
            for item in data.get("texts", []):
                if "label" in item and item["label"] in [
                    DocItemLabel.PAGE_HEADER.value,
                    DocItemLabel.PAGE_FOOTER.value,
                ]:
                    item["content_layer"] = "furniture"
        return data

    def _create_pos_by_cell_id(self, graph: GraphData) -> dict[int, int]:
        return {cell.cell_id: idx for idx, cell in enumerate(graph.cells)}

    def _migrate_to_field_regions(self) -> None:
        has_single_kv_item = len(self.key_value_items) == 1
        last_item_is_kv_item = False
        found_last_kv_item = False
        if has_single_kv_item:
            for item, _ in self.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            ):
                if found_last_kv_item:
                    last_item_is_kv_item = False
                    break
                elif item == self.key_value_items[0]:
                    found_last_kv_item = True
                    last_item_is_kv_item = True

        is_annot_case = has_single_kv_item and last_item_is_kv_item
        if is_annot_case:
            self._migrate_annot_forms_to_field_regions(self.key_value_items[0])
            self._post_migration_cleanup()
        else:
            to_delete: list[NodeItem] = []

            for item, _ in self.iterate_items():
                if isinstance(item, FormItem | KeyValueItem):
                    pos_by_cell_id = self._create_pos_by_cell_id(item.graph)
                    to_delete.append(item)

                    visited: set[str] = set()
                    outgoing_links: dict[int, list[int]] = {}

                    kvm = FieldRegionItem(
                        self_ref="#",
                        prov=item.prov,
                        content_layer=item.content_layer,
                        meta=item.meta,
                        comments=item.comments,
                        source=item.source,
                    )
                    self.insert_item_after_sibling(new_item=kvm, sibling=item)

                    for link in item.graph.links:
                        if link.label == GraphLinkLabel.TO_VALUE:
                            key_cell = item.graph.cells[pos_by_cell_id[link.source_cell_id]]
                            value_cell = item.graph.cells[pos_by_cell_id[link.target_cell_id]]
                        elif link.label == GraphLinkLabel.TO_KEY:
                            value_cell = item.graph.cells[pos_by_cell_id[link.source_cell_id]]
                            key_cell = item.graph.cells[pos_by_cell_id[link.target_cell_id]]

                        # check if we have already seen this key-value pair
                        if (key_val_id := f"{key_cell.cell_id}-{value_cell.cell_id}") not in visited:
                            visited.add(key_val_id)
                            outgoing_links.setdefault(key_cell.cell_id, []).append(value_cell.cell_id)

                    for key_cell_id, value_cell_ids in outgoing_links.items():
                        kve = self.add_field_item(parent=kvm)
                        key_cell = item.graph.cells[pos_by_cell_id[key_cell_id]]
                        self.add_field_key(
                            text=key_cell.text,
                            parent=kve,
                            prov=key_cell.prov,
                        )
                        for value_cell_id in value_cell_ids:
                            value_cell = item.graph.cells[pos_by_cell_id[value_cell_id]]
                            self.add_field_value(
                                text=value_cell.text,
                                parent=kve,
                                prov=value_cell.prov,
                            )

            self.delete_items(node_items=to_delete)

        self._normalize_references()

    class _KVMigrData(BaseModel):
        value_crefs: list[str] = []
        key_cell: GraphCell = GraphCell(label=GraphCellLabel.KEY, cell_id=0, text="", orig="")
        value_cells: list[GraphCell] = []

    def _serialize_prov(self, prov: ProvenanceItem) -> str:
        return f"{prov.page_no},{prov.bbox.l},{prov.bbox.t},{prov.bbox.r},{prov.bbox.b},{prov.bbox.coord_origin.value}"

    def _provs_match(self, prov1: ProvenanceItem, prov2: ProvenanceItem, iou_threshold: float = 0.01) -> bool:
        if prov1.page_no != prov2.page_no or prov1.bbox.coord_origin != prov2.bbox.coord_origin:
            return False  # TODO: can normalize and compare but needs page size
        return prov1.bbox.intersection_over_union(other=prov2.bbox, eps=0.0) > iou_threshold

    def _build_prov_index(self, kvi: KeyValueItem, pos_by_cell_id: dict[int, int]) -> dict[str, Optional[NodeItem]]:
        visited: set[str] = set()
        prov_index: dict[str, NodeItem | None] = {}
        text_index: dict[str, str | None] = {}
        for link in kvi.graph.links:
            if link.label not in {GraphLinkLabel.TO_VALUE, GraphLinkLabel.TO_KEY}:
                continue
            key_cell = (
                kvi.graph.cells[pos_by_cell_id[link.source_cell_id]]
                if link.label == GraphLinkLabel.TO_VALUE
                else kvi.graph.cells[pos_by_cell_id[link.target_cell_id]]
            )
            value_cell = (
                kvi.graph.cells[pos_by_cell_id[link.target_cell_id]]
                if link.label == GraphLinkLabel.TO_VALUE
                else kvi.graph.cells[pos_by_cell_id[link.source_cell_id]]
            )
            # check if we have already seen this key-value pair
            if (
                key_cell.prov
                and value_cell.prov
                and (key_val_id := f"{key_cell.cell_id}-{value_cell.cell_id}") not in visited
            ):
                visited.add(key_val_id)

                for item, _ in self.iterate_items(
                    with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
                ):
                    if isinstance(item, DocItem) and item.prov:
                        if self._provs_match(item.prov[0], key_cell.prov):
                            serialized_prov = self._serialize_prov(key_cell.prov)
                            prov_index[serialized_prov] = item
                            text_index[key_cell.text] = (
                                item.self_ref + "|" + (item.text if isinstance(item, TextItem) else "")
                            )
                        if self._provs_match(item.prov[0], value_cell.prov):
                            serialized_prov = self._serialize_prov(value_cell.prov)
                            prov_index[serialized_prov] = item
                            text_index[value_cell.text] = (
                                item.self_ref + "|" + (item.text if isinstance(item, TextItem) else "")
                            )
        return prov_index

    def _find_node_by_prov(self, prov: ProvenanceItem, prov_index: dict[str, NodeItem | None]) -> NodeItem | None:
        val = prov_index.get(self._serialize_prov(prov))
        if val is not None:
            pass
        return val

    def _build_kv_migration_index(self, kvi: KeyValueItem) -> dict[str, dict[int, "DoclingDocument._KVMigrData"]]:
        outgoing_links: dict[str, dict[int, DoclingDocument._KVMigrData]] = {}
        visited: set[str] = set()
        pos_by_cell_id = self._create_pos_by_cell_id(kvi.graph)
        prov_index = self._build_prov_index(kvi, pos_by_cell_id)

        for link in kvi.graph.links:
            key_cell = (
                kvi.graph.cells[pos_by_cell_id[link.source_cell_id]]
                if link.label == GraphLinkLabel.TO_VALUE
                else kvi.graph.cells[pos_by_cell_id[link.target_cell_id]]
            )
            value_cell = (
                kvi.graph.cells[pos_by_cell_id[link.target_cell_id]]
                if link.label == GraphLinkLabel.TO_VALUE
                else kvi.graph.cells[pos_by_cell_id[link.source_cell_id]]
            )
            # check if we have already seen this key-value pair
            if (key_val_id := f"{key_cell.cell_id}-{value_cell.cell_id}") not in visited:
                key_item_ref = key_cell.item_ref or (
                    node.get_ref()
                    if key_cell.prov and (node := self._find_node_by_prov(key_cell.prov, prov_index))
                    else None
                )
                val_item_ref = value_cell.item_ref or (
                    node.get_ref()
                    if value_cell.prov and (node := self._find_node_by_prov(value_cell.prov, prov_index))
                    else None
                )
                if key_item_ref and val_item_ref:
                    visited.add(key_val_id)

                    migr_data = outgoing_links.setdefault(key_item_ref.cref, {})
                    migr_data.setdefault(key_cell.cell_id, DoclingDocument._KVMigrData())
                    migr_data[key_cell.cell_id].value_crefs.append(val_item_ref.cref)
                    migr_data[key_cell.cell_id].key_cell = key_cell
                    migr_data[key_cell.cell_id].value_cells.append(value_cell)

        return outgoing_links

    def _eq_prov(self, prov1: ProvenanceItem, prov2: ProvenanceItem) -> bool:
        """Compare provenances while tolerating charspan discrepancies."""
        return prov1.bbox == prov2.bbox and prov1.page_no == prov2.page_no

    def _migrate_annot_forms_to_field_regions(self, kvi: KeyValueItem) -> None:
        to_delete: list[NodeItem] = [kvi]
        outgoing_links = self._build_kv_migration_index(kvi)

        for key_cref, key_cell_dict in outgoing_links.items():
            existing_key_item = RefItem(cref=key_cref).resolve(doc=self)
            fri = FieldRegionItem(self_ref="#")

            if ex_key_item_is_li := isinstance(existing_key_item, ListItem):
                self.append_child_item(child=fri, parent=existing_key_item)
            else:
                self._shift_down(old_subroot=existing_key_item, new_subroot=fri)

            for _, migr_data_item in key_cell_dict.items():
                reuse_existing_key_item = False

                cell_and_ex_key_item_provs_equal = (
                    migr_data_item.key_cell.prov
                    and isinstance(existing_key_item, DocItem)
                    and existing_key_item.prov
                    and self._eq_prov(migr_data_item.key_cell.prov, existing_key_item.prov[0])
                )

                if len(key_cell_dict) == 1 and (
                    migr_data_item.key_cell.prov is None or cell_and_ex_key_item_provs_equal
                ):
                    reuse_existing_key_item = True

                key_prov = (
                    migr_data_item.key_cell.prov
                    if migr_data_item.key_cell.prov
                    else (existing_key_item.prov[0] if existing_key_item.prov else None)
                )

                if reuse_existing_key_item:
                    key_item = existing_key_item
                else:  # case where single key_cref maps to multiple key cells
                    if isinstance(existing_key_item, TextItem):
                        existing_key_item.text = ""
                    key_item = self.add_text(  # temporary item
                        label=DocItemLabel.TEXT,
                        text=migr_data_item.key_cell.text,
                        parent=existing_key_item,
                        prov=key_prov,
                    )
                skip_ki_deletion = key_item in to_delete

                fi = self.add_field_item(parent=fri)
                if isinstance(key_item, TextItem):
                    self.add_field_key(text=migr_data_item.key_cell.text or key_item.text, parent=fi, prov=key_prov)
                    if isinstance(key_item, ListItem):
                        skip_ki_deletion = True
                        key_item.text = ""
                        if cell_and_ex_key_item_provs_equal:
                            key_item.prov = []
                elif isinstance(key_item, PictureItem):
                    fk = self.add_field_key(text=migr_data_item.key_cell.text, parent=fi, prov=key_prov)
                    if not key_item.children:
                        self.append_child_item(child=key_item.model_copy(deep=True), parent=fk)
                    else:
                        skip_ki_deletion = True
                else:
                    continue  # TODO: handle other key item types

                for idx, value_cref in enumerate(migr_data_item.value_crefs):
                    value_item: NodeItem = RefItem(cref=value_cref).resolve(doc=self)
                    # giving priority to the provenance from the graph cells
                    value_prov = migr_data_item.value_cells[idx].prov or (
                        value_item.prov[0] if isinstance(value_item, DocItem) and value_item.prov else None
                    )
                    skip_vi_deletion = value_item in to_delete
                    if isinstance(value_item, TextItem):
                        # giving priority to the text from the graph cells
                        value_text = migr_data_item.value_cells[idx].text or value_item.text
                        if value_item.label in {DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED}:
                            if not value_item.children:
                                fv = self.add_field_value(text="", parent=fi)
                                new_text_item = value_item.model_copy(deep=True)
                                new_text_item.prov = [value_prov] if value_prov else []
                                new_text_item.text = value_text
                                self.append_child_item(child=new_text_item, parent=fv)
                            else:
                                skip_vi_deletion = True
                        else:
                            fv = self.add_field_value(text=value_text, parent=fi, prov=value_prov)
                            if value_item.label == DocItemLabel.EMPTY_VALUE:
                                fv.kind = "fillable"

                        if isinstance(value_item, ListItem):
                            skip_vi_deletion = True
                    elif isinstance(value_item, PictureItem):
                        fv = self.add_field_value(text=migr_data_item.value_cells[idx].text, parent=fi, prov=value_prov)
                        if not value_item.children:
                            self.append_child_item(child=value_item.model_copy(deep=True), parent=fv)
                        else:
                            skip_vi_deletion = True
                    else:
                        continue  # TODO: handle other value item types

                    if not skip_vi_deletion:
                        to_delete.append(value_item)
                if not skip_ki_deletion:
                    to_delete.append(key_item)

                if existing_key_item.prov and not cell_and_ex_key_item_provs_equal and not ex_key_item_is_li:
                    fi.prov = existing_key_item.prov

        self.delete_items(node_items=to_delete)

    def _has_field_region_ancestor(self, item: NodeItem) -> bool:
        while item.parent is not None:
            item = item.parent.resolve(doc=self)
            if isinstance(item, FieldRegionItem):
                return True
        return False

    def _post_migration_cleanup(self) -> None:
        # replace kv-associated form items with field region items
        to_shift_up: list[NodeItem] = []
        to_replace_with_fri: list[FormItem] = []
        for fri in self.field_regions:
            form_ancestor: FormItem | None = None
            curr: NodeItem = fri
            while True:
                if isinstance(curr, FormItem):
                    form_ancestor = curr
                    break
                elif curr.parent is None:
                    break
                curr = curr.parent.resolve(doc=self)
            if form_ancestor is not None:
                to_shift_up.append(fri)
                if form_ancestor not in to_replace_with_fri:
                    to_replace_with_fri.append(form_ancestor)
        for form_item in to_replace_with_fri:
            self._shift_down(
                old_subroot=form_item,
                new_subroot=FieldRegionItem(
                    self_ref="#",
                    prov=form_item.prov,
                    content_layer=form_item.content_layer,
                    meta=form_item.meta,
                    comments=form_item.comments,
                    source=form_item.source,
                ),
            )
            self._shift_up(old_subroot=form_item)
        for node in to_shift_up:
            self._shift_up(old_subroot=node)

        # handle any remaining (just value-associated) form items:
        value_groups: list[tuple[NodeItem, list[TextItem]]] = []  # tracking consecutive non-migrated values
        for outer, _ in self.iterate_items(
            with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
        ):
            if isinstance(outer, DocItem | GroupItem) and outer.label in {GroupLabel.FORM_AREA, DocItemLabel.FORM}:
                prev_is_value = False
                prev_level = -1
                for inner, level in self.iterate_items(
                    root=outer, with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
                ):
                    if (
                        isinstance(inner, TextItem)
                        and inner.label
                        in [DocItemLabel.EMPTY_VALUE, DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED]
                        and not (inner.parent and isinstance(inner.parent.resolve(doc=self), FieldValueItem))
                    ):
                        if prev_is_value and level == prev_level:
                            a, b = value_groups[-1]
                            value_groups[-1] = a, b + [inner]
                        else:
                            value_groups.append((outer, [inner]))
                        prev_is_value = True
                    else:
                        prev_is_value = False
                    prev_level = level

        already_shifted: list[NodeItem] = []
        for outer, vg in value_groups:
            if outer not in already_shifted:
                already_shifted.append(outer)
                if not self._has_field_region_ancestor(item=outer):
                    fri = FieldRegionItem(self_ref="#")
                    if isinstance(outer, DocItem):
                        fri.prov = outer.prov
                    self._shift_down(old_subroot=outer, new_subroot=fri)
                self._shift_up(old_subroot=outer)

            fi = FieldItem(self_ref="#")
            self.insert_item_before_sibling(new_item=fi, sibling=vg[0])
            for value_item in vg:
                fv = FieldValueItem(self_ref="#", orig="", text="")
                self._shift_down(old_subroot=value_item, new_subroot=fv)
                self._move_subtree(old_subroot=fv, new_subroot=fi)

        # handle any remaining form areas:
        to_shift_up = []
        for outer, _ in self.iterate_items(
            with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
        ):
            if isinstance(outer, GroupItem) and outer.label == GroupLabel.FORM_AREA:
                to_shift_up.append(outer)
        for node in to_shift_up:
            self._shift_up(old_subroot=node)

    # ---------------------------
    # Public Manipulation methods
    # ---------------------------

    def append_child_item(self, *, child: NodeItem, parent: Optional[NodeItem] = None) -> None:
        """Adds an item."""
        if len(child.children) > 0:
            raise ValueError("Can not append a child with children")

        parent = parent if parent is not None else self.body

        success, stack = self._get_stack_of_item(item=parent)

        if not success:
            raise ValueError(f"Could not resolve the parent node in the document tree: {parent}")

        # Append the item to the attributes of the doc
        self._append_item(item=child, parent_ref=parent.get_ref())

        # Update the tree of the doc
        success = self.body._add_child(doc=self, new_ref=child.get_ref(), stack=stack)

        # Clean the attribute (orphan) if not successful
        if not success:
            self._pop_item(item=child)
            raise ValueError(f"Could not append child: {child} to parent: {parent}")

    def insert_item_after_sibling(self, *, new_item: NodeItem, sibling: NodeItem) -> None:
        """Inserts an item, given its node_item instance, after other as a sibling."""
        self._insert_item_at_refitem(item=new_item, ref=sibling.get_ref(), after=True)

    def insert_item_before_sibling(self, *, new_item: NodeItem, sibling: NodeItem) -> None:
        """Inserts an item, given its node_item instance, before other as a sibling."""
        self._insert_item_at_refitem(item=new_item, ref=sibling.get_ref(), after=False)

    def delete_items(self, *, node_items: list[NodeItem]) -> None:
        """Deletes an item, given its instance or ref, and any children it has."""
        refs = []
        for _ in node_items:
            refs.append(_.get_ref())

        self._delete_items(refs=refs)

    def replace_item(self, *, new_item: NodeItem, old_item: NodeItem) -> None:
        """Replace item with new item."""
        self.insert_item_after_sibling(new_item=new_item, sibling=old_item)
        self.delete_items(node_items=[old_item])

    # ----------------------------
    # Private Manipulation methods
    # ----------------------------

    def _get_stack_of_item(self, item: NodeItem) -> tuple[bool, list[int]]:
        """Find the stack indices of the item."""
        return self._get_stack_of_refitem(ref=item.get_ref())

    def _get_stack_of_refitem(self, ref: RefItem) -> tuple[bool, list[int]]:
        """Find the stack indices of the reference."""
        if ref == self.body.get_ref():
            return (True, [])

        node = ref.resolve(doc=self)
        parent_ref = node._get_parent_ref(doc=self, stack=[])

        if parent_ref is None:
            return (False, [])

        stack: list[int] = []
        while parent_ref is not None:
            parent = parent_ref.resolve(doc=self)

            index = parent.children.index(node.get_ref())
            stack.insert(0, index)  # prepend the index

            node = parent
            parent_ref = node._get_parent_ref(doc=self, stack=[])

        return (True, stack)

    def _insert_item_at_refitem(self, item: NodeItem, ref: RefItem, after: bool) -> RefItem:
        """Insert node-item using the self-reference."""
        success, stack = self._get_stack_of_refitem(ref=ref)

        if not success:
            raise ValueError(f"Could not insert at {ref.cref}: could not find the stack")

        return self._insert_item_at_stack(item=item, stack=stack, after=after)

    def _append_item(self, *, item: NodeItem, parent_ref: RefItem) -> RefItem:
        """Append item of its type."""
        cref: str = ""  # to be updated

        if isinstance(item, TextItem):
            item_label = "texts"
            item_index = len(self.texts)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.texts.append(item)

        elif isinstance(item, TableItem):
            item_label = "tables"
            item_index = len(self.tables)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.tables.append(item)

        elif isinstance(item, PictureItem):
            item_label = "pictures"
            item_index = len(self.pictures)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.pictures.append(item)

        elif isinstance(item, KeyValueItem):
            item_label = "key_value_items"
            item_index = len(self.key_value_items)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.key_value_items.append(item)

        elif isinstance(item, FormItem):
            item_label = "form_items"
            item_index = len(self.form_items)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.form_items.append(item)

        elif isinstance(item, FieldRegionItem):
            item_label = "field_regions"
            item_index = len(self.field_regions)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.field_regions.append(item)

        elif isinstance(item, FieldItem):
            item_label = "field_items"
            item_index = len(self.field_items)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.field_items.append(item)

        elif isinstance(item, ListGroup | InlineGroup):
            item_label = "groups"
            item_index = len(self.groups)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.groups.append(item)
        elif isinstance(item, GroupItem):
            item_label = "groups"
            item_index = len(self.groups)

            cref = f"#/{item_label}/{item_index}"

            item.self_ref = cref
            item.parent = parent_ref

            self.groups.append(item)

        else:
            raise ValueError(f"Item {item} is not supported for insertion")

        return RefItem(cref=cref)

    def _pop_item(self, *, item: NodeItem):
        """Pop the last item of its type."""
        path = item.self_ref.split("/")

        if len(path) != 3:
            raise ValueError(f"Can not pop item with path: {path}")

        item_label = path[1]
        item_index = int(path[2])

        if len(self.__getattribute__(item_label)) == item_index + 1:  # we can only pop the last item
            del self.__getattribute__(item_label)[item_index]
        else:
            msg = f"index:{item_index}, len:{len(self.__getattribute__(item_label))}"
            raise ValueError(f"Failed to pop: item is not last ({msg})")

    def _insert_item_at_stack(self, item: NodeItem, stack: list[int], after: bool) -> RefItem:
        """Insert node-item using the self-reference."""
        parent_ref = self.body._get_parent_ref(doc=self, stack=stack)

        if parent_ref is None:
            raise ValueError(f"Could not find a parent at stack: {stack}")

        new_ref = self._append_item(item=item, parent_ref=parent_ref)

        success = self.body._add_sibling(doc=self, stack=stack, new_ref=new_ref, after=after)

        if not success:
            self._pop_item(item=item)

            raise ValueError(f"Could not insert item: {item} under parent: {parent_ref.resolve(doc=self)}")

        return item.get_ref()

    def _delete_items(self, refs: list[RefItem]):
        """Delete document item using the self-reference."""
        to_be_deleted_items: dict[tuple[int, ...], str] = {}  # stack to cref

        if not refs:
            return

        # Identify the to_be_deleted_items
        for item, stack in self._iterate_items_with_stack(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers=set(ContentLayer),
        ):
            ref = item.get_ref()

            if ref in refs:
                to_be_deleted_items[tuple(stack)] = ref.cref

            substacks = [stack[0 : i + 1] for i in range(len(stack) - 1)]
            for substack in substacks:
                if tuple(substack) in to_be_deleted_items:
                    to_be_deleted_items[tuple(stack)] = ref.cref

        if len(to_be_deleted_items) < len(refs):
            raise ValueError(f"Cannot find all provided RefItems in doc: {[r.cref for r in refs]}")

        # Clean the tree, reverse the order to not have to update
        for stack_, ref_ in sorted(to_be_deleted_items.items(), reverse=True):
            success = self.body._delete_child(doc=self, stack=list(stack_))

            if not success:
                del to_be_deleted_items[stack_]
            else:
                _logger.debug(f"deleted item in tree at stack: {stack_} => {ref_}")

        # Create a new lookup of the orphans:
        # dict of item_label (`texts`, `tables`, ...) to a
        # dict of item_label with delta (default = -1).
        lookup: dict[str, dict[int, int]] = {}

        for stack_, ref_ in to_be_deleted_items.items():
            path = ref_.split("/")
            if len(path) == 3:
                item_label = path[1]
                item_index = int(path[2])

                if item_label not in lookup:
                    lookup[item_label] = {}

                lookup[item_label][item_index] = -1

        # Remove the orphans in reverse order
        for item_label, item_inds in lookup.items():
            for item_index, _ in sorted(item_inds.items(), reverse=True):  # delete the last in the list first!
                _logger.debug(f"deleting item in doc for {item_label} for {item_index}")
                del self.__getattribute__(item_label)[item_index]

        self._update_breadth_first_with_lookup(node=self.body, refs_to_be_deleted=refs, lookup=lookup)

    # Update the references
    def _update_ref_with_lookup(self, item_label: str, item_index: int, lookup: dict[str, dict[int, int]]) -> RefItem:
        """Update ref with lookup."""
        if item_label not in lookup:  # Nothing to be done
            return RefItem(cref=f"#/{item_label}/{item_index}")

        # Count how many items have been deleted in front of you
        delta = sum(val if item_index >= key else 0 for key, val in lookup[item_label].items())
        new_index = item_index + delta

        return RefItem(cref=f"#/{item_label}/{new_index}")

    def _update_refitems_with_lookup(
        self,
        ref_items: list[RefItem],
        refs_to_be_deleted: list[RefItem],
        lookup: dict[str, dict[int, int]],
    ) -> list[RefItem]:
        """Update refitems with lookup."""
        new_refitems = []
        for ref_item in ref_items:
            if ref_item not in refs_to_be_deleted:  # if ref_item is in ref, then delete/skip them
                path = ref_item._split_ref_to_path()
                if len(path) == 3:
                    new_refitems.append(
                        self._update_ref_with_lookup(
                            item_label=path[1],
                            item_index=int(path[2]),
                            lookup=lookup,
                        )
                    )
                else:
                    new_refitems.append(ref_item)

        return new_refitems

    def _update_breadth_first_with_lookup(
        self,
        node: NodeItem,
        refs_to_be_deleted: list[RefItem],
        lookup: dict[str, dict[int, int]],
    ):
        """Update breadth first with lookup."""
        # Update the comments references on any DocItem
        if isinstance(node, DocItem):
            node.comments = [ref_item for ref_item in node.comments if ref_item not in refs_to_be_deleted]
            for ref_item in node.comments:
                ref_item._update_with_lookup(lookup=lookup)

        # Update the captions, references and footnote references
        if isinstance(node, FloatingItem):
            node.captions = self._update_refitems_with_lookup(
                ref_items=node.captions,
                refs_to_be_deleted=refs_to_be_deleted,
                lookup=lookup,
            )
            node.references = self._update_refitems_with_lookup(
                ref_items=node.references,
                refs_to_be_deleted=refs_to_be_deleted,
                lookup=lookup,
            )
            node.footnotes = self._update_refitems_with_lookup(
                ref_items=node.footnotes,
                refs_to_be_deleted=refs_to_be_deleted,
                lookup=lookup,
            )
            if isinstance(node, TableItem):
                for cell in node.data.table_cells:
                    if isinstance(cell, RichTableCell):
                        path = cell.ref._split_ref_to_path()
                        cell.ref = self._update_ref_with_lookup(
                            item_label=path[1],
                            item_index=int(path[2]),
                            lookup=lookup,
                        )

        # Update the self_ref reference
        if node.parent is not None:
            path = node.parent._split_ref_to_path()
            if len(path) == 3:
                node.parent = self._update_ref_with_lookup(item_label=path[1], item_index=int(path[2]), lookup=lookup)

        # Update the parent reference
        if node.self_ref is not None:
            path = node.self_ref.split("/")
            if len(path) == 3:
                _ref = self._update_ref_with_lookup(item_label=path[1], item_index=int(path[2]), lookup=lookup)
                node.self_ref = _ref.cref

        # Update the child references
        node.children = self._update_refitems_with_lookup(
            ref_items=node.children,
            refs_to_be_deleted=refs_to_be_deleted,
            lookup=lookup,
        )

        for i, child_ref in enumerate(node.children):
            node = child_ref.resolve(self)
            self._update_breadth_first_with_lookup(node=node, refs_to_be_deleted=refs_to_be_deleted, lookup=lookup)

    def _shift_up(self, *, old_subroot: NodeItem) -> None:
        """Move a subtree up in the document tree, removing the old subroot.

        :param old_subroot: The root of the subtree to move up (NodeItem should be part of the document tree)
        :return: None
        """
        if old_subroot.parent is None:
            raise ValueError("Can not shift up the root")

        # use old subroot's parent as new subroot
        new_subroot: NodeItem = old_subroot.parent.resolve(doc=self)

        old_subr_idx = new_subroot.children.index(old_subroot.get_ref())
        for i, child_ref in enumerate(old_subroot.children):
            # link new subroot => children (right after the old subroot)
            new_subroot.children.insert(old_subr_idx + i + 1, child_ref)

            # link children => new subroot
            child: NodeItem = child_ref.resolve(doc=self)
            child.parent = new_subroot.get_ref()

        # unlink old subroot its parent
        new_subroot.children.remove(old_subroot.get_ref())

    def _shift_down(self, *, old_subroot: NodeItem, new_subroot: NodeItem) -> None:
        """Move a subtree down in the document tree, introducing a new subroot.

        :param old_subroot: The subtree to move down (NodeItem must be part of the document tree)
        :param new_subroot: The new subroot to introduce (NodeItem must not be part of the document tree)
        :return: None
        """
        if old_subroot.parent is None:
            raise ValueError("Can not shift down the root")

        self.insert_item_before_sibling(new_item=new_subroot, sibling=old_subroot)

        self._move_subtree(old_subroot=old_subroot, new_subroot=new_subroot)

    def _move_subtree(self, *, old_subroot: NodeItem, new_subroot: NodeItem, pos: int | None = None) -> None:
        """Move a subtree to a new parent."""
        if old_subroot.parent is None:
            raise ValueError("Can not move the root")

        # unlink old subroot from its previous parent
        previous_parent: NodeItem = old_subroot.parent.resolve(doc=self)
        previous_parent.children.remove(old_subroot.get_ref())

        # link new subroot => old subroot
        if pos is not None:
            new_subroot.children.insert(pos, old_subroot.get_ref())
        else:
            new_subroot.children.append(old_subroot.get_ref())

        # link old subroot => new subroot
        old_subroot.parent = new_subroot.get_ref()

    def _get_heading_level(self, node: NodeItem) -> Optional[int]:
        """Get the level of a node if it has heading semantics (TitleItem, SectionHeaderItem or root), else None."""
        if isinstance(node, TitleItem):
            return 0
        elif isinstance(node, SectionHeaderItem):
            return node.level
        elif node == self.body:
            return -1
        else:
            return None

    def _is_descendant_of(self, node1: NodeItem, node2: NodeItem) -> bool:
        """Check if node1 is a descendant of node2."""
        curr = node1
        while curr.parent is not None:
            curr = curr.parent.resolve(doc=self)
            if curr == node2:
                return True
        return False

    def _hierarchize(self):
        """Structure the document's titles and headings into an explicit hierarchy based on their levels."""
        section_root_by_level: dict[int, Union[TitleItem, SectionHeaderItem]] = {-1: self.body}
        resume_node: Optional[NodeItem] = self.body

        while resume_node:
            for item, _ in self.iterate_items(
                with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
            ):
                # mechanism for resuming the traversal from a desired node
                if resume_node:
                    if item != resume_node:
                        continue
                    else:
                        resume_node = None

                # Skip items that are descendants of FloatingItem containers (tables, pictures, key-values, forms)
                # These have structural parent-child relationships that must be preserved:
                if item.parent:
                    curr = item
                    is_floating_descendant = False
                    while curr.parent is not None:
                        parent = curr.parent.resolve(doc=self)
                        if isinstance(parent, FloatingItem):
                            is_floating_descendant = True
                            break
                        curr = parent

                    if is_floating_descendant:
                        continue

                # determine which section root this item should belong to
                introduced_level = self._get_heading_level(node=item)
                target_root_level = -1
                for level in section_root_by_level.keys():
                    if level > target_root_level and (introduced_level is None or level < introduced_level):
                        target_root_level = level
                target_section_root = section_root_by_level[target_root_level]

                # ensure the item is hooked to the target root
                if item.parent and not self._is_descendant_of(node1=item, node2=target_section_root):
                    # print(f"Will move {item.self_ref=} to {target_root.self_ref=}")
                    self._move_subtree(old_subroot=item, new_subroot=target_section_root)
                    # in case of moving, restart traversal from the right place in the updated tree
                    resume_node = item
                    break

                if introduced_level is not None:  # i.e. a heading was found
                    # remove shadowed headings
                    keys_to_delete = [k for k in section_root_by_level if k >= introduced_level]
                    for k in keys_to_delete:
                        del section_root_by_level[k]
                    section_root_by_level[introduced_level] = item

    def _flatten(self):
        """Flatten the document's titles and headings into a single level."""
        resume_node: Optional[NodeItem] = self.body

        while resume_node:
            for item, _ in self.iterate_items(
                with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
            ):
                # mechanism for resuming the traversal from a desired node
                if resume_node:
                    if item != resume_node:
                        continue
                    else:
                        resume_node = None

                if isinstance(item, TitleItem | SectionHeaderItem):
                    child_refs_to_move = [
                        child_ref
                        for idx, child_ref in enumerate(item.children)
                        if not (  # don't move inline groups that capture the actual heading text (e.g. rich)
                            item.text == "" and idx == 0 and isinstance(child_ref.resolve(doc=self), InlineGroup)
                        )
                    ]
                    if child_refs_to_move:
                        self._shift_down(
                            old_subroot=item,
                            new_subroot=GroupItem(self_ref="#"),
                        )
                        tmp_node = item.parent.resolve(doc=self)

                        for child_ref in child_refs_to_move:
                            child = child_ref.resolve(doc=self)
                            self._move_subtree(old_subroot=child, new_subroot=tmp_node)
                        self._shift_up(old_subroot=tmp_node)
                        resume_node = item
                        break

    ###################################
    # TODO: refactor add* methods below
    ###################################

    def add_list_group(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> ListGroup:
        """add_list_group."""
        _parent = parent or self.body
        cref = f"#/groups/{len(self.groups)}"
        group = ListGroup(self_ref=cref, parent=_parent.get_ref())
        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self.groups.append(group)
        _parent.children.append(RefItem(cref=cref))
        return group

    @deprecated("Use add_list_group() instead.")
    def add_ordered_list(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> GroupItem:
        """add_ordered_list."""
        return self.add_list_group(
            name=name,
            parent=parent,
            content_layer=content_layer,
        )

    @deprecated("Use add_list_group() instead.")
    def add_unordered_list(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> GroupItem:
        """add_unordered_list."""
        return self.add_list_group(
            name=name,
            parent=parent,
            content_layer=content_layer,
        )

    def add_inline_group(
        self,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> InlineGroup:
        """add_inline_group."""
        _parent = parent or self.body
        cref = f"#/groups/{len(self.groups)}"
        group = InlineGroup(self_ref=cref, parent=_parent.get_ref())
        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self.groups.append(group)
        _parent.children.append(RefItem(cref=cref))
        return group

    def add_group(
        self,
        label: Optional[GroupLabel] = None,
        name: Optional[str] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> GroupItem:
        """add_group.

        :param label: Optional[GroupLabel]:  (Default value = None)
        :param name: Optional[str]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if label in [GroupLabel.LIST, GroupLabel.ORDERED_LIST]:
            return self.add_list_group(
                name=name,
                parent=parent,
                content_layer=content_layer,
            )
        elif label == GroupLabel.INLINE:
            return self.add_inline_group(
                name=name,
                parent=parent,
                content_layer=content_layer,
            )

        if not parent:
            parent = self.body

        group_index = len(self.groups)
        cref = f"#/groups/{group_index}"

        group = GroupItem(self_ref=cref, parent=parent.get_ref())
        if name is not None:
            group.name = name
        if label is not None:
            group.label = label
        if content_layer:
            group.content_layer = content_layer

        self.groups.append(group)
        parent.children.append(RefItem(cref=cref))

        return group

    def add_list_item(
        self,
        text: str,
        enumerated: bool = False,
        marker: Optional[str] = None,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_list_item.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        if not isinstance(parent, ListGroup):
            warnings.warn(
                "ListItem parent must be a list group, creating one on the fly.",
                DeprecationWarning,
            )
            parent = self.add_list_group(parent=parent)

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            enumerated=enumerated,
            marker=marker or "",
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            list_item.prov.append(prov)
        if content_layer:
            list_item.content_layer = content_layer

        self.texts.append(list_item)
        parent.children.append(RefItem(cref=cref))

        return list_item

    def add_text(
        self,
        label: DocItemLabel,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        *,
        source: Optional[SourceType] = None,
        **kwargs: Any,
    ):
        """add_text.

        :param label: str:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)

        """
        # Catch a few cases that are in principle allowed
        # but that will create confusion down the road
        if label in [DocItemLabel.TITLE]:
            return self.add_title(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif label in [DocItemLabel.LIST_ITEM]:
            return self.add_list_item(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif label in [DocItemLabel.SECTION_HEADER]:
            return self.add_heading(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                **kwargs,
            )

        elif label in [DocItemLabel.CODE]:
            return self.add_code(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )
        elif label in [DocItemLabel.FORMULA]:
            return self.add_formula(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
            )
        elif label in [DocItemLabel.FIELD_HEADING]:
            return self.add_field_heading(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                **kwargs,
            )
        elif label in [DocItemLabel.FIELD_VALUE]:
            return self.add_field_value(
                text=text,
                orig=orig,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                **kwargs,
            )

        else:
            if not parent:
                parent = self.body

            if not orig:
                orig = text

            text_index = len(self.texts)
            cref = f"#/texts/{text_index}"
            text_item = TextItem(
                label=label,
                text=text,
                orig=orig,
                self_ref=cref,
                parent=parent.get_ref(),
                formatting=formatting,
                hyperlink=hyperlink,
            )
            if prov:
                text_item.prov.append(prov)
            if source:
                text_item.source.append(source)

            if content_layer:
                text_item.content_layer = content_layer

            self.texts.append(text_item)
            parent.children.append(RefItem(cref=cref))

            return text_item

    def add_comment(
        self,
        *,
        text: str,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        targets: Optional[list[Union[DocItem, tuple[DocItem, tuple[int, int]]]]] = None,
    ):
        """Adds a comment to the document, assigning it to the given targets.

        :param text: str:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param targets: list[Union[DocItem, tuple[DocItem, tuple[int, int]]]]:  (Default value = None) Each list element
            can be either a single DocItem or a tuple of a DocItem and a span range (start_inclusive, end_exclusive).
        """
        item = self.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=prov,
            parent=parent,
            content_layer=ContentLayer.NOTES,
        )
        if targets:
            for target in targets:
                range = None
                if isinstance(target, tuple):
                    target, range = target
                ref = FineRef(cref=item.self_ref, range=range)
                target.comments.append(ref)
        return item

    def add_table(
        self,
        data: TableData,
        caption: Optional[Union[TextItem, RefItem]] = None,  # This is not cool yet.
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        label: DocItemLabel = DocItemLabel.TABLE,
        content_layer: Optional[ContentLayer] = None,
        annotations: Optional[list[TableAnnotationType]] = None,
    ):
        """add_table.

        :param data: TableData:
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param label: DocItemLabel:  (Default value = DocItemLabel.TABLE)

        """
        if not parent:
            parent = self.body

        table_index = len(self.tables)
        cref = f"#/tables/{table_index}"

        tbl_item = TableItem(
            label=label,
            data=data,
            self_ref=cref,
            parent=parent.get_ref(),
            annotations=annotations or [],
        )
        if prov:
            tbl_item.prov.append(prov)
        if content_layer:
            tbl_item.content_layer = content_layer

        if caption:
            tbl_item.captions.append(caption.get_ref())

        self.tables.append(tbl_item)
        parent.children.append(RefItem(cref=cref))

        return tbl_item

    def add_picture(
        self,
        annotations: Optional[list[PictureDataType]] = None,
        image: Optional[ImageRef] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ):
        """add_picture.

        :param data: Optional[list[PictureData]]: (Default value = None)
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        picture_index = len(self.pictures)
        cref = f"#/pictures/{picture_index}"

        fig_item = PictureItem(
            label=DocItemLabel.PICTURE,
            annotations=annotations or [],
            image=image,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            fig_item.prov.append(prov)
        if content_layer:
            fig_item.content_layer = content_layer
        if caption:
            fig_item.captions.append(caption.get_ref())

        self.pictures.append(fig_item)
        parent.children.append(RefItem(cref=cref))

        return fig_item

    def add_title(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_title.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        item = TitleItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            item.prov.append(prov)
        if content_layer:
            item.content_layer = content_layer

        self.texts.append(item)
        parent.children.append(RefItem(cref=cref))

        return item

    def add_code(
        self,
        text: str,
        code_language: Optional[CodeLanguageLabel] = None,
        orig: Optional[str] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_code.

        :param text: str:
        :param code_language: Optional[CodeLanguageLabel]: (Default value = None)
        :param orig: Optional[str]:  (Default value = None)
        :param caption: Optional[Union[TextItem:
        :param RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        code_item = CodeItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if code_language:
            code_item.code_language = code_language
        if content_layer:
            code_item.content_layer = content_layer
        if prov:
            code_item.prov.append(prov)
        if caption:
            code_item.captions.append(caption.get_ref())

        self.texts.append(code_item)
        parent.children.append(RefItem(cref=cref))

        return code_item

    def add_formula(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_formula.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        section_header_item = FormulaItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            section_header_item.prov.append(prov)
        if content_layer:
            section_header_item.content_layer = content_layer

        self.texts.append(section_header_item)
        parent.children.append(RefItem(cref=cref))

        return section_header_item

    def add_heading(
        self,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_heading.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        section_header_item = SectionHeaderItem(
            level=level,
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            section_header_item.prov.append(prov)
        if content_layer:
            section_header_item.content_layer = content_layer

        self.texts.append(section_header_item)
        parent.children.append(RefItem(cref=cref))

        return section_header_item

    def add_key_values(
        self,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_key_values.

        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        key_value_index = len(self.key_value_items)
        cref = f"#/key_value_items/{key_value_index}"

        kv_item = KeyValueItem(
            graph=graph,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            kv_item.prov.append(prov)

        self.key_value_items.append(kv_item)
        parent.children.append(RefItem(cref=cref))

        return kv_item

    def add_form(
        self,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ):
        """add_form.

        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        form_index = len(self.form_items)
        cref = f"#/form_items/{form_index}"

        form_item = FormItem(
            graph=graph,
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            form_item.prov.append(prov)

        self.form_items.append(form_item)
        parent.children.append(RefItem(cref=cref))

        return form_item

    def add_field_region(
        self,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
    ) -> FieldRegionItem:
        """add_field_region.

        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        key_value_index = len(self.field_regions)
        cref = f"#/field_regions/{key_value_index}"

        kv_item = FieldRegionItem(
            self_ref=cref,
            parent=parent.get_ref(),
        )
        if prov:
            kv_item.prov.append(prov)

        self.field_regions.append(kv_item)
        parent.children.append(RefItem(cref=cref))

        return kv_item

    def add_field_heading(
        self,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_kv_heading.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        item = FieldHeadingItem(
            level=level,
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if prov:
            item.prov.append(prov)
        if content_layer:
            item.content_layer = content_layer

        self.texts.append(item)
        parent.children.append(RefItem(cref=cref))

        return item

    def add_field_item(
        self,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
    ) -> FieldItem:
        """add_kv_entry."""
        _parent = parent or self.body
        cref = f"#/field_items/{len(self.field_items)}"
        item = FieldItem(
            self_ref=cref,
            parent=_parent.get_ref(),
        )
        if prov:
            item.prov.append(prov)
        if content_layer:
            item.content_layer = content_layer

        self.field_items.append(item)
        _parent.children.append(RefItem(cref=cref))
        return item

    def add_field_key(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_field_key.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        """
        item = self.add_text(
            label=DocItemLabel.FIELD_KEY,
            text=text,
            orig=orig,
            prov=prov,
            parent=parent,
            content_layer=content_layer,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return item

    def add_field_value(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        kind: Optional[typing.Literal["read_only", "fillable"]] = "read_only",
    ):
        """add_field_value.

        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param kind: Optional[typing.Literal["read_only", "fillable", "fillable_with_hint"]]:  (Default value = "read_only")
        """
        if not parent:
            parent = self.body

        if not orig:
            orig = text

        text_index = len(self.texts)
        cref = f"#/texts/{text_index}"
        item = FieldValueItem(
            text=text,
            orig=orig,
            self_ref=cref,
            parent=parent.get_ref(),
            formatting=formatting,
            hyperlink=hyperlink,
            kind=kind,
        )
        if prov:
            item.prov.append(prov)
        if content_layer:
            item.content_layer = content_layer

        self.texts.append(item)
        parent.children.append(RefItem(cref=cref))

        return item

    def add_field_hint(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_field_hint.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        """
        item = self.add_text(
            label=DocItemLabel.FIELD_HINT,
            text=text,
            orig=orig,
            prov=prov,
            parent=parent,
            content_layer=content_layer,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return item

    def add_marker(
        self,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        parent: Optional[NodeItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        """add_marker.

        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param parent: Optional[NodeItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        """
        item = self.add_text(
            label=DocItemLabel.MARKER,
            text=text,
            orig=orig,
            prov=prov,
            parent=parent,
            content_layer=content_layer,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return item

    # ---------------------------
    # Node Item Insertion Methods
    # ---------------------------

    def _get_insertion_stack_and_parent(self, sibling: NodeItem) -> tuple[list[int], RefItem]:
        """Get the stack and parent reference for inserting a new item at a sibling."""
        # Get the stack of the sibling
        sibling_ref = sibling.get_ref()

        success, stack = self._get_stack_of_refitem(ref=sibling_ref)

        if not success:
            raise ValueError(f"Could not insert at {sibling_ref.cref}: could not find the stack")

        # Get the parent RefItem
        parent_ref = self.body._get_parent_ref(doc=self, stack=stack)

        if parent_ref is None:
            raise ValueError(f"Could not find a parent at stack: {stack}")

        return stack, parent_ref

    def _insert_in_structure(
        self,
        item: NodeItem,
        stack: list[int],
        after: bool,
        created_parent: Optional[bool] = False,
    ) -> None:
        """Insert item into the document structure at the specified stack and handle errors."""
        # Ensure the item has a parent reference
        if item.parent is None:
            item.parent = self.body.get_ref()

        self._append_item(item=item, parent_ref=item.parent)

        new_ref = item.get_ref()

        success = self.body._add_sibling(doc=self, stack=stack, new_ref=new_ref, after=after)

        # Error handling can be determined here
        if not success:
            self._pop_item(item=item)

            if created_parent:
                self.delete_items(node_items=[item.parent.resolve(self)])

            raise ValueError(f"Could not insert item: {item} under parent: {item.parent.resolve(doc=self)}")

    def insert_list_group(
        self,
        sibling: NodeItem,
        name: Optional[str] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> ListGroup:
        """Creates a new ListGroup item and inserts it into the document.

        :param sibling: NodeItem:
        :param name: Optional[str]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: ListGroup: The newly created ListGroup item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        group = ListGroup(self_ref="#", parent=parent_ref)

        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self._insert_in_structure(item=group, stack=stack, after=after)

        return group

    def insert_inline_group(
        self,
        sibling: NodeItem,
        name: Optional[str] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> InlineGroup:
        """Creates a new InlineGroup item and inserts it into the document.

        :param sibling: NodeItem:
        :param name: Optional[str]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: InlineGroup: The newly created InlineGroup item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new InlineGroup NodeItem
        group = InlineGroup(self_ref="#", parent=parent_ref)

        if name is not None:
            group.name = name
        if content_layer:
            group.content_layer = content_layer

        self._insert_in_structure(item=group, stack=stack, after=after)

        return group

    def insert_group(
        self,
        sibling: NodeItem,
        label: Optional[GroupLabel] = None,
        name: Optional[str] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> GroupItem:
        """Creates a new GroupItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param label: Optional[GroupLabel]:  (Default value = None)
        :param name: Optional[str]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: GroupItem: The newly created GroupItem.
        """
        if label in [GroupLabel.LIST, GroupLabel.ORDERED_LIST]:
            return self.insert_list_group(
                sibling=sibling,
                name=name,
                content_layer=content_layer,
                after=after,
            )
        elif label == GroupLabel.INLINE:
            return self.insert_inline_group(
                sibling=sibling,
                name=name,
                content_layer=content_layer,
                after=after,
            )

        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new GroupItem NodeItem
        group = GroupItem(self_ref="#", parent=parent_ref)

        if name is not None:
            group.name = name
        if label is not None:
            group.label = label
        if content_layer:
            group.content_layer = content_layer

        self._insert_in_structure(item=group, stack=stack, after=after)

        return group

    def insert_list_item(
        self,
        sibling: NodeItem,
        text: str,
        enumerated: bool = False,
        marker: Optional[str] = None,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> ListItem:
        """Creates a new ListItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param enumerated: bool:  (Default value = False)
        :param marker: Optional[str]:  (Default value = None)
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: ListItem: The newly created ListItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Ensure the parent is a ListGroup

        parent = parent_ref.resolve(self)
        set_parent = False

        if not isinstance(parent, ListGroup):
            warnings.warn(
                "ListItem parent must be a ListGroup, creating one on the fly.",
                DeprecationWarning,
            )
            parent = self.insert_list_group(sibling=sibling, after=after)
            parent_ref = parent.get_ref()
            if after:
                stack[-1] += 1
            stack.append(0)
            after = False
            set_parent = True

        # Create a new ListItem NodeItem
        if not orig:
            orig = text

        list_item = ListItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            enumerated=enumerated,
            marker=marker or "",
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            list_item.prov.append(prov)
        if content_layer:
            list_item.content_layer = content_layer

        self._insert_in_structure(item=list_item, stack=stack, after=after, created_parent=set_parent)

        return list_item

    def insert_text(
        self,
        sibling: NodeItem,
        label: DocItemLabel,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> TextItem:
        """Creates a new TextItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param label: DocItemLabel:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: TextItem: The newly created TextItem item.
        """
        if label in [DocItemLabel.TITLE]:
            return self.insert_title(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.LIST_ITEM]:
            return self.insert_list_item(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.SECTION_HEADER]:
            return self.insert_heading(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.CODE]:
            return self.insert_code(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        elif label in [DocItemLabel.FORMULA]:
            return self.insert_formula(
                sibling=sibling,
                text=text,
                orig=orig,
                prov=prov,
                content_layer=content_layer,
                formatting=formatting,
                hyperlink=hyperlink,
                after=after,
            )

        else:
            # Get stack and parent reference of the sibling
            stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

            # Create a new TextItem NodeItem
            if not orig:
                orig = text

            text_item = TextItem(
                label=label,
                text=text,
                orig=orig,
                self_ref="#",
                parent=parent_ref,
                formatting=formatting,
                hyperlink=hyperlink,
            )

            if prov:
                text_item.prov.append(prov)
            if content_layer:
                text_item.content_layer = content_layer

            self._insert_in_structure(item=text_item, stack=stack, after=after)

            return text_item

    def insert_table(
        self,
        sibling: NodeItem,
        data: TableData,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        label: DocItemLabel = DocItemLabel.TABLE,
        content_layer: Optional[ContentLayer] = None,
        annotations: Optional[list[TableAnnotationType]] = None,
        after: bool = True,
    ) -> TableItem:
        """Creates a new TableItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param data: TableData:
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param label: DocItemLabel:  (Default value = DocItemLabel.TABLE)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param annotations: Optional[list[TableAnnotationType]]: (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: TableItem: The newly created TableItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new ListItem NodeItem
        table_item = TableItem(
            label=label,
            data=data,
            self_ref="#",
            parent=parent_ref,
            annotations=annotations or [],
        )

        if prov:
            table_item.prov.append(prov)
        if content_layer:
            table_item.content_layer = content_layer
        if caption:
            table_item.captions.append(caption.get_ref())

        self._insert_in_structure(item=table_item, stack=stack, after=after)

        return table_item

    def insert_picture(
        self,
        sibling: NodeItem,
        annotations: Optional[list[PictureDataType]] = None,
        image: Optional[ImageRef] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        after: bool = True,
    ) -> PictureItem:
        """Creates a new PictureItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param annotations: Optional[list[PictureDataType]]: (Default value = None)
        :param image: Optional[ImageRef]:  (Default value = None)
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: PictureItem: The newly created PictureItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new PictureItem NodeItem
        picture_item = PictureItem(
            label=DocItemLabel.PICTURE,
            annotations=annotations or [],
            image=image,
            self_ref="#",
            parent=parent_ref,
        )

        if prov:
            picture_item.prov.append(prov)
        if content_layer:
            picture_item.content_layer = content_layer
        if caption:
            picture_item.captions.append(caption.get_ref())

        self._insert_in_structure(item=picture_item, stack=stack, after=after)

        return picture_item

    def insert_title(
        self,
        sibling: NodeItem,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> TitleItem:
        """Creates a new TitleItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: TitleItem: The newly created TitleItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new TitleItem NodeItem
        if not orig:
            orig = text

        title_item = TitleItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            title_item.prov.append(prov)
        if content_layer:
            title_item.content_layer = content_layer

        self._insert_in_structure(item=title_item, stack=stack, after=after)

        return title_item

    def insert_code(
        self,
        sibling: NodeItem,
        text: str,
        code_language: Optional[CodeLanguageLabel] = None,
        orig: Optional[str] = None,
        caption: Optional[Union[TextItem, RefItem]] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> CodeItem:
        """Creates a new CodeItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param code_language: Optional[str]: (Default value = None)
        :param orig: Optional[str]:  (Default value = None)
        :param caption: Optional[Union[TextItem, RefItem]]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: CodeItem: The newly created CodeItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new CodeItem NodeItem
        if not orig:
            orig = text

        code_item = CodeItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if code_language:
            code_item.code_language = code_language
        if content_layer:
            code_item.content_layer = content_layer
        if prov:
            code_item.prov.append(prov)
        if caption:
            code_item.captions.append(caption.get_ref())

        self._insert_in_structure(item=code_item, stack=stack, after=after)

        return code_item

    def insert_formula(
        self,
        sibling: NodeItem,
        text: str,
        orig: Optional[str] = None,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> FormulaItem:
        """Creates a new FormulaItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: FormulaItem: The newly created FormulaItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new FormulaItem NodeItem
        if not orig:
            orig = text

        formula_item = FormulaItem(
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            formula_item.prov.append(prov)
        if content_layer:
            formula_item.content_layer = content_layer

        self._insert_in_structure(item=formula_item, stack=stack, after=after)

        return formula_item

    def insert_heading(
        self,
        sibling: NodeItem,
        text: str,
        orig: Optional[str] = None,
        level: LevelNumber = 1,
        prov: Optional[ProvenanceItem] = None,
        content_layer: Optional[ContentLayer] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        after: bool = True,
    ) -> SectionHeaderItem:
        """Creates a new SectionHeaderItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param text: str:
        :param orig: Optional[str]:  (Default value = None)
        :param level: LevelNumber:  (Default value = 1)
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param content_layer: Optional[ContentLayer]:  (Default value = None)
        :param formatting: Optional[Formatting]:  (Default value = None)
        :param hyperlink: Optional[Union[AnyUrl, Path]]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: SectionHeaderItem: The newly created SectionHeaderItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new SectionHeaderItem NodeItem
        if not orig:
            orig = text

        section_header_item = SectionHeaderItem(
            level=level,
            text=text,
            orig=orig,
            self_ref="#",
            parent=parent_ref,
            formatting=formatting,
            hyperlink=hyperlink,
        )

        if prov:
            section_header_item.prov.append(prov)
        if content_layer:
            section_header_item.content_layer = content_layer

        self._insert_in_structure(item=section_header_item, stack=stack, after=after)

        return section_header_item

    def insert_key_values(
        self,
        sibling: NodeItem,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        after: bool = True,
    ) -> KeyValueItem:
        """Creates a new KeyValueItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: KeyValueItem: The newly created KeyValueItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new KeyValueItem NodeItem
        key_value_item = KeyValueItem(graph=graph, self_ref="#", parent=parent_ref)

        if prov:
            key_value_item.prov.append(prov)

        self._insert_in_structure(item=key_value_item, stack=stack, after=after)

        return key_value_item

    def insert_form(
        self,
        sibling: NodeItem,
        graph: GraphData,
        prov: Optional[ProvenanceItem] = None,
        after: bool = True,
    ) -> FormItem:
        """Creates a new FormItem item and inserts it into the document.

        :param sibling: NodeItem:
        :param graph: GraphData:
        :param prov: Optional[ProvenanceItem]:  (Default value = None)
        :param after: bool:  (Default value = True)

        :returns: FormItem: The newly created FormItem item.
        """
        # Get stack and parent reference of the sibling
        stack, parent_ref = self._get_insertion_stack_and_parent(sibling=sibling)

        # Create a new FormItem NodeItem
        form_item = FormItem(graph=graph, self_ref="#", parent=parent_ref)

        if prov:
            form_item.prov.append(prov)

        self._insert_in_structure(item=form_item, stack=stack, after=after)

        return form_item

    # ---------------------------
    # Range Manipulation Methods
    # ---------------------------

    def delete_items_range(
        self,
        *,
        start: NodeItem,
        end: NodeItem,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
    ) -> None:
        """Deletes all NodeItems and their children in the range from the start NodeItem to the end NodeItem.

        :param start: NodeItem:  The starting NodeItem of the range
        :param end: NodeItem:  The ending NodeItem of the range
        :param start_inclusive: bool:  (Default value = True):  If True, the start NodeItem will also be deleted
        :param end_inclusive: bool:  (Default value = True):  If True, the end NodeItem will also be deleted

        :returns: None
        """
        start_parent_ref = start.parent if start.parent is not None else self.body.get_ref()
        end_parent_ref = end.parent if end.parent is not None else self.body.get_ref()

        if start.parent != end.parent:
            raise ValueError("Start and end NodeItems must have the same parent to delete a range.")

        start_ref = start.get_ref()
        end_ref = end.get_ref()

        start_parent = start_parent_ref.resolve(doc=self)
        end_parent = end_parent_ref.resolve(doc=self)

        start_index = start_parent.children.index(start_ref)
        end_index = end_parent.children.index(end_ref)

        if start_index > end_index:
            raise ValueError(
                "Start NodeItem must come before or be the same as the end NodeItem in the document structure."
            )

        to_delete = start_parent.children[start_index : end_index + 1]

        if not start_inclusive:
            to_delete = to_delete[1:]
        if not end_inclusive:
            to_delete = to_delete[:-1]

        self._delete_items(refs=to_delete)

    def extract_items_range(
        self,
        *,
        start: NodeItem,
        end: NodeItem,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
        delete: bool = False,
    ) -> "DoclingDocument":
        """Extracts NodeItems and children in the range from the start NodeItem to the end as a new DoclingDocument.

        :param start: NodeItem:  The starting NodeItem of the range (must be a direct child of the document body)
        :param end: NodeItem:  The ending NodeItem of the range  (must be a direct child of the document body)
        :param start_inclusive: bool:  (Default value = True):  If True, the start NodeItem will also be extracted
        :param end_inclusive: bool:  (Default value = True):  If True, the end NodeItem will also be extracted
        :param delete: bool:  (Default value = False):  If True, extracted items are deleted in the original document

        :returns: DoclingDocument: A new document containing the extracted NodeItems and their children
        """
        if not start.parent == end.parent:
            raise ValueError("Start and end NodeItems must have the same parent to extract a range.")

        start_ref = start.get_ref()
        end_ref = end.get_ref()

        start_parent_ref = start.parent if start.parent is not None else self.body.get_ref()
        end_parent_ref = end.parent if end.parent is not None else self.body.get_ref()

        start_parent = start_parent_ref.resolve(doc=self)
        end_parent = end_parent_ref.resolve(doc=self)

        start_index = start_parent.children.index(start_ref) + (0 if start_inclusive else 1)
        end_index = end_parent.children.index(end_ref) + (1 if end_inclusive else 0)

        if start_index > end_index:
            raise ValueError(
                "Start NodeItem must come before or be the same as the end NodeItem in the document structure."
            )

        new_doc = DoclingDocument(name=f"{self.name}- Extracted Range")

        ref_items = start_parent.children[start_index:end_index]
        node_items = [ref.resolve(self) for ref in ref_items]

        new_doc.add_node_items(node_items=node_items, doc=self)

        if delete:
            self.delete_items_range(
                start=start,
                end=end,
                start_inclusive=start_inclusive,
                end_inclusive=end_inclusive,
            )

        return new_doc

    def insert_document(
        self,
        doc: "DoclingDocument",
        sibling: NodeItem,
        after: bool = True,
    ) -> None:
        """Inserts the content from the body of a DoclingDocument into this document at a specific position.

        :param doc: DoclingDocument: The document whose content will be inserted
        :param sibling: NodeItem: The NodeItem after/before which the new items will be inserted
        :param after: bool: If True, insert after the sibling; if False, insert before (Default value = True)

        :returns: None
        """
        ref_items = doc.body.children
        node_items = [ref.resolve(doc) for ref in ref_items]
        self.insert_node_items(sibling=sibling, node_items=node_items, doc=doc, after=after)

    def add_document(
        self,
        doc: "DoclingDocument",
        parent: Optional[NodeItem] = None,
    ) -> None:
        """Adds the content from the body of a DoclingDocument to this document under a specific parent.

        :param doc: DoclingDocument: The document whose content will be added
        :param parent: Optional[NodeItem]: The parent NodeItem under which new items are added (Default value = None)

        :returns: None
        """
        ref_items = doc.body.children
        node_items = [ref.resolve(doc) for ref in ref_items]
        self.add_node_items(node_items=node_items, doc=doc, parent=parent)

    def add_node_items(
        self,
        node_items: list[NodeItem],
        doc: "DoclingDocument",
        parent: Optional[NodeItem] = None,
    ) -> None:
        """Adds multiple NodeItems and their children under a parent in this document.

        :param node_items: list[NodeItem]: The NodeItems to be added
        :param doc: DoclingDocument: The document to which the NodeItems and their children belong
        :param parent: Optional[NodeItem]: The parent NodeItem under which new items are added (Default value = None)

        :returns: None
        """
        parent = self.body if parent is None else parent

        # Check for ListItem parent violations
        if not isinstance(parent, ListGroup):
            for item in node_items:
                if isinstance(item, ListItem):
                    raise ValueError("Cannot add ListItem into a non-ListGroup parent.")

        # Append the NodeItems to the document content

        parent_ref = parent.get_ref()

        new_refs = self._append_item_copies(node_items=node_items, parent_ref=parent_ref, doc=doc)

        # Add the new item refs in the document structure

        for ref in new_refs:
            parent.children.append(ref)

    def insert_node_items(
        self,
        sibling: NodeItem,
        node_items: list[NodeItem],
        doc: "DoclingDocument",
        after: bool = True,
    ) -> None:
        """Insert multiple NodeItems and their children at a specific position in the document.

        :param sibling: NodeItem: The NodeItem after/before which the new items will be inserted
        :param node_items: list[NodeItem]: The NodeItems to be inserted
        :param doc: DoclingDocument: The document to which the NodeItems and their children belong
        :param after: bool: If True, insert after the sibling; if False, insert before (Default value = True)

        :returns: None
        """
        # Check for ListItem parent violations
        parent = sibling.parent.resolve(self) if sibling.parent else self.body

        if not isinstance(parent, ListGroup):
            for item in node_items:
                if isinstance(item, ListItem):
                    raise ValueError("Cannot insert ListItem into a non-ListGroup parent.")

        # Append the NodeItems to the document content

        parent_ref = parent.get_ref()

        new_refs = self._append_item_copies(node_items=node_items, parent_ref=parent_ref, doc=doc)

        # Get the stack of the sibling

        sibling_ref = sibling.get_ref()

        success, stack = self._get_stack_of_refitem(ref=sibling_ref)

        if not success:
            raise ValueError(f"Could not insert at {sibling_ref.cref}: could not find the stack")

        # Insert the new item refs in the document structure

        reversed_new_refs = new_refs[::-1]

        for ref in reversed_new_refs:
            success = self.body._add_sibling(doc=self, stack=stack, new_ref=ref, after=after)

            if not success:
                raise ValueError(f"Could not insert item {ref.cref} at {sibling.get_ref().cref}")

    def _append_item_copies(
        self,
        node_items: list[NodeItem],
        parent_ref: RefItem,
        doc: "DoclingDocument",
    ) -> list[RefItem]:
        """Append node item copies (with their children) from a different document to the content of this document.

        :param node_items: list[NodeItem]: The NodeItems to be appended
        :param parent_ref: RefItem: The reference of the parent of the new items in this document
        :param doc: DoclingDocument: The document from which the NodeItems are taken

        :returns: list[RefItem]: A list of references to the newly added items in this document
        """
        new_refs: list[RefItem] = []

        for item in node_items:
            item_copy = item.model_copy(deep=True)

            self._append_item(item=item_copy, parent_ref=parent_ref)

            if item_copy.children:
                children_node_items = [ref.resolve(doc) for ref in item_copy.children]

                item_copy.children = self._append_item_copies(
                    node_items=children_node_items,
                    parent_ref=item_copy.get_ref(),
                    doc=doc,
                )

            new_ref = item_copy.get_ref()
            new_refs.append(new_ref)

        return new_refs

    def num_pages(self):
        """num_pages."""
        return len(self.pages.values())

    def validate_tree(self, root: NodeItem, raise_on_error: bool = False) -> bool:
        """validate_tree."""
        # TODO add check if heading levels conflict with hierarchy (e.g. level-1 heading with level-2 parent)
        for child_ref in root.children:
            child = child_ref.resolve(self)
            if child.parent.resolve(self) != root or not self.validate_tree(child, raise_on_error=raise_on_error):
                if raise_on_error:
                    raise ValueError(
                        f"Document hierarchy is inconsistent. {root.self_ref} has child {child.self_ref} "
                        f"with parent {child.parent.resolve(self).self_ref}"
                    )
                return False

        if isinstance(root, TableItem):
            root_children_crefs = {child.cref for child in root.children}
            for cell in root.data.table_cells:
                if isinstance(cell, RichTableCell) and (
                    (par_ref := cell.ref.resolve(self).parent) is None
                    or par_ref.resolve(self) != root
                    or cell.ref.cref not in root_children_crefs
                ):
                    if raise_on_error:
                        raise ValueError(
                            f"Document hierarchy is inconsistent. {root.self_ref} has cell {cell.ref.cref} "
                            f"with parent {cell.ref.resolve(self).parent.resolve(self).self_ref}"
                        )
                    return False

        return True

    def iterate_items(
        self,
        root: Optional[NodeItem] = None,
        with_groups: bool = False,
        traverse_pictures: bool = False,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        _level: int = 0,  # deprecated
    ) -> typing.Iterable[tuple[NodeItem, int]]:  # tuple of node and level
        """Iterate elements with level."""
        for item, stack in self._iterate_items_with_stack(
            root=root,
            with_groups=with_groups,
            traverse_pictures=traverse_pictures,
            page_nrs={page_no} if page_no is not None else None,
            included_content_layers=included_content_layers,
        ):
            yield item, len(stack)

    def _iterate_items_with_stack(
        self,
        root: Optional[NodeItem] = None,
        with_groups: bool = False,
        traverse_pictures: bool = False,
        page_nrs: Optional[set[int]] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        _stack: Optional[list[int]] = None,
    ) -> typing.Iterable[tuple[NodeItem, list[int]]]:  # tuple of node and level
        """Iterate elements with stack."""
        my_layers = included_content_layers if included_content_layers is not None else DEFAULT_CONTENT_LAYERS
        my_stack: list[int] = _stack if _stack is not None else []

        if not root:
            root = self.body

        # Yield non-group items or group items when with_groups=True

        # Combine conditions to have a single yield point
        should_yield = (
            (not isinstance(root, GroupItem) or with_groups)
            and (
                not isinstance(root, DocItem)
                or (page_nrs is None or any(prov.page_no in page_nrs for prov in root.prov))
            )
            and root.content_layer in my_layers
        )

        if should_yield:
            yield root, my_stack

        my_stack.append(-1)

        allowed_pic_refs: set[str] = (
            {r.cref for r in root.captions} if (root_is_picture := isinstance(root, PictureItem)) else set()
        )

        # Traverse children
        for child_ind, child_ref in enumerate(root.children):
            child = child_ref.resolve(self)
            if (
                root_is_picture
                and not traverse_pictures
                and isinstance(child, NodeItem)
                and child.self_ref not in allowed_pic_refs
            ):
                continue
            my_stack[-1] = child_ind

            if isinstance(child, NodeItem):
                yield from self._iterate_items_with_stack(
                    child,
                    with_groups=with_groups,
                    traverse_pictures=traverse_pictures,
                    page_nrs=page_nrs,
                    _stack=my_stack,
                    included_content_layers=my_layers,
                )

        my_stack.pop()

    def _clear_picture_pil_cache(self):
        """Clear cache storage of all images."""
        for item, level in self.iterate_items(with_groups=False):
            if isinstance(item, PictureItem):
                if item.image is not None and item.image._pil is not None:
                    item.image._pil.close()

    def _list_images_on_disk(self) -> list[Path]:
        """List all images on disk."""
        result: list[Path] = []

        for item, level in self.iterate_items(with_groups=False):
            if isinstance(item, PictureItem):
                if item.image is not None:
                    if (
                        isinstance(item.image.uri, AnyUrl)
                        and item.image.uri.scheme == "file"
                        and item.image.uri.path is not None
                    ):
                        local_path = Path(unquote(item.image.uri.path))
                        result.append(local_path)
                    elif isinstance(item.image.uri, Path):
                        result.append(item.image.uri)

        return result

    def _with_embedded_pictures(self) -> "DoclingDocument":
        """Document with embedded images.

        Creates a copy of this document where all pictures referenced
        through a file URI are turned into base64 embedded form.
        """
        result: DoclingDocument = copy.deepcopy(self)

        for ix, (item, level) in enumerate(result.iterate_items(with_groups=True)):
            if isinstance(item, PictureItem):
                if item.image is not None:
                    if isinstance(item.image.uri, AnyUrl) and item.image.uri.scheme == "file":
                        assert isinstance(item.image.uri.path, str)
                        tmp_image = PILImage.open(str(unquote(item.image.uri.path)))
                        item.image = ImageRef.from_pil(tmp_image, dpi=item.image.dpi)

                    elif isinstance(item.image.uri, Path):
                        tmp_image = PILImage.open(str(item.image.uri))
                        item.image = ImageRef.from_pil(tmp_image, dpi=item.image.dpi)

        return result

    @staticmethod
    def _save_image_and_resolve_uri(
        img: PILImage.Image,
        loc_path: Path,
        reference_path: Optional[Path],
    ) -> Union[AnyUrl, Path]:
        """Save *img* to *loc_path* and return the URI to store on the ImageRef.

        Uses a BytesIO intermediate buffer so that the write is compatible with
        UPath remote backends (PIL cannot write directly to non-local paths).
        For remote paths the URI is returned as an absolute AnyUrl string; for
        local paths it is returned relative to *reference_path* when available,
        or as the absolute *loc_path* when *reference_path* is None.

        Args:
            img: The PIL image to save.
            loc_path: Destination path (local or remote UPath).
            reference_path: Base path used to compute a relative URI for local
                storage. Pass ``None`` to store the absolute path instead.

        Returns:
            An AnyUrl for remote paths, or a Path (relative or absolute) for local paths.
        """
        buf = BytesIO()
        img.save(buf, format="PNG")
        loc_path.write_bytes(buf.getvalue())

        if is_remote_path(loc_path) or is_remote_path(reference_path):
            return AnyUrl(str(loc_path))
        if reference_path is not None:
            return relative_path(reference_path.resolve(), loc_path.resolve())
        return loc_path

    def _with_pictures_refs(
        self,
        image_dir: Path,
        page_no: Optional[int],
        reference_path: Optional[Path] = None,
        include_page_images: bool = False,
    ) -> "DoclingDocument":
        """Document with images as refs.

        Creates a copy of this document where all picture data is
        saved to image_dir and referenced through file URIs.

        When ``include_page_images`` is True, the page images are saved to
        image_dir and referenced through file URIs as well (instead of being
        kept as embedded base64 blobs).
        """
        result: DoclingDocument = copy.deepcopy(self)

        image_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        for item, _ in result.iterate_items(page_no=page_no, with_groups=False):
            if isinstance(item, PictureItem):
                img = item.get_image(doc=self)
                if img is not None:
                    hexhash = PictureItem._image_to_hexhash(img)
                    if hexhash is not None:
                        loc_path = image_dir / f"image_{img_count:06}_{hexhash}.png"
                        obj_path = self._save_image_and_resolve_uri(img, loc_path, reference_path)
                        if item.image is None:
                            scale = img.size[0] / item.prov[0].bbox.width
                            item.image = ImageRef.from_pil(image=img, dpi=round(72 * scale))
                        item.image.uri = obj_path  # type: ignore[assignment]
                img_count += 1

        if include_page_images:
            for p_no, page in result.pages.items():
                if page_no is not None and p_no != page_no:
                    continue
                if page.image is None:
                    continue
                img = page.image.pil_image
                if img is None:
                    continue
                hexhash = PictureItem._image_to_hexhash(img)
                if hexhash is None:
                    continue
                loc_path = image_dir / f"page_{p_no:06}_{hexhash}.png"
                page.image.uri = self._save_image_and_resolve_uri(img, loc_path, reference_path)  # type: ignore[assignment]

        return result

    def print_element_tree(self):
        """Print_element_tree."""
        for ix, (item, level) in enumerate(
            self.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            )
        ):
            if isinstance(item, GroupItem):
                print(
                    " " * level,
                    f"{ix}: {item.label.value} with name={item.name}",
                )
            elif isinstance(item, TextItem):
                print(
                    " " * level,
                    f"{ix}: {item.label.value}: {item.text[: min(len(item.text), 100)]}",
                )

            elif isinstance(item, DocItem):
                print(" " * level, f"{ix}: {item.label.value}")

    def export_to_element_tree(self) -> str:
        """Export_to_element_tree."""
        texts = []
        for ix, (item, level) in enumerate(
            self.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            )
        ):
            if isinstance(item, GroupItem):
                texts.append(" " * level + f"{ix}: {item.label.value} with name={item.name}")
            elif isinstance(item, TextItem):
                texts.append(" " * level + f"{ix}: {item.label.value}: {item.text[: min(len(item.text), 100)]}")
            elif isinstance(item, DocItem):
                texts.append(" " * level + f"{ix}: {item.label.value}")

        return "\n".join(texts)

    def save_as_json(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        indent: int = 2,
        coord_precision: Optional[int] = None,
        confid_precision: Optional[int] = None,
    ):
        """Save as json."""
        if isinstance(filename, str):
            filename = Path(filename)
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            artifacts_dir.mkdir(parents=True, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir,
            image_mode,
            page_no=None,
            reference_path=reference_path,
            include_page_images=True,
        )

        out = new_doc.export_to_dict(coord_precision=coord_precision, confid_precision=confid_precision)
        filename.write_text(json.dumps(out, indent=indent), encoding="utf-8")

    @classmethod
    def load_from_json(cls, filename: Union[str, Path]) -> "DoclingDocument":
        """load_from_json.

        :param filename: The filename to load a saved DoclingDocument from a .json.
        :type filename: Path

        :returns: The loaded DoclingDocument.
        :rtype: DoclingDocument

        """
        if isinstance(filename, str):
            filename = Path(filename)
        return cls.model_validate_json(filename.read_text(encoding="utf-8"))

    def save_as_yaml(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        default_flow_style: bool = False,
        coord_precision: Optional[int] = None,
        confid_precision: Optional[int] = None,
    ):
        """Save as yaml."""
        if isinstance(filename, str):
            filename = Path(filename)
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            artifacts_dir.mkdir(parents=True, exist_ok=True)

        new_doc = self._make_copy_with_refmode(
            artifacts_dir,
            image_mode,
            page_no=None,
            reference_path=reference_path,
            include_page_images=True,
        )

        out = new_doc.export_to_dict(coord_precision=coord_precision, confid_precision=confid_precision)
        stream = StringIO()
        yaml.dump(out, stream, default_flow_style=default_flow_style)
        filename.write_text(stream.getvalue(), encoding="utf-8")

    @classmethod
    def load_from_yaml(cls, filename: Union[str, Path]) -> "DoclingDocument":
        """load_from_yaml.

        Args:
            filename: The filename to load a YAML-serialized DoclingDocument from.

        Returns:
            DoclingDocument: the loaded DoclingDocument
        """
        if isinstance(filename, str):
            filename = Path(filename)
        data = yaml.load(filename.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
        return DoclingDocument.model_validate(data)

    def export_to_dict(
        self,
        mode: str = "json",
        by_alias: bool = True,
        exclude_none: bool = True,
        coord_precision: Optional[int] = None,
        confid_precision: Optional[int] = None,
    ) -> dict[str, Any]:
        """Export to dict."""
        context = {}
        if coord_precision is not None:
            context[PydanticSerCtxKey.COORD_PREC.value] = coord_precision
        if confid_precision is not None:
            context[PydanticSerCtxKey.CONFID_PREC.value] = confid_precision
        out = self.model_dump(mode=mode, by_alias=by_alias, exclude_none=exclude_none, context=context)

        return out

    def save_as_markdown(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        strict_text: bool = False,
        escape_html: bool = True,
        escaping_underscores: bool = True,
        image_placeholder: str = "<!-- image -->",
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        page_break_placeholder: Optional[str] = None,
        include_annotations: bool = True,
        compact_tables: bool = False,
        *,
        mark_meta: bool = False,
        use_legacy_annotations: Optional[bool] = None,  # deprecated
    ):
        """Save to markdown."""
        if isinstance(filename, str):
            filename = Path(filename)
        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            artifacts_dir.mkdir(parents=True, exist_ok=True)

        new_doc = self._make_copy_with_refmode(artifacts_dir, image_mode, page_no, reference_path=reference_path)

        md_out = new_doc.export_to_markdown(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            strict_text=strict_text,
            escape_html=escape_html,
            escape_underscores=escaping_underscores,
            image_placeholder=image_placeholder,
            image_mode=image_mode,
            indent=indent,
            text_width=text_width,
            page_no=page_no,
            included_content_layers=included_content_layers,
            page_break_placeholder=page_break_placeholder,
            include_annotations=include_annotations,
            compact_tables=compact_tables,
            use_legacy_annotations=use_legacy_annotations,
            mark_meta=mark_meta,
        )

        filename.write_text(md_out, encoding="utf-8")

    def export_to_markdown(
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        strict_text: bool = False,
        escape_html: bool = True,
        escape_underscores: bool = True,
        image_placeholder: str = "<!-- image -->",
        enable_chart_tables: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        indent: int = 4,
        text_width: int = -1,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        page_break_placeholder: Optional[str] = None,  # e.g. "<!-- page break -->",
        include_annotations: bool = True,
        mark_annotations: bool = False,
        compact_tables: bool = False,
        traverse_pictures: bool = False,
        *,
        use_legacy_annotations: Optional[bool] = None,  # deprecated
        allowed_meta_names: Optional[set[str]] = None,
        blocked_meta_names: Optional[set[str]] = None,
        mark_meta: bool = False,
    ) -> str:
        r"""Serialize to Markdown.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole document.

        :param delim: Deprecated.
        :type delim: str = "\n\n"
        :param from_element: Body slicing start index (inclusive).
                (Default value = 0).
        :type from_element: int = 0
        :param to_element: Body slicing stop index
                (exclusive). (Default value = maxint).
        :type to_element: int = sys.maxsize
        :param labels: The set of document labels to include in the export. None falls
            back to the system-defined default.
        :type labels: Optional[set[DocItemLabel]] = None
        :param strict_text: Deprecated.
        :type strict_text: bool = False
        :param escape_html: bool: Whether to escape HTML reserved characters in the
            text content of the document. (Default value = True).
        :param escape_underscores: bool: Whether to escape underscores in the
            text content of the document. (Default value = True).
        :type escape_underscores: bool = True
        :param image_placeholder: The placeholder to include to position
            images in the markdown. (Default value = "\<!-- image --\>").
        :type image_placeholder: str = "<!-- image -->"
        :param image_mode: The mode to use for including images in the
            markdown. (Default value = ImageRefMode.PLACEHOLDER).
        :type image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
        :param indent: The indent in spaces of the nested lists.
            (Default value = 4).
        :type indent: int = 4
        :param included_content_layers: The set of layels to include in the export. None
            falls back to the system-defined default.
        :type included_content_layers: Optional[set[ContentLayer]] = None
        :param page_break_placeholder: The placeholder to include for marking page
            breaks. None means no page break placeholder will be used.
        :type page_break_placeholder: Optional[str] = None
        :param include_annotations: bool: Whether to include annotations in the export; only considered if item does not
            have meta. (Default value = True).
        :type include_annotations: bool = True
        :param mark_annotations: bool: Whether to mark annotations in the export; only considered if item does not have
            meta. (Default value = False).
        :type mark_annotations: bool = False
        :param compact_tables: bool: Whether to use compact table format without column padding. (Default value = False).
        :type compact_tables: bool = False
        :param traverse_pictures: bool: Whether to traverse into picture items and
            serialize their text children. Must be set to True for scanned/image-based
            PDFs processed with full-page OCR, where the layout model places all OCR
            text as children of a top-level PictureItem. (Default value = False).
        :type traverse_pictures: bool = False
        :param use_legacy_annotations: bool: Deprecated; legacy annotations considered only when meta not present.
        :type use_legacy_annotations: Optional[bool] = None
        :param mark_meta: bool: Whether to mark meta in the export
        :type mark_meta: bool = False
        :returns: The exported Markdown representation.
        :rtype: str
        :param allowed_meta_names: Optional[set[str]]: Meta names to allow; None means all meta names are allowed.
        :type allowed_meta_names: Optional[set[str]] = None
        :param blocked_meta_names: Optional[set[str]]: Meta names to block; takes precedence over allowed_meta_names.
        :type blocked_meta_names: Optional[set[str]] = None
        """
        from docling_core.transforms.serializer.markdown import (
            MarkdownDocSerializer,
            MarkdownParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        my_layers = included_content_layers if included_content_layers is not None else DEFAULT_CONTENT_LAYERS

        if use_legacy_annotations is not None:
            warnings.warn(
                "Parameter `use_legacy_annotations` has been deprecated and will be ignored.",
                DeprecationWarning,
            )

        serializer = MarkdownDocSerializer(
            doc=self,
            params=MarkdownParams(
                labels=my_labels,
                layers=my_layers,
                pages={page_no} if page_no is not None else None,
                start_idx=from_element,
                stop_idx=to_element,
                escape_html=escape_html,
                escape_underscores=escape_underscores,
                image_placeholder=image_placeholder,
                enable_chart_tables=enable_chart_tables,
                image_mode=image_mode,
                indent=indent,
                wrap_width=text_width if text_width > 0 else None,
                page_break_placeholder=page_break_placeholder,
                mark_meta=mark_meta,
                include_annotations=include_annotations,
                allowed_meta_names=allowed_meta_names,
                blocked_meta_names=blocked_meta_names or set(),
                mark_annotations=mark_annotations,
                compact_tables=compact_tables,
                traverse_pictures=traverse_pictures,
            ),
        )
        ser_res = serializer.serialize()

        if delim != "\n\n":
            _logger.warning(
                "Parameter `delim` has been deprecated and will be ignored.",
            )
        if strict_text:
            _logger.warning(
                "Parameter `strict_text` has been deprecated and will be ignored.",
            )

        return ser_res.text

    def export_to_text(
        self,
        delim: str = "\n\n",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        page_no: Optional[int] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
        page_break_placeholder: Optional[str] = None,
        traverse_pictures: bool = False,
    ) -> str:
        r"""Export to plain text.

        Produces clean plain text without any Markdown decoration. Heading
        markers (``#``), bold/italic markers, and hyperlink syntax are all
        stripped. List bullets (``-``), ordered list numbers, and table-cell
        separators (``|``) are preserved as they aid readability.

        :param delim: Deprecated.
        :type delim: str = "\\n\\n"
        :param from_element: Body slicing start index (inclusive). (Default value = 0).
        :type from_element: int = 0
        :param to_element: Body slicing stop index (exclusive). (Default value = maxint).
        :type to_element: int = sys.maxsize
        :param labels: The set of document labels to include in the export. None falls
            back to the system-defined default.
        :type labels: Optional[set[DocItemLabel]] = None
        :param page_no: If set, only content from this page is exported.
        :type page_no: Optional[int] = None
        :param included_content_layers: The set of layers to include. None falls back
            to the system-defined default.
        :type included_content_layers: Optional[set[ContentLayer]] = None
        :param page_break_placeholder: String inserted at page boundaries. None means
            no page-break marker is emitted.
        :type page_break_placeholder: Optional[str] = None
        :param traverse_pictures: bool: Whether to traverse into picture items and
            include their text children. Must be set to True for scanned/image-based
            PDFs processed with full-page OCR, where the layout model places all OCR
            text as children of a top-level PictureItem. (Default value = False).
        :type traverse_pictures: bool = False
        :returns: The exported plain-text representation.
        :rtype: str
        """
        from docling_core.transforms.serializer.plain_text import (
            PlainTextDocSerializer,
            PlainTextParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        my_layers = included_content_layers if included_content_layers is not None else DEFAULT_CONTENT_LAYERS

        if delim != "\n\n":
            _logger.warning(
                "Parameter `delim` has been deprecated and will be ignored.",
            )

        serializer = PlainTextDocSerializer(
            doc=self,
            params=PlainTextParams(
                labels=my_labels,
                layers=my_layers,
                pages={page_no} if page_no is not None else None,
                start_idx=from_element,
                stop_idx=to_element,
                page_break_placeholder=page_break_placeholder,
                traverse_pictures=traverse_pictures,
            ),
        )
        return serializer.serialize().text

    def save_as_html(
        self,
        filename: Union[str, Path],
        artifacts_dir: Optional[Path] = None,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        formula_to_mathml: bool = True,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = "null",  # should be deprecated
        included_content_layers: Optional[set[ContentLayer]] = None,
        split_page_view: bool = False,
        include_annotations: bool = True,
    ):
        """Save to HTML."""
        if isinstance(filename, str):
            filename = Path(filename)

        artifacts_dir, reference_path = self._get_output_paths(filename, artifacts_dir)

        if image_mode == ImageRefMode.REFERENCED:
            artifacts_dir.mkdir(parents=True, exist_ok=True)

        new_doc = self._make_copy_with_refmode(artifacts_dir, image_mode, page_no, reference_path=reference_path)

        html_out = new_doc.export_to_html(
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            image_mode=image_mode,
            formula_to_mathml=formula_to_mathml,
            page_no=page_no,
            html_lang=html_lang,
            html_head=html_head,
            included_content_layers=included_content_layers,
            split_page_view=split_page_view,
            include_annotations=include_annotations,
        )

        filename.write_text(html_out, encoding="utf-8")

    def _get_output_paths(
        self,
        filename: Path,
        artifacts_dir: Optional[Path] = None,
    ) -> tuple[Path, Optional[Path]]:
        """Resolve output and artifacts directory paths from the given filename.

        Both ``Path`` and ``UPath`` objects are accepted since ``UPath`` is ``Path``-compatible for local paths, and remote ``UPath`` objects implement the same interface used here (``with_suffix``, ``with_name``,
        ``is_absolute``, ``parent``, ``/``).

        Args:
            filename: Destination file path.
            artifacts_dir: Optional explicit artifacts directory. When ``None``, a sibling
                directory named ``<stem>_artifacts`` is derived from ``filename``.

        Returns:
            A tuple of ``(artifacts_dir, reference_path)`` where ``reference_path`` is
            the parent of ``filename`` when ``artifacts_dir`` is relative, or ``None``
            when it is absolute.
        """
        if artifacts_dir is None:
            # Remove the extension and add '_artifacts'
            artifacts_dir = filename.with_suffix("")
            artifacts_dir = artifacts_dir.with_name(artifacts_dir.name + "_artifacts")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = filename.parent
            artifacts_dir = reference_path / artifacts_dir

        return artifacts_dir, reference_path

    def _make_copy_with_refmode(
        self,
        artifacts_dir: Path,
        image_mode: ImageRefMode,
        page_no: Optional[int],
        reference_path: Optional[Path] = None,
        include_page_images: bool = False,
    ):
        new_doc = None
        if image_mode == ImageRefMode.PLACEHOLDER:
            new_doc = self
        elif image_mode == ImageRefMode.REFERENCED:
            new_doc = self._with_pictures_refs(
                image_dir=artifacts_dir,
                page_no=page_no,
                reference_path=reference_path,
                include_page_images=include_page_images,
            )
        elif image_mode == ImageRefMode.EMBEDDED:
            new_doc = self._with_embedded_pictures()
        else:
            raise ValueError("Unsupported ImageRefMode")
        return new_doc

    def export_to_html(
        self,
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        enable_chart_tables: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        formula_to_mathml: bool = True,
        page_no: Optional[int] = None,
        html_lang: str = "en",
        html_head: str = "null",  # should be deprecated ...
        included_content_layers: Optional[set[ContentLayer]] = None,
        split_page_view: bool = False,
        include_annotations: bool = True,
    ) -> str:
        r"""Serialize to HTML."""
        from docling_core.transforms.serializer.html import (
            HTMLDocSerializer,
            HTMLOutputStyle,
            HTMLParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        my_layers = included_content_layers if included_content_layers is not None else DEFAULT_CONTENT_LAYERS

        output_style = HTMLOutputStyle.SINGLE_COLUMN
        if split_page_view:
            output_style = HTMLOutputStyle.SPLIT_PAGE

        params = HTMLParams(
            labels=my_labels,
            layers=my_layers,
            pages={page_no} if page_no is not None else None,
            start_idx=from_element,
            stop_idx=to_element,
            image_mode=image_mode,
            enable_chart_tables=enable_chart_tables,
            formula_to_mathml=formula_to_mathml,
            html_head=html_head,
            html_lang=html_lang,
            output_style=output_style,
            include_annotations=include_annotations,
        )

        if html_head == "null":
            params.html_head = None

        serializer = HTMLDocSerializer(
            doc=self,
            params=params,
        )
        ser_res = serializer.serialize()

        return ser_res.text

    def export_to_vtt(
        self,
        included_content_layers: set[ContentLayer] | None = None,
        omit_hours_if_zero: bool = False,
        omit_voice_end: bool = False,
    ) -> str:
        """Serializes the Docling document to WebVTT format.

        Args:
            included_content_layers: The content layers to serialize. If omitted, the `DEFAULT_CONTENT_LAYERS` will
                be serialized.
            omit_hours_if_zero: If True, omit hours when they are 0 in the timings.
            omit_voice_end: If True and cue blocks have a WebVTT cue voice span as the only component, omit the voice
                end tag for brevity.

        Returns:
            A string representation of the Docling document in WebVTT format.
        """
        from docling_core.transforms.serializer.webvtt import WebVTTDocSerializer, WebVTTParams

        my_layers = included_content_layers if included_content_layers is not None else DEFAULT_CONTENT_LAYERS

        params = WebVTTParams(layers=my_layers, omit_hours_if_zero=omit_hours_if_zero, omit_voice_end=omit_voice_end)

        serializer = WebVTTDocSerializer(
            doc=self,
            params=params,
        )
        ser_res = serializer.serialize()

        return ser_res.text

    def save_as_vtt(
        self,
        filename: str | Path,
        included_content_layers: set[ContentLayer] | None = None,
        omit_hours_if_zero: bool = False,
        omit_voice_end: bool = True,
    ) -> None:
        """Saves the Docling document to a file in WebVTT format.

        Args:
            filename: The path to the WebVTT file.
            included_content_layers: The content layers to serialize. If omitted, the `DEFAULT_CONTENT_LAYERS` will
                be serialized.
            omit_hours_if_zero: If True, omit hours when they are 0 in the timings.
            omit_voice_end: If True and cue blocks have a WebVTT cue voice span as the only component, omit the voice
                end tag for brevity.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        vtt_out = self.export_to_vtt(
            included_content_layers=included_content_layers,
            omit_hours_if_zero=omit_hours_if_zero,
            omit_voice_end=omit_voice_end,
        )

        filename.write_text(vtt_out, encoding="utf-8")

    @staticmethod
    def load_from_doctags(  # noqa: C901
        doctag_document: DocTagsDocument, document_name: str = "Document"
    ) -> "DoclingDocument":
        r"""Load Docling document from lists of DocTags and Images."""
        # Maps the recognized tag to a Docling label.
        # Code items will be given DocItemLabel.CODE
        tag_to_doclabel = {
            "title": DocItemLabel.TITLE,
            "document_index": DocItemLabel.DOCUMENT_INDEX,
            "otsl": DocItemLabel.TABLE,
            "section_header_level_1": DocItemLabel.SECTION_HEADER,
            "section_header_level_2": DocItemLabel.SECTION_HEADER,
            "section_header_level_3": DocItemLabel.SECTION_HEADER,
            "section_header_level_4": DocItemLabel.SECTION_HEADER,
            "section_header_level_5": DocItemLabel.SECTION_HEADER,
            "section_header_level_6": DocItemLabel.SECTION_HEADER,
            "checkbox_selected": DocItemLabel.CHECKBOX_SELECTED,
            "checkbox_unselected": DocItemLabel.CHECKBOX_UNSELECTED,
            "text": DocItemLabel.TEXT,
            "page_header": DocItemLabel.PAGE_HEADER,
            "page_footer": DocItemLabel.PAGE_FOOTER,
            "formula": DocItemLabel.FORMULA,
            "caption": DocItemLabel.CAPTION,
            "picture": DocItemLabel.PICTURE,
            "list_item": DocItemLabel.LIST_ITEM,
            "footnote": DocItemLabel.FOOTNOTE,
            "code": DocItemLabel.CODE,
            "key_value_region": DocItemLabel.KEY_VALUE_REGION,
            "key_value_map": DocItemLabel.FIELD_REGION,
        }

        doc = DoclingDocument(name=document_name)

        def extract_bounding_box(text_chunk: str) -> Optional[BoundingBox]:
            """Extract <loc_...> coords from the chunk, normalized by / 500."""
            coords = re.findall(r"<loc_(\d+)>", text_chunk)
            if len(coords) > 4:
                coords = coords[:4]
            if len(coords) == 4:
                l, t, r, b = map(float, coords)
                return BoundingBox(l=l / 500, t=t / 500, r=r / 500, b=b / 500)
            return None

        def extract_inner_text(text_chunk: str) -> str:
            """Strip all <...> tags (except <_..._>) to get the raw text content."""
            # The tag name must start with a letter or "/", and the match must
            # not cross a ">" boundary. This avoids deleting spans of plain text
            # that merely contain "<" followed by a later ">" (e.g. statistical
            # notation like "P < 0.05 ... P > 0.05"). See #618.
            return re.sub(r"<(?!_.*?_>)[a-zA-Z/][^>]*>", "", text_chunk).strip()

        def extract_caption(
            text_chunk: str,
        ) -> tuple[Optional[TextItem], Optional[BoundingBox]]:
            """Extract caption text from the chunk."""
            caption = re.search(r"<caption>(.*?)</caption>", text_chunk)
            if caption is not None:
                caption_content = caption.group(1)
                bbox = extract_bounding_box(caption_content)
                caption_text = extract_inner_text(caption_content)
                caption_item = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=caption_text,
                    parent=None,
                )
            else:
                caption_item = None
                bbox = None
            return caption_item, bbox

        def extract_picture_classification(text_chunk: str):
            """Extract any picture classification label from the chunk."""
            label = None

            # Current v2 model labels (DocumentFigureClassifier-v2.0)
            all_labels: list[PictureClassificationLabel | str] = [
                PictureClassificationLabel.LOGO,
                PictureClassificationLabel.PHOTOGRAPH,
                PictureClassificationLabel.ICON,
                PictureClassificationLabel.ENGINEERING_DRAWING,
                PictureClassificationLabel.LINE_CHART,
                PictureClassificationLabel.BAR_CHART,
                PictureClassificationLabel.OTHER,
                PictureClassificationLabel.TABLE,
                PictureClassificationLabel.FLOW_CHART,
                PictureClassificationLabel.SCREENSHOT_FROM_COMPUTER,
                PictureClassificationLabel.SIGNATURE,
                PictureClassificationLabel.SCREENSHOT_FROM_MANUAL,
                PictureClassificationLabel.GEOGRAPHICAL_MAP,
                PictureClassificationLabel.PIE_CHART,
                PictureClassificationLabel.PAGE_THUMBNAIL,
                PictureClassificationLabel.STAMP,
                PictureClassificationLabel.MUSIC,
                PictureClassificationLabel.CALENDAR,
                PictureClassificationLabel.QR_CODE,
                PictureClassificationLabel.BAR_CODE,
                PictureClassificationLabel.FULL_PAGE_IMAGE,
                PictureClassificationLabel.SCATTER_PLOT,
                PictureClassificationLabel.CHEMISTRY_STRUCTURE,
                PictureClassificationLabel.TOPOGRAPHICAL_MAP,
                PictureClassificationLabel.CROSSWORD_PUZZLE,
                PictureClassificationLabel.BOX_PLOT,
            ]

            # Legacy v1 model labels
            all_labels.extend(
                [
                    PictureClassificationLabel.STACKED_BAR_CHART,
                    PictureClassificationLabel.SCATTER_CHART,
                    PictureClassificationLabel.HEATMAP,
                    PictureClassificationLabel.NATURAL_IMAGE,
                    PictureClassificationLabel.REMOTE_SENSING,
                    PictureClassificationLabel.SCREENSHOT,
                    PictureClassificationLabel.MOLECULAR_STRUCTURE,
                    PictureClassificationLabel.MARKUSH_STRUCTURE,
                    PictureClassificationLabel.PICTURE_GROUP,
                ]
            )

            # Legacy SmolDocling labels
            all_labels.extend(
                [
                    "line",
                    "dot_line",
                    "vbar_categorical",
                    "hbar_categorical",
                ]
            )

            # Mapping for legacy labels
            label_mapping = {
                "line": PictureClassificationLabel.LINE_CHART,
                "dot_line": PictureClassificationLabel.LINE_CHART,
                "vbar_categorical": PictureClassificationLabel.BAR_CHART,
                "hbar_categorical": PictureClassificationLabel.BAR_CHART,
            }

            for clabel in all_labels:
                tag = f"<{clabel}>"
                if tag in text_chunk:
                    if clabel in label_mapping:
                        label = PictureClassificationLabel(label_mapping[clabel])
                    else:
                        label = PictureClassificationLabel(clabel)
                    break
            return label

        def parse_key_value_item(
            tokens: str, image: Optional[PILImage.Image] = None
        ) -> tuple[GraphData, Optional[ProvenanceItem]]:
            if image is not None:
                pg_width = image.width
                pg_height = image.height
            else:
                pg_width = 1
                pg_height = 1

            start_locs_match = re.search(r"<key_value_region>(.*?)<key", tokens)
            if start_locs_match:
                overall_locs = start_locs_match.group(1)
                overall_bbox = extract_bounding_box(overall_locs) if image else None
                overall_prov = (
                    ProvenanceItem(
                        bbox=overall_bbox.resize_by_scale(pg_width, pg_height),
                        charspan=(0, 0),
                        page_no=1,
                    )
                    if overall_bbox
                    else None
                )
            else:
                overall_prov = None

            # here we assumed the labels as only key or value, later on we can update
            # it to have unspecified, checkbox etc.
            cell_pattern = re.compile(
                r"<(?P<label>key|value)_(?P<id>\d+)>"
                r"(?P<content>.*?)"
                r"</(?P=label)_(?P=id)>",
                re.DOTALL,
            )

            cells: list[GraphCell] = []
            links: list[GraphLink] = []
            raw_link_predictions = []

            for cell_match in cell_pattern.finditer(tokens):
                cell_label_str = cell_match.group("label")  # "key" or "value"
                cell_id = int(cell_match.group("id"))
                raw_content = cell_match.group("content")

                # link tokens
                link_matches = re.findall(r"<link_(\d+)>", raw_content)

                cell_bbox = extract_bounding_box(raw_content) if image else None
                cell_prov = None
                if cell_bbox is not None:
                    cell_prov = ProvenanceItem(
                        bbox=cell_bbox.resize_by_scale(pg_width, pg_height),
                        charspan=(0, 0),
                        page_no=1,
                    )

                cleaned_text = re.sub(r"<loc_\d+>", "", raw_content)
                cleaned_text = re.sub(r"<link_\d+>", "", cleaned_text).strip()

                cell_obj = GraphCell(
                    label=GraphCellLabel(cell_label_str),
                    cell_id=cell_id,
                    text=cleaned_text,
                    orig=cleaned_text,
                    prov=cell_prov,
                    item_ref=None,
                )
                cells.append(cell_obj)

                cell_ids = {cell.cell_id for cell in cells}

                for target_str in link_matches:
                    raw_link_predictions.append((cell_id, int(target_str)))

            cell_ids = {cell.cell_id for cell in cells}

            for source_id, target_id in raw_link_predictions:
                # basic check to validate the prediction
                if target_id not in cell_ids:
                    continue
                link_obj = GraphLink(
                    label=GraphLinkLabel.TO_VALUE,
                    source_cell_id=source_id,
                    target_cell_id=target_id,
                )
                links.append(link_obj)

            return (GraphData(cells=cells, links=links), overall_prov)

        def _add_text(
            full_chunk: str,
            bbox: Optional[BoundingBox],
            pg_width: int,
            pg_height: int,
            page_no: int,
            tag_name: str,
            doc_label: DocItemLabel,
            doc: DoclingDocument,
            parent: Optional[NodeItem],
        ):
            # For everything else, treat as text
            text_content = extract_inner_text(full_chunk)
            element_prov = (
                ProvenanceItem(
                    bbox=bbox.resize_by_scale(pg_width, pg_height),
                    charspan=(0, len(text_content)),
                    page_no=page_no,
                )
                if bbox
                else None
            )

            content_layer = ContentLayer.BODY
            if tag_name in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                content_layer = ContentLayer.FURNITURE

            if doc_label == DocItemLabel.SECTION_HEADER:
                # Extract level from tag_name (e.g. "section_level_header_1" -> 1)
                level = int(tag_name.split("_")[-1])
                doc.add_heading(
                    text=text_content,
                    level=level,
                    prov=element_prov,
                    parent=parent,
                    content_layer=content_layer,
                )
            elif doc_label == DocItemLabel.CODE:
                code_language = CodeLanguageLabel.UNKNOWN
                m = re.match(r"^<_([^>]+)_>", text_content)
                if m:
                    text_content = text_content[m.end() :]
                    try:
                        code_language = CodeLanguageLabel(m.group(1))
                    except ValueError:
                        pass
                doc.add_code(
                    text=text_content,
                    code_language=code_language,
                    prov=element_prov,
                    parent=parent,
                    content_layer=content_layer,
                )

            else:
                doc.add_text(
                    label=doc_label,
                    text=text_content,
                    prov=element_prov,
                    parent=parent,
                    content_layer=content_layer,
                )

        # doc = DoclingDocument(name="Document")
        for pg_idx, doctag_page in enumerate(doctag_document.pages):
            page_doctags = doctag_page.tokens
            image = doctag_page.image

            page_no = pg_idx + 1
            # bounding_boxes = []

            if image is not None:
                pg_width = image.width
                pg_height = image.height
            else:
                pg_width = 1
                pg_height = 1

            doc.add_page(
                page_no=page_no,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=image, dpi=72) if image else None,
            )

            """
            1. Finds all <tag>...</tag>
               blocks in the entire string (multi-line friendly)
               in the order they appear.
            2. For each chunk, extracts bounding box (if any) and inner text.
            3. Adds the item to a DoclingDocument structure with the right label.
            4. Tracks bounding boxes+color in a separate list for later visualization.
            """

            # Regex for root level recognized tags
            tag_pattern = (
                rf"<(?P<tag>{DocItemLabel.TITLE}|{DocItemLabel.DOCUMENT_INDEX}|"
                rf"{DocItemLabel.CHECKBOX_UNSELECTED}|{DocItemLabel.CHECKBOX_SELECTED}|"
                rf"{DocItemLabel.TEXT}|{DocItemLabel.PAGE_HEADER}|{GroupLabel.INLINE}|"
                rf"{DocItemLabel.PAGE_FOOTER}|{DocItemLabel.FORMULA}|"
                rf"{DocItemLabel.CAPTION}|{DocItemLabel.PICTURE}|"
                rf"{DocItemLabel.FOOTNOTE}|{DocItemLabel.CODE}|"
                rf"{DocItemLabel.SECTION_HEADER}_level_[1-6]|"
                rf"{DocumentToken.ORDERED_LIST.value}|"
                rf"{DocumentToken.UNORDERED_LIST.value}|"
                rf"{DocItemLabel.KEY_VALUE_REGION}|"
                rf"{DocumentToken.CHART.value}|"
                rf"{DocumentToken.OTSL.value})>"
                rf"(?P<content>.*?)"
                rf"(?:(?P<closed></(?P=tag)>)|(?P<eof>$))"
            )
            pattern = re.compile(tag_pattern, re.DOTALL)

            # Go through each match in order
            for match in pattern.finditer(page_doctags):
                full_chunk = match.group(0)
                tag_name = match.group("tag")

                bbox = extract_bounding_box(full_chunk)  # Extracts first bbox
                if not match.group("closed"):
                    # no closing tag; only the existence of the item is recovered
                    full_chunk = f"<{tag_name}></{tag_name}>"

                doc_label = tag_to_doclabel.get(tag_name, DocItemLabel.TEXT)

                if tag_name == DocumentToken.OTSL.value:
                    table_data = parse_otsl_table_content(full_chunk)
                    caption, caption_bbox = extract_caption(full_chunk)
                    if caption is not None and caption_bbox is not None:
                        caption.prov.append(
                            ProvenanceItem(
                                bbox=caption_bbox.resize_by_scale(pg_width, pg_height),
                                charspan=(0, len(caption.text)),
                                page_no=page_no,
                            )
                        )
                    if bbox:
                        prov = ProvenanceItem(
                            bbox=bbox.resize_by_scale(pg_width, pg_height),
                            charspan=(0, 0),
                            page_no=page_no,
                        )
                        doc.add_table(data=table_data, prov=prov, caption=caption)
                    else:
                        doc.add_table(data=table_data, caption=caption)

                elif tag_name == GroupLabel.INLINE:
                    inline_group = doc.add_inline_group()
                    content = match.group("content")
                    common_bbox = extract_bounding_box(content)
                    for item_match in pattern.finditer(content):
                        item_tag = item_match.group("tag")
                        _add_text(
                            full_chunk=item_match.group(0),
                            bbox=common_bbox,
                            pg_width=pg_width,
                            pg_height=pg_height,
                            page_no=page_no,
                            tag_name=item_tag,
                            doc_label=tag_to_doclabel.get(item_tag, DocItemLabel.TEXT),
                            doc=doc,
                            parent=inline_group,
                        )

                elif tag_name in [DocItemLabel.PICTURE, DocItemLabel.CHART]:
                    caption, caption_bbox = extract_caption(full_chunk)
                    table_data = None
                    image_classification = extract_picture_classification(full_chunk)
                    if tag_name == DocumentToken.CHART.value:
                        table_data = parse_otsl_table_content(full_chunk)
                    pic_title = image_classification if image_classification is not None else "other"
                    if image:
                        if bbox:
                            im_width, im_height = image.size

                            crop_box = (
                                int(bbox.l * im_width),
                                int(bbox.t * im_height),
                                int(bbox.r * im_width),
                                int(bbox.b * im_height),
                            )
                            cropped_image = image.crop(crop_box)
                            pic = doc.add_picture(
                                parent=None,
                                image=ImageRef.from_pil(image=cropped_image, dpi=72),
                                prov=(
                                    ProvenanceItem(
                                        bbox=bbox.resize_by_scale(pg_width, pg_height),
                                        charspan=(0, 0),
                                        page_no=page_no,
                                    )
                                ),
                            )
                            # If there is a caption to an image, add it as well
                            if caption is not None and caption_bbox is not None:
                                caption.prov.append(
                                    ProvenanceItem(
                                        bbox=caption_bbox.resize_by_scale(pg_width, pg_height),
                                        charspan=(0, len(caption.text)),
                                        page_no=page_no,
                                    )
                                )
                                pic.captions.append(caption.get_ref())

                            pic_classification = None
                            if image_classification is not None:
                                pic_classification = PictureClassificationMetaField(
                                    predictions=[
                                        PictureClassificationPrediction(
                                            class_name=image_classification,
                                            confidence=1.0,
                                            created_by="load_from_doctags",
                                        )
                                    ]
                                )
                            pic_tabular_chart = None
                            if table_data is not None:
                                pic_tabular_chart = TabularChartMetaField(
                                    title=pic_title,
                                    chart_data=table_data,
                                )

                            if pic_classification is not None or pic_tabular_chart is not None:
                                pic.meta = PictureMeta(
                                    classification=pic_classification,
                                    tabular_chart=pic_tabular_chart,
                                )
                    else:
                        if bbox:
                            # In case we don't have access to an binary of an image
                            pic = doc.add_picture(
                                parent=None,
                                prov=ProvenanceItem(bbox=bbox, charspan=(0, 0), page_no=page_no),
                            )
                            # If there is a caption to an image, add it as well
                            if caption is not None and caption_bbox is not None:
                                caption.prov.append(
                                    ProvenanceItem(
                                        bbox=caption_bbox.resize_by_scale(pg_width, pg_height),
                                        charspan=(0, len(caption.text)),
                                        page_no=page_no,
                                    )
                                )
                                pic.captions.append(caption.get_ref())
                            if image_classification is not None:
                                if pic.meta is None:
                                    pic.meta = PictureMeta()
                                pic.meta.classification = PictureClassificationMetaField(
                                    predictions=[
                                        PictureClassificationPrediction(
                                            class_name=image_classification,
                                            confidence=1.0,
                                            created_by="load_from_doctags",
                                        )
                                    ]
                                )
                                # legacy annotations:
                                pic.annotations.append(
                                    PictureClassificationData(
                                        provenance="load_from_doctags",
                                        predicted_classes=[
                                            # chart_type
                                            PictureClassificationClass(class_name=image_classification, confidence=1.0)
                                        ],
                                    )
                                )
                            if table_data is not None:
                                if pic.meta is None:
                                    pic.meta = PictureMeta()
                                pic.meta.tabular_chart = TabularChartMetaField(chart_data=table_data, title=pic_title)
                                # legacy annotations:
                                # Add chart data as PictureTabularChartData
                                pd = PictureTabularChartData(chart_data=table_data, title=pic_title)
                                pic.annotations.append(pd)

                elif tag_name == DocItemLabel.KEY_VALUE_REGION:
                    key_value_data, kv_item_prov = parse_key_value_item(full_chunk, image)
                    doc.add_key_values(graph=key_value_data, prov=kv_item_prov)
                elif tag_name in [
                    DocumentToken.ORDERED_LIST.value,
                    DocumentToken.UNORDERED_LIST.value,
                ]:
                    GroupLabel.LIST
                    enum_marker = ""
                    enum_value = 0
                    if tag_name == DocumentToken.ORDERED_LIST.value:
                        GroupLabel.ORDERED_LIST

                    list_item_pattern = rf"<(?P<tag>{DocItemLabel.LIST_ITEM})>.*?</(?P=tag)>"
                    li_pattern = re.compile(list_item_pattern, re.DOTALL)
                    # Add list group:
                    new_list = doc.add_list_group(name="list")
                    # Pricess list items
                    for li_match in li_pattern.finditer(full_chunk):
                        enum_value += 1
                        if tag_name == DocumentToken.ORDERED_LIST.value:
                            enum_marker = str(enum_value) + "."

                        li_full_chunk = li_match.group(0)
                        li_bbox = extract_bounding_box(li_full_chunk) if image else None
                        text_content = extract_inner_text(li_full_chunk)
                        # Add list item
                        doc.add_list_item(
                            marker=enum_marker,
                            enumerated=(tag_name == DocumentToken.ORDERED_LIST.value),
                            parent=new_list,
                            text=text_content,
                            prov=(
                                ProvenanceItem(
                                    bbox=li_bbox.resize_by_scale(pg_width, pg_height),
                                    charspan=(0, len(text_content)),
                                    page_no=page_no,
                                )
                                if li_bbox
                                else None
                            ),
                        )
                else:
                    # For everything else, treat as text
                    _add_text(
                        full_chunk=full_chunk,
                        bbox=bbox,
                        pg_width=pg_width,
                        pg_height=pg_height,
                        page_no=page_no,
                        tag_name=tag_name,
                        doc_label=doc_label,
                        doc=doc,
                        parent=None,
                    )
        return doc

    @deprecated("Use save_as_doctags instead.")
    def save_as_document_tokens(self, *args, **kwargs):
        r"""Save the document content to a DocumentToken format."""
        return self.save_as_doctags(*args, **kwargs)

    def save_as_doctags(
        self,
        filename: Union[str, Path],
        delim: str = "",
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_text: bool = True,
        minified: bool = False,
    ):
        r"""Save the document content to DocTags format."""
        if isinstance(filename, str):
            filename = Path(filename)
        out = self.export_to_doctags(
            delim=delim,
            from_element=from_element,
            to_element=to_element,
            labels=labels,
            xsize=xsize,
            ysize=ysize,
            add_location=add_location,
            add_content=add_content,
            add_page_index=add_page_index,
            # table specific flags
            add_table_cell_location=add_table_cell_location,
            add_table_cell_text=add_table_cell_text,
            minified=minified,
        )

        filename.write_text(out, encoding="utf-8")

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        delim: str = "",  # deprecated
        from_element: int = 0,
        to_element: int = sys.maxsize,
        labels: Optional[set[DocItemLabel]] = None,
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
        add_page_index: bool = True,
        # table specific flags
        add_table_cell_location: bool = False,
        add_table_cell_text: bool = True,
        minified: bool = False,
        pages: Optional[set[int]] = None,
    ) -> str:
        r"""Exports the document content to a DocumentToken format.

        Operates on a slice of the document's body as defined through arguments
        from_element and to_element; defaulting to the whole main_text.

        :param delim: str:  (Default value = "")  Deprecated
        :param from_element: int:  (Default value = 0)
        :param to_element: Optional[int]:  (Default value = None)
        :param labels: set[DocItemLabel]
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param add_page_index: bool:  (Default value = True)
        :param # table specific flagsadd_table_cell_location: bool
        :param add_table_cell_text: bool:  (Default value = True)
        :param minified: bool:  (Default value = False)
        :param pages: set[int]: (Default value = None)
        :returns: The content of the document formatted as a DocTags string.
        :rtype: str
        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
        serializer = DocTagsDocSerializer(
            doc=self,
            params=DocTagsParams(
                labels=my_labels,
                # layers=...,  # not exposed
                start_idx=from_element,
                stop_idx=to_element,
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                # add_caption=...,  # not exposed
                add_content=add_content,
                add_page_break=add_page_index,
                add_table_cell_location=add_table_cell_location,
                add_table_cell_text=add_table_cell_text,
                pages=pages,
                mode=(DocTagsParams.Mode.MINIFIED if minified else DocTagsParams.Mode.HUMAN_FRIENDLY),
            ),
        )
        ser_res = serializer.serialize()
        return ser_res.text

    def export_to_doclang(
        self,
    ) -> str:
        """Export to DocLang."""
        from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams

        serializer = DocLangDocSerializer(
            doc=self,
            params=DocLangParams(),
        )
        return serializer.serialize().text

    def save_as_doclang(
        self,
        filename: Union[str, Path],
    ) -> None:
        """Save the document as DocLang."""
        if isinstance(filename, str):
            filename = Path(filename)
        out = self.export_to_doclang()
        filename.write_text(f"{out}\n", encoding="utf-8")

    def save_as_doclang_archive(
        self,
        filename: Union[str, Path],
        *,
        artifacts_dir: Optional[Path] = None,
        validate: bool = False,
    ) -> None:
        """Save the document as a DocLang OPC archive (``.dclx``).

        Picture and page images are always stored outside the markup, under
        ``assets/`` and ``pages/`` in the archive respectively.
        """
        from doclang import pack

        from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams

        if isinstance(filename, str):
            filename = Path(filename)

        def _write_archive(staging_root: Path) -> None:
            assets_dir = staging_root / "assets"
            pages_dir = staging_root / "pages"

            doc = self._make_copy_with_refmode(
                artifacts_dir=assets_dir,
                image_mode=ImageRefMode.REFERENCED,
                page_no=None,
                reference_path=staging_root,
            )

            serializer = DocLangDocSerializer(
                doc=doc,
                params=DocLangParams(image_mode=ImageRefMode.REFERENCED),
            )
            document_path = staging_root / "document.dclg.xml"
            document_path.write_text(f"{serializer.serialize().text}\n", encoding="utf-8")

            pack_kwargs: dict[str, Any] = {
                "document": document_path,
                "output": filename,
                "validate": validate,
            }

            pages: dict[int, Path] = {}
            for p_no, page in self.pages.items():
                if page.image is None:
                    continue
                img = page.image.pil_image
                if img is None:
                    continue
                page_path = pages_dir / f"{p_no}.png"
                page_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(page_path)
                pages[p_no] = page_path
            if pages:
                pack_kwargs["pages"] = pages

            if assets_dir.is_dir() and any(assets_dir.iterdir()):
                pack_kwargs["assets"] = assets_dir

            pack(**pack_kwargs)

        if artifacts_dir is None:
            with tempfile.TemporaryDirectory() as staging_name:
                _write_archive(Path(staging_name))
        else:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            _write_archive(artifacts_dir)

    @classmethod
    def load_from_doclang_archive(
        cls,
        filename: Union[str, Path],
        *,
        artifacts_dir: Optional[Path] = None,
        validate: bool = False,
    ) -> "DoclingDocument":
        """Load a DoclingDocument from a DocLang OPC archive (``.dclx``).

        The archive is extracted to ``artifacts_dir`` (default: ``<name>_artifacts`` next
        to the ``.dclx`` file). Relative ``<src uri=\"...\"/>`` paths and optional
        ``pages/`` images are resolved inside that directory.
        """
        from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer

        if isinstance(filename, str):
            filename = Path(filename)

        artifacts_dir, _ = cls(name="")._get_output_paths(filename, artifacts_dir)
        safe_extract_zip_archive(filename, artifacts_dir)

        document_xml = artifacts_dir / "document.xml"
        if not document_xml.is_file():
            raise ValueError(f"DocLang archive missing document.xml: {filename}")

        if validate:
            from doclang.validation import validate as doclang_validate

            doclang_validate(document_xml)

        doc = DocLangDocDeserializer().deserialize_str(
            document_xml.read_text(encoding="utf-8"),
            media_root=artifacts_dir,
        )
        doc.name = filename.stem
        cls._restore_doclang_archive_page_images(doc=doc, archive_root=artifacts_dir)
        return doc

    @staticmethod
    def _restore_doclang_archive_page_images(*, doc: "DoclingDocument", archive_root: Path) -> None:
        pages_dir = archive_root / "pages"
        if not pages_dir.is_dir():
            return

        archive_root = archive_root.resolve()
        for page_file in sorted(pages_dir.iterdir()):
            if not page_file.is_file():
                continue

            archive_rel = f"pages/{page_file.name}"
            validate_archive_relative_path(archive_rel, label="page image")
            resolved = resolve_archive_path(archive_root, archive_rel)
            stem = page_file.stem
            if not stem.isdigit():
                continue

            page_no = int(stem)
            if page_no not in doc.pages:
                continue

            with PILImage.open(resolved) as pil:
                pil_copy = pil.copy()
            mimetype = mimetypes.guess_type(page_file.name)[0] or "image/png"
            doc.pages[page_no].image = ImageRef(
                mimetype=mimetype,
                dpi=72,
                size=Size(width=pil_copy.width, height=pil_copy.height),
                uri=resolved,
            )

    def _export_to_indented_text(
        self,
        indent="  ",
        max_text_len: int = -1,
        explicit_tables: bool = False,
    ):
        """Export the document to indented text to expose hierarchy."""
        result = []

        def get_text(text: str, max_text_len: int):
            middle = " ... "

            if max_text_len == -1:
                return text
            elif len(text) < max_text_len + len(middle):
                return text
            else:
                tbeg = int((max_text_len - len(middle)) / 2)
                tend = int(max_text_len - tbeg)

                return text[0:tbeg] + middle + text[-tend:]

        for i, (item, level) in enumerate(self.iterate_items(with_groups=True)):
            if isinstance(item, GroupItem):
                result.append(indent * level + f"item-{i} at level {level}: {item.label}: group {item.name}")

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.TITLE]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(indent * level + f"item-{i} at level {level}: {item.label}: {text}")

            elif isinstance(item, SectionHeaderItem):
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(indent * level + f"item-{i} at level {level}: {item.label}: {text}")

            elif isinstance(item, TextItem) and item.label in [DocItemLabel.CODE]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(indent * level + f"item-{i} at level {level}: {item.label}: {text}")

            elif isinstance(item, ListItem) and item.label in [DocItemLabel.LIST_ITEM]:
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(indent * level + f"item-{i} at level {level}: {item.label}: {text}")

            elif isinstance(item, TextItem):
                text = get_text(text=item.text, max_text_len=max_text_len)

                result.append(indent * level + f"item-{i} at level {level}: {item.label}: {text}")

            elif isinstance(item, TableItem):
                result.append(
                    indent * level
                    + f"item-{i} at level {level}: {item.label} with "
                    + f"[{item.data.num_rows}x{item.data.num_cols}]"
                )

                for _ in item.captions:
                    caption = _.resolve(self)
                    result.append(
                        indent * (level + 1) + f"item-{i} at level {level + 1}: {caption.label}: " + f"{caption.text}"
                    )

                if explicit_tables:
                    grid: list[list[str]] = []
                    for i, row in enumerate(item.data.grid):
                        grid.append([])
                        for j, cell in enumerate(row):
                            if j < 10:
                                text = get_text(cell._get_text(doc=self), max_text_len=16)
                                grid[-1].append(text)

                    result.append("\n" + tabulate(grid) + "\n")

            elif isinstance(item, PictureItem):
                result.append(indent * level + f"item-{i} at level {level}: {item.label}")

                for _ in item.captions:
                    caption = _.resolve(self)
                    result.append(
                        indent * (level + 1) + f"item-{i} at level {level + 1}: {caption.label}: " + f"{caption.text}"
                    )

            elif isinstance(item, DocItem):
                result.append(indent * (level + 1) + f"item-{i} at level {level}: {item.label}: ignored")

        return "\n".join(result)

    def add_page(self, page_no: int, size: Size, image: Optional[ImageRef] = None) -> PageItem:
        """add_page.

        :param page_no: int:
        :param size: Size:

        """
        pitem = PageItem(page_no=page_no, size=size, image=image)

        self.pages[page_no] = pitem
        return pitem

    def get_visualization(
        self,
        show_label: bool = True,
        show_branch_numbering: bool = False,
        viz_mode: Literal["reading_order", "key_value"] = "reading_order",
        show_cell_id: bool = False,
    ) -> dict[Optional[int], PILImage.Image]:
        """Get visualization of the document as images by page.

        :param show_label: Show labels on elements (applies to all visualizers).
        :type show_label: bool
        :param show_branch_numbering: Show branch numbering (reading order visualizer only).
        :type show_branch_numbering: bool
        :param visualizer: Which visualizer to use. One of 'reading_order' (default), 'key_value'.
        :type visualizer: str
        :param show_cell_id: Show cell IDs (key value visualizer only).
        :type show_cell_id: bool

        :returns: Dictionary mapping page numbers to PIL images.
        :rtype: dict[Optional[int], PILImage.Image]
        """
        from docling_core.transforms.visualizer.base import BaseVisualizer
        from docling_core.transforms.visualizer.key_value_visualizer import (
            KeyValueVisualizer,
        )
        from docling_core.transforms.visualizer.layout_visualizer import (
            LayoutVisualizer,
        )
        from docling_core.transforms.visualizer.reading_order_visualizer import (
            ReadingOrderVisualizer,
        )

        visualizer_obj: BaseVisualizer
        if viz_mode == "reading_order":
            visualizer_obj = ReadingOrderVisualizer(
                base_visualizer=LayoutVisualizer(
                    params=LayoutVisualizer.Params(
                        show_label=show_label,
                    ),
                ),
                params=ReadingOrderVisualizer.Params(
                    show_branch_numbering=show_branch_numbering,
                ),
            )
        elif viz_mode == "key_value":
            visualizer_obj = KeyValueVisualizer(
                base_visualizer=LayoutVisualizer(
                    params=LayoutVisualizer.Params(
                        show_label=show_label,
                    ),
                ),
                params=KeyValueVisualizer.Params(
                    show_label=show_label,
                    show_cell_id=show_cell_id,
                ),
            )
        else:
            raise ValueError(f"Unknown visualization mode: {viz_mode}")

        images = visualizer_obj.get_visualization(doc=self)
        return images

    @field_validator("version")
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this document version is compatible with SDK schema version."""
        sdk_match = re.match(VERSION_PATTERN, CURRENT_VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or sdk_match is None
            or int(doc_match["major"]) != int(sdk_match["major"])
            or int(doc_match["minor"]) > int(sdk_match["minor"])
        ):
            raise ValueError(f"Doc version {v} incompatible with SDK schema version {CURRENT_VERSION}")
        else:
            return CURRENT_VERSION

    @model_validator(mode="after")
    def _validate_unique_refs(self) -> Self:
        seen: set[str] = set()
        item_lists: list[Iterable[NodeItem]] = [
            self.groups,
            self.texts,
            self.pictures,
            self.tables,
            self.key_value_items,
            self.form_items,
            self.field_regions,
            self.field_items,
        ]
        for item_list in item_lists:
            for item in item_list:
                if isinstance(item, NodeItem):
                    if item.self_ref in seen:
                        raise ValueError(f"Duplicate ref: {item.self_ref}")
                    seen.add(item.self_ref)
                else:
                    raise ValueError(f"Unknown element encountered: {item}")
        return self

    def _normalize_table_children_from_rich_cells(self):
        """Repair missing table child links for already-parented rich table cells."""
        for table in self.tables:
            child_crefs = {child.cref for child in table.children}

            for cell in table.data.table_cells:
                if not isinstance(cell, RichTableCell):
                    continue

                item = cell.ref.resolve(self)
                if item.parent is not None and item.parent.cref == table.self_ref and cell.ref.cref not in child_crefs:
                    table.children.append(cell.ref)
                    child_crefs.add(cell.ref.cref)

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        """validate_document."""
        with warnings.catch_warnings():
            # ignore warning from deprecated furniture
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.validate_tree(self.body, raise_on_error=True)
            self.validate_tree(self.furniture, raise_on_error=True)

        self._clamp_provenance_bboxes_to_pages()

        return self

    @model_validator(mode="after")
    def validate_misplaced_list_items(self) -> Self:
        """validate_misplaced_list_items."""
        # find list items without list parent, putting successive ones together
        misplaced_list_items: list[list[ListItem]] = []
        prev: Optional[NodeItem] = None
        for item, _ in self.iterate_items(
            traverse_pictures=True,
            included_content_layers=set(ContentLayer),
            with_groups=True,  # so that we can distinguish neighboring lists
        ):
            if isinstance(item, ListItem) and (
                item.parent is None or not isinstance(item.parent.resolve(doc=self), ListGroup)
            ):
                if isinstance(prev, ListItem) and (
                    prev.parent is None or prev.parent.resolve(self) == self.body
                ):  # case of continuing list
                    misplaced_list_items[-1].append(item)
                else:  # case of new list
                    misplaced_list_items.append([item])
            prev = item

        for curr_list_items in reversed(misplaced_list_items):
            # add group
            new_group = ListGroup(self_ref="#")
            self.insert_item_before_sibling(
                new_item=new_group,
                sibling=curr_list_items[0],
            )

            # delete list items from document (should not be affected by group addition)
            self.delete_items(node_items=list(curr_list_items))

            # add list items to new group
            for li in curr_list_items:
                self.add_list_item(
                    text=li.text,
                    enumerated=li.enumerated,
                    marker=li.marker,
                    orig=li.orig,
                    prov=li.prov[0] if li.prov else None,
                    parent=new_group,
                    content_layer=li.content_layer,
                    formatting=li.formatting,
                    hyperlink=li.hyperlink,
                )
        return self

    class _DocIndex(BaseModel):
        """A document merge buffer."""

        groups: list[GroupItem] = []
        texts: list[TextItem] = []
        pictures: list[PictureItem] = []
        tables: list[TableItem] = []
        key_value_items: list[KeyValueItem] = []
        form_items: list[FormItem] = []
        field_regions: list[FieldRegionItem] = []
        field_items: list[FieldItem] = []

        pages: dict[int, PageItem] = {}

        _body: Optional[GroupItem] = None
        _max_page: int = 0
        _names: list[str] = []

        def get_item_list(self, key: str) -> list[NodeItem]:
            return getattr(self, key)

        def index(self, doc: "DoclingDocument", page_nrs: Optional[set[int]] = None) -> None:
            if page_nrs is not None and (unavailable_page_nrs := page_nrs - set(doc.pages.keys())):
                raise ValueError(f"The following page numbers are not present in the document: {unavailable_page_nrs}")

            orig_ref_to_new_ref: dict[str, str] = {}
            page_delta = self._max_page - min(doc.pages.keys()) + 1 if doc.pages else 0

            if self._body is None:
                self._body = GroupItem(**doc.body.model_dump(exclude={"children"}))

            self._names.append(doc.name)

            # record starting indices so post-processing only touches new items
            post_processing_keys = [
                "groups",
                "texts",
                "pictures",
                "tables",
                "key_value_items",
                "form_items",
                "field_regions",
                "field_items",
            ]
            start_indices = {k: len(self.get_item_list(k)) for k in post_processing_keys}

            # collect items in traversal order
            for item, _ in doc._iterate_items_with_stack(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
                page_nrs=page_nrs,
            ):
                key = item.self_ref.split("/")[1]
                is_body = key == "body"
                new_cref = "#/body" if is_body else f"#/{key}/{len(self.get_item_list(key))}"
                # register cref mapping:
                orig_ref_to_new_ref[item.self_ref] = new_cref

                if not is_body:
                    new_item = copy.deepcopy(item)
                    new_item.children = []

                    # put item in the right list
                    self.get_item_list(key).append(new_item)

                    # update item's self reference
                    new_item.self_ref = new_cref

                    if isinstance(new_item, DocItem):
                        # update page numbers
                        for prov in new_item.prov:
                            prov.page_no += page_delta
                        if isinstance(new_item, KeyValueItem | FormItem):
                            for graph_cell in new_item.graph.cells:
                                if graph_cell.prov is not None:
                                    graph_cell.prov.page_no += page_delta

                    if item.parent:
                        # set item's parent
                        new_parent_cref = orig_ref_to_new_ref.get(item.parent.cref)
                        if new_parent_cref is None:
                            parent_ref = item.parent
                            while new_parent_cref is None and parent_ref is not None:
                                parent_ref = RefItem(cref=parent_ref.resolve(doc).parent.cref)
                                new_parent_cref = orig_ref_to_new_ref.get(parent_ref.cref)

                            if new_parent_cref is not None:
                                warnings.warn(
                                    f"Parent {item.parent.cref} not found in indexed nodes, "
                                    f"using ancestor {new_parent_cref} instead"
                                )
                            else:
                                warnings.warn("No ancestor found in indexed nodes, using body as parent")
                                new_parent_cref = "#/body"

                        new_item.parent = RefItem(cref=new_parent_cref)

                        # add item to parent's children
                        path_components = new_parent_cref.split("/")
                        num_components = len(path_components)
                        if num_components == 3:
                            _, parent_key, parent_index_str = path_components
                            parent_index = int(parent_index_str)
                            parent_item = self.get_item_list(parent_key)[parent_index]

                            # update rich table cells references:
                            if isinstance(parent_item, TableItem):
                                for table_cell in parent_item.data.table_cells:
                                    if isinstance(table_cell, RichTableCell) and table_cell.ref.cref == item.self_ref:
                                        table_cell.ref.cref = new_cref
                                        break

                        elif num_components == 2 and path_components[1] == "body":
                            parent_item = self._body
                        else:
                            raise RuntimeError(f"Unsupported ref format: {new_parent_cref}")
                        parent_item.children.append(RefItem(cref=new_cref))

            # rewrite FloatingItem explicit refs starting from start_indices to avoid corrupting items from prior calls
            for key in post_processing_keys:
                for idx_item in self.get_item_list(key)[start_indices[key] :]:
                    if isinstance(idx_item, FloatingItem):
                        idx_item.captions = [
                            RefItem(cref=orig_ref_to_new_ref[cap.cref])
                            for cap in idx_item.captions
                            if cap.cref in orig_ref_to_new_ref
                        ]
                        idx_item.references = [
                            RefItem(cref=orig_ref_to_new_ref[ref.cref])
                            for ref in idx_item.references
                            if ref.cref in orig_ref_to_new_ref
                        ]
                        idx_item.footnotes = [
                            RefItem(cref=orig_ref_to_new_ref[fn.cref])
                            for fn in idx_item.footnotes
                            if fn.cref in orig_ref_to_new_ref
                        ]

            # update pages
            new_max_page = None
            for page_nr in doc.pages:
                if page_nrs is None or page_nr in page_nrs:
                    new_page = copy.deepcopy(doc.pages[page_nr])
                    new_page_nr = page_nr + page_delta
                    new_page.page_no = new_page_nr
                    self.pages[new_page_nr] = new_page
                    if new_max_page is None or new_page_nr > new_max_page:
                        new_max_page = new_page_nr
            if new_max_page is not None:
                self._max_page = new_max_page

        def get_name(self) -> str:
            if not self._names:
                return ""
            squeezed: list[str] = [self._names[0]]
            for name in self._names[1:]:
                if name != squeezed[-1]:
                    squeezed.append(name)
            return " + ".join(squeezed)

    def _update_from_index(self, doc_index: "_DocIndex") -> None:
        if doc_index._body is not None:
            self.body = doc_index._body
        self.groups = doc_index.groups
        self.texts = doc_index.texts
        self.pictures = doc_index.pictures
        self.tables = doc_index.tables
        self.key_value_items = doc_index.key_value_items
        self.form_items = doc_index.form_items
        self.field_regions = doc_index.field_regions
        self.field_items = doc_index.field_items
        self.pages = doc_index.pages
        self.name = doc_index.get_name()

    def _normalize_references(self) -> None:
        doc_index = DoclingDocument._DocIndex()
        doc_index.index(doc=self)
        self._update_from_index(doc_index)

    def filter(self, page_nrs: Optional[set[int]] = None) -> "DoclingDocument":
        """Create a new document based on the provided filter parameters."""
        doc_index = DoclingDocument._DocIndex()
        doc_index.index(doc=self, page_nrs=page_nrs)
        res_doc = DoclingDocument(name=self.name)
        res_doc._update_from_index(doc_index)
        return res_doc

    @classmethod
    def concatenate(cls, docs: Sequence["DoclingDocument"]) -> "DoclingDocument":
        """Concatenate multiple documents into a single document."""
        doc_index = DoclingDocument._DocIndex()
        for doc in docs:
            doc_index.index(doc=doc)

        res_doc = DoclingDocument(name=doc_index.get_name())
        res_doc._update_from_index(doc_index)
        return res_doc

    def _validate_rules(self, raise_on_error: bool = True):
        def _handle(error: Exception):
            if raise_on_error:
                raise error
            else:
                warnings.warn(str(error))

        def validate_furniture(doc: DoclingDocument):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                has_furniture_children = len(doc.furniture.children) > 0
            if has_furniture_children:
                _handle(
                    ValueError(f"Deprecated furniture node {doc.furniture.self_ref} has children"),
                )

        def validate_list_group(doc: DoclingDocument, item: ListGroup):
            for ref in item.children:
                child = ref.resolve(doc)
                if not isinstance(child, ListItem):
                    _handle(
                        ValueError(
                            f"ListGroup {item.self_ref} contains non-ListItem {child.self_ref} ({child.label=})"
                        ),
                    )

        def validate_list_item(doc: DoclingDocument, item: ListItem):
            if item.parent is None:
                _handle(
                    ValueError(f"ListItem {item.self_ref} has no parent"),
                )
            elif not isinstance(item.parent.resolve(doc), ListGroup):
                _handle(
                    ValueError(f"ListItem {item.self_ref} has non-ListGroup parent: {item.parent.cref}"),
                )

        def validate_group(doc: DoclingDocument, item: GroupItem):
            if item.parent and not item.children:  # tolerate empty body, but not other groups
                _handle(
                    ValueError(f"Group {item.self_ref} has no children"),
                )

        validate_furniture(self)

        for item, _ in self.iterate_items(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers=set(ContentLayer),
        ):
            if isinstance(item, ListGroup):
                validate_list_group(self, item)

            elif isinstance(item, GroupItem):
                validate_group(self, item)

            elif isinstance(item, ListItem):
                validate_list_item(self, item)

    def add_table_cell(self, table_item: TableItem, cell: TableCell) -> None:
        """Add a table cell to the table."""
        if isinstance(cell, RichTableCell):
            item = cell.ref.resolve(doc=self)
            if isinstance(item, NodeItem) and ((not item.parent) or item.parent.cref != table_item.self_ref):
                raise ValueError(f"Trying to add cell with another parent {item.parent} to {table_item.self_ref}")
        table_item.data.table_cells.append(cell)


# deprecated aliases (kept for backwards compatibility):
