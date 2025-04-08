#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define base classes for serialization."""
import sys
from abc import abstractmethod
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import AnyUrl, BaseModel, NonNegativeInt, computed_field
from typing_extensions import Self, override

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
from docling_core.types.doc.document import (
    DOCUMENT_TOKENS_EXPORT_LABELS,
    ContentLayer,
    DocItem,
    DoclingDocument,
    FloatingItem,
    Formatting,
    FormItem,
    InlineGroup,
    KeyValueItem,
    NodeItem,
    OrderedList,
    PictureItem,
    TableItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel

_DEFAULT_LABELS = DOCUMENT_TOKENS_EXPORT_LABELS
_DEFAULT_LAYERS = {cl for cl in ContentLayer}


class CommonParams(BaseModel):
    """Common serialization parameters."""

    # allowlists with non-recursive semantics, i.e. if a list group node is outside the
    # range and some of its children items are within, they will be serialized
    labels: set[DocItemLabel] = _DEFAULT_LABELS
    layers: set[ContentLayer] = _DEFAULT_LAYERS
    pages: Optional[set[int]] = None  # None means all pages are allowed

    # slice-like semantics: start is included, stop is excluded
    start_idx: NonNegativeInt = 0
    stop_idx: NonNegativeInt = sys.maxsize

    include_formatting: bool = True
    include_hyperlinks: bool = True
    caption_delim: str = " "

    def merge_with_patch(self, patch: dict[str, Any]) -> Self:
        """Create an instance by merging the provided patch dict on top of self."""
        res = self.model_validate({**self.model_dump(), **patch})
        return res


class DocSerializer(BaseModel, BaseDocSerializer):
    """Class for document serializers."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "forbid"

    doc: DoclingDocument

    text_serializer: BaseTextSerializer
    table_serializer: BaseTableSerializer
    picture_serializer: BasePictureSerializer
    key_value_serializer: BaseKeyValueSerializer
    form_serializer: BaseFormSerializer
    fallback_serializer: BaseFallbackSerializer

    list_serializer: BaseListSerializer
    inline_serializer: BaseInlineSerializer

    params: CommonParams = CommonParams()

    _excluded_refs_cache: dict[str, list[str]] = {}

    @computed_field  # type: ignore[misc]
    @cached_property
    def _captions_of_some_item(self) -> set[str]:
        layers = {cl for cl in ContentLayer}  # TODO review
        refs = {
            cap.cref
            for (item, _) in self.doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=layers,
            )
            for cap in (item.captions if isinstance(item, FloatingItem) else [])
        }
        return refs

    @override
    def get_excluded_refs(self, **kwargs) -> list[str]:
        """References to excluded items."""
        params = self.params.merge_with_patch(patch=kwargs)
        params_json = params.model_dump_json()
        refs = self._excluded_refs_cache.get(params_json)
        if refs is None:
            refs = [
                item.self_ref
                for ix, (item, _) in enumerate(
                    self.doc.iterate_items(
                        with_groups=True,
                        traverse_pictures=True,
                        included_content_layers=params.layers,
                    )
                )
                if (
                    (ix < params.start_idx or ix >= params.stop_idx)
                    or (
                        isinstance(item, DocItem)
                        and (
                            item.label not in params.labels
                            or item.content_layer not in params.layers
                            or (
                                params.pages is not None
                                and (
                                    (not item.prov)
                                    or item.prov[0].page_no not in params.pages
                                )
                            )
                        )
                    )
                )
            ]
            self._excluded_refs_cache[params_json] = refs
        return refs

    @abstractmethod
    def serialize_page(
        self, *, parts: list[SerializationResult], **kwargs
    ) -> SerializationResult:
        """Serialize a page out of its parts."""
        ...

    @abstractmethod
    def serialize_doc(
        self, *, pages: dict[Optional[int], SerializationResult], **kwargs
    ) -> SerializationResult:
        """Serialize a document out of its pages."""
        ...

    def _serialize_body(self) -> SerializationResult:
        """Serialize the document body."""
        # find page ranges if available; otherwise regard whole doc as a single page
        prev_start: int = 0
        prev_page_nr: Optional[int] = None
        range_by_page_nr: dict[Optional[int], tuple[int, int]] = {}

        for ix, (item, _) in enumerate(
            self.doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=self.params.layers,
            )
        ):
            if isinstance(item, DocItem):
                if item.prov:
                    page_no = item.prov[0].page_no
                    if prev_page_nr is None or page_no > prev_page_nr:
                        if prev_page_nr is not None:  # close previous range
                            range_by_page_nr[prev_page_nr] = (prev_start, ix)

                        prev_start = ix
                        # could alternatively always start 1st page from 0:
                        # prev_start = ix if prev_page_nr is not None else 0

                        prev_page_nr = page_no

        # close last (and single if no pages) range
        range_by_page_nr[prev_page_nr] = (prev_start, sys.maxsize)

        page_results: dict[Optional[int], SerializationResult] = {}
        for page_nr in range_by_page_nr:
            page_range = range_by_page_nr[page_nr]
            params_to_pass = deepcopy(self.params)
            params_to_pass.start_idx = page_range[0]
            params_to_pass.stop_idx = page_range[1]
            subparts = self.get_parts(**params_to_pass.model_dump())
            page_res = self.serialize_page(parts=subparts)
            page_results[page_nr] = page_res
        res = self.serialize_doc(pages=page_results)
        return res

    @override
    def serialize(
        self,
        *,
        item: Optional[NodeItem] = None,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serialize a given node."""
        my_visited: set[str] = visited if visited is not None else set()
        empty_res = SerializationResult(text="")
        if item is None or item == self.doc.body:
            if self.doc.body.self_ref not in my_visited:
                my_visited.add(self.doc.body.self_ref)
                return self._serialize_body()
            else:
                return empty_res

        ########
        # groups
        ########
        if isinstance(item, (UnorderedList, OrderedList)):
            part = self.list_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                list_level=list_level,
                is_inline_scope=is_inline_scope,
                visited=my_visited,
                **kwargs,
            )
        elif isinstance(item, InlineGroup):
            part = self.inline_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                list_level=list_level,
                visited=my_visited,
                **kwargs,
            )
        ###########
        # doc items
        ###########
        elif isinstance(item, TextItem):
            if item.self_ref in self._captions_of_some_item:
                # those captions will be handled by the floating item holding them
                return empty_res
            else:
                part = (
                    self.text_serializer.serialize(
                        item=item,
                        doc_serializer=self,
                        doc=self.doc,
                        is_inline_scope=is_inline_scope,
                        **kwargs,
                    )
                    if item.self_ref not in self.get_excluded_refs(**kwargs)
                    else empty_res
                )
        elif isinstance(item, TableItem):
            part = self.table_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs,
            )
        elif isinstance(item, PictureItem):
            part = self.picture_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                visited=my_visited,
                **kwargs,
            )
        elif isinstance(item, KeyValueItem):
            part = self.key_value_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs,
            )
        elif isinstance(item, FormItem):
            part = self.form_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs,
            )
        else:
            part = self.fallback_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs,
            )
        return part

    # making some assumptions about the kwargs it can pass
    @override
    def get_parts(
        self,
        item: Optional[NodeItem] = None,
        *,
        traverse_pictures: bool = False,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> list[SerializationResult]:
        """Get the components to be combined for serializing this node."""
        parts: list[SerializationResult] = []
        my_visited: set[str] = visited if visited is not None else set()
        params = self.params.merge_with_patch(patch=kwargs)
        for item, _ in self.doc.iterate_items(
            root=item,
            with_groups=True,
            traverse_pictures=traverse_pictures,
            included_content_layers=params.layers,
        ):
            if item.self_ref in my_visited:
                continue
            else:
                my_visited.add(item.self_ref)
            part = self.serialize(
                item=item,
                list_level=list_level,
                is_inline_scope=is_inline_scope,
                visited=my_visited,
                **kwargs,
            )
            if part.text:
                parts.append(part)
        return parts

    @override
    def post_process(
        self,
        text: str,
        *,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs,
    ) -> str:
        """Apply some text post-processing steps."""
        params = self.params.merge_with_patch(patch=kwargs)
        res = text
        if params.include_formatting and formatting:
            if formatting.bold:
                res = self.serialize_bold(text=res)
            if formatting.italic:
                res = self.serialize_italic(text=res)
            if formatting.underline:
                res = self.serialize_underline(text=res)
            if formatting.strikethrough:
                res = self.serialize_strikethrough(text=res)
        if params.include_hyperlinks and hyperlink:
            res = self.serialize_hyperlink(text=res, hyperlink=hyperlink)
        return res

    @override
    def serialize_bold(self, text: str, **kwargs) -> str:
        """Hook for bold formatting serialization."""
        return text

    @override
    def serialize_italic(self, text: str, **kwargs) -> str:
        """Hook for italic formatting serialization."""
        return text

    @override
    def serialize_underline(self, text: str, **kwargs) -> str:
        """Hook for underline formatting serialization."""
        return text

    @override
    def serialize_strikethrough(self, text: str, **kwargs) -> str:
        """Hook for strikethrough formatting serialization."""
        return text

    @override
    def serialize_hyperlink(
        self, text: str, hyperlink: Union[AnyUrl, Path], **kwargs
    ) -> str:
        """Hook for hyperlink serialization."""
        return text

    @override
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        params = self.params.merge_with_patch(patch=kwargs)
        if DocItemLabel.CAPTION in params.labels:
            text_parts: list[str] = [
                it.text
                for cap in item.captions
                if isinstance(it := cap.resolve(self.doc), TextItem)
                and it.self_ref not in self.get_excluded_refs(**kwargs)
            ]
            text_res = params.caption_delim.join(text_parts)
            text_res = self.post_process(text=text_res)
        else:
            text_res = ""
        return SerializationResult(text=text_res)
