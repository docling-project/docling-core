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
from docling_core.types.doc.document import (
    DEFAULT_CONTENT_LAYERS,
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
    PictureClassificationData,
    PictureDescriptionData,
    PictureItem,
    PictureMoleculeData,
    TableItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel

_DEFAULT_LABELS = DOCUMENT_TOKENS_EXPORT_LABELS


class CommonParams(BaseModel):
    """Common serialization parameters."""

    # slice-like semantics: start is included, stop is excluded
    start_idx: NonNegativeInt = 0
    stop_idx: NonNegativeInt = sys.maxsize


class DocSerializer(BaseModel, BaseDocSerializer):
    """Class for document serializers."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    doc: DoclingDocument

    include_formatting: bool = True
    include_hyperlinks: bool = True
    escape_underscores: bool = True

    # this filtering criteria are non-recursive;
    # e.g. if a list group node is outside the range and some of its children items are
    # within, they will be serialized
    labels: set[DocItemLabel] = _DEFAULT_LABELS
    layers: set[ContentLayer] = DEFAULT_CONTENT_LAYERS
    pages: Optional[set[int]] = None

    text_serializer: BaseTextSerializer
    table_serializer: BaseTableSerializer
    picture_serializer: BasePictureSerializer
    key_value_serializer: BaseKeyValueSerializer
    form_serializer: BaseFormSerializer
    fallback_serializer: BaseFallbackSerializer

    list_serializer: BaseListSerializer
    inline_serializer: BaseInlineSerializer

    params: CommonParams = CommonParams()

    @computed_field  # type: ignore[misc]
    @cached_property
    def _params_dict(self) -> dict[str, Any]:
        return self.params.model_dump()

    # TODO add cache base on start-stop params
    @override
    def get_excluded_refs(self, **kwargs) -> list[str]:
        """References to excluded items."""
        params = CommonParams(**kwargs)
        refs: list[str] = [
            item.self_ref
            for ix, (item, _) in enumerate(
                self.doc.iterate_items(
                    with_groups=True,
                    traverse_pictures=True,
                )
            )
            if (
                (ix < params.start_idx or ix >= params.stop_idx)
                or (
                    isinstance(item, DocItem)
                    and (
                        item.label not in self.labels
                        or item.content_layer not in self.layers
                        or (
                            self.pages is not None
                            and (
                                (not item.prov)
                                or item.prov[0].page_no not in self.pages
                            )
                        )
                    )
                )
            )
        ]
        return refs

    @abstractmethod
    def serialize_page(self, parts: list[SerializationResult]) -> SerializationResult:
        """Serialize a page out of its parts."""
        ...

    @abstractmethod
    def serialize_doc(self, pages: list[SerializationResult]) -> SerializationResult:
        """Serialize a document out of its pages."""
        ...

    def _serialize_body(self) -> SerializationResult:
        """Serialize the document body."""
        # find page ranges if available; otherwise regard whole doc as a single page
        last_page: Optional[int] = None
        starts: list[int] = []
        for ix, (item, _) in enumerate(
            self.doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
            )
        ):
            if isinstance(item, DocItem):
                if item.prov:
                    if last_page is None or item.prov[0].page_no > last_page:
                        starts.append(ix)
                        last_page = item.prov[0].page_no
        page_ranges = [
            (
                (starts[i] if i > 0 else 0),
                (starts[i + 1] if i < len(starts) - 1 else sys.maxsize),
            )
            for i, _ in enumerate(starts)
        ] or [
            (0, sys.maxsize)
        ]  # use whole range if no pages detected

        page_results: list[SerializationResult] = []
        for page_range in page_ranges:
            params_to_pass = deepcopy(self.params)
            params_to_pass.start_idx = page_range[0]
            params_to_pass.stop_idx = page_range[1]
            subparts = self.get_parts(**params_to_pass.model_dump())
            page_res = self.serialize_page(subparts)
            page_results.append(page_res)
        res = self.serialize_doc(page_results)
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

        label_blocklist = {
            # captions only considered in context of floating items (pictures, tables)
            DocItemLabel.CAPTION,
        }

        kwargs_to_pass = {**self._params_dict, **kwargs}

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
                **kwargs_to_pass,
            )
        elif isinstance(item, InlineGroup):
            part = self.inline_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                list_level=list_level,
                visited=my_visited,
                **kwargs_to_pass,
            )
        ###########
        # doc items
        ###########
        elif isinstance(item, DocItem) and item.label in label_blocklist:
            return empty_res
        elif isinstance(item, TextItem):
            part = (
                self.text_serializer.serialize(
                    item=item,
                    doc_serializer=self,
                    doc=self.doc,
                    is_inline_scope=is_inline_scope,
                    **kwargs_to_pass,
                )
                if item.self_ref not in self.get_excluded_refs(**kwargs_to_pass)
                else empty_res
            )
        elif isinstance(item, TableItem):
            part = self.table_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs_to_pass,
            )
        elif isinstance(item, PictureItem):
            part = self.picture_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                visited=my_visited,
                **kwargs_to_pass,
            )
        elif isinstance(item, KeyValueItem):
            part = self.key_value_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs_to_pass,
            )
        elif isinstance(item, FormItem):
            part = self.form_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs_to_pass,
            )
        else:
            part = self.fallback_serializer.serialize(
                item=item,
                doc_serializer=self,
                doc=self.doc,
                **kwargs_to_pass,
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

        for item, _ in self.doc.iterate_items(
            root=item,
            with_groups=True,
            traverse_pictures=traverse_pictures,
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
        res = text
        if self.include_formatting and formatting:
            if formatting.bold:
                res = self.serialize_bold(text=res)
            if formatting.italic:
                res = self.serialize_italic(text=res)
            if formatting.underline:
                res = self.serialize_underline(text=res)
            if formatting.strikethrough:
                res = self.serialize_strikethrough(text=res)
        if self.include_hyperlinks and hyperlink:
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
        separator: Optional[str] = None,
        **kwargs,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        if DocItemLabel.CAPTION in self.labels:
            text_parts: list[str] = [
                it.text
                for cap in item.captions
                if isinstance(it := cap.resolve(self.doc), TextItem)
                and it.self_ref not in self.get_excluded_refs(**kwargs)
            ]
            text_res = (separator or "\n").join(text_parts)
            text_res = self.post_process(text=text_res)
        else:
            text_res = ""
        return SerializationResult(text=text_res)


class PictureSerializer(BasePictureSerializer):
    """Class for picture serializers."""

    # helper function
    def _serialize_content(
        self,
        item: PictureItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        separator: Optional[str] = None,
        visited: Optional[set[str]] = None,
        **kwargs,
    ) -> SerializationResult:
        parts = doc_serializer.get_parts(
            item=item,
            traverse_pictures=True,
            visited=visited,
        )
        text_res = (separator or " ").join([p.text for p in parts])
        # NOTE: we do no postprocessing since already done as needed
        return SerializationResult(text=text_res)

    # helper function
    def _serialize_annotations(
        self,
        item: PictureItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs,
    ) -> SerializationResult:
        text_parts: list[str] = []
        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                predicted_class = (
                    annotation.predicted_classes[0].class_name
                    if annotation.predicted_classes
                    else None
                )
                if predicted_class is not None:
                    text_parts.append(f"Picture type: {predicted_class}")
            elif isinstance(annotation, PictureMoleculeData):
                text_parts.append(f"SMILES: {annotation.smi}")
            elif isinstance(annotation, PictureDescriptionData):
                text_parts.append(f"Description: {annotation.text}")

        text_res = (separator or "\n").join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return SerializationResult(text=text_res)
