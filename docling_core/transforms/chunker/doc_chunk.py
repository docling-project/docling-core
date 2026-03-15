"""Data model for chunk metadata."""

from __future__ import annotations

import logging
import re
from copy import copy
from typing import Annotated, ClassVar, Final, Literal, Optional

from pydantic import Field, StringConstraints, field_validator

from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseMeta
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.serializer.common import CommonParams
from docling_core.types.doc.document import DocItem, DoclingDocument, DocumentOrigin, InlineGroup, ListGroup, RefItem

_VERSION: Final = "1.0.0"

_KEY_SCHEMA_NAME = "schema_name"
_KEY_VERSION = "version"
_KEY_DOC_ITEMS = "doc_items"
_KEY_HEADINGS = "headings"
_KEY_CAPTIONS = "captions"
_KEY_ORIGIN = "origin"

_logger = logging.getLogger(__name__)


class DocMeta(BaseMeta):
    """Data model for Hierarchical Chunker chunk metadata."""

    schema_name: Literal["docling_core.transforms.chunker.DocMeta"] = Field(
        default="docling_core.transforms.chunker.DocMeta",
        alias=_KEY_SCHEMA_NAME,
    )
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = Field(
        default=_VERSION,
        alias=_KEY_VERSION,
    )
    doc_items: list[DocItem] = Field(
        alias=_KEY_DOC_ITEMS,
        min_length=1,
    )
    headings: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_HEADINGS,
        min_length=1,
    )
    captions: Optional[list[str]] = Field(  # deprecated
        deprecated=True,
        default=None,
        alias=_KEY_CAPTIONS,
        min_length=1,
    )
    origin: Optional[DocumentOrigin] = Field(
        default=None,
        alias=_KEY_ORIGIN,
    )

    excluded_embed: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]
    excluded_llm: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]

    @field_validator(_KEY_VERSION)
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this meta item version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, _VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(f"incompatible version {v} with schema version {_VERSION}")
        else:
            return _VERSION


class DocChunk(BaseChunk):
    """Data model for document chunks."""

    meta: DocMeta

    def get_top_containing_objects(self, doc: DoclingDocument) -> list[DocItem] | None:
        objects = {}
        ref_items = [item.self_ref for item in self.meta.doc_items]
        for item in ref_items:
            # traverse document tree till top level (body)
            obj = RefItem(cref=item).resolve(doc)
            while obj.parent != doc.body.get_ref():
                obj = obj.parent.resolve(doc)
            objects[obj.self_ref] = obj

        # maintain the reading order as in the original document
        doc_body_refs = [ref.cref for ref in doc.body.children]
        doc_ordered_refs = [ref for ref in doc_body_refs if ref in objects]
        if len(doc_ordered_refs) > 0:
            return [objects[ref] for ref in doc_ordered_refs]
        return None

    def expand_to_object(self, dl_doc: DoclingDocument, serializer: BaseDocSerializer) -> DocChunk:
        top_objects = self.get_top_containing_objects(dl_doc)
        if not top_objects:
            _logger.warning(f"error in getting top objects of {self}")
            return self

        content = ""
        doc_items = []

        for top_object in top_objects:
            if isinstance(top_object, ListGroup | InlineGroup | DocItem):
                try:
                    ser_res = serializer.serialize(item=top_object)
                    content += ser_res.text + " "
                    doc_items.append(top_object)

                except Exception as e:
                    _logger.warning(f"error in extacting text of {top_object}: {e}")
        if len(content.strip()) == 0:
            _logger.warning(f"expansion of {self} did not yield any text")
            return self

        # fix me: update meta.headings

        meta = copy(self.meta)
        meta.doc_items = doc_items
        return DocChunk(
            text=content,
            meta=self.meta,
        )

    def expand_to_page(self, doc: DoclingDocument, serializer: BaseDocSerializer) -> DocChunk | None:
        page_ids = [i.page_no for item in self.meta.doc_items for i in item.prov]
        ser_params: CommonParams | None = getattr(serializer, "params", None)
        if len(doc.pages) == 0 or page_ids is None or len(page_ids) == 0 or not ser_params:
            _logger.warning(f"cannot expand to page the following chunk: {self}")
            return self

        ser_params.pages = page_ids
        pages_content = serializer.serialize().text

        return DocChunk(
            text=pages_content,
            meta=self.meta,
        )
