#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar, Final, Iterator, Literal, Optional

from pydantic import ConfigDict, Field, StringConstraints, field_validator
from typing_extensions import Annotated, override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
    BaseTableSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import create_ser_result
from docling_core.experimental.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    DocumentOrigin,
    InlineGroup,
    LevelNumber,
    OrderedList,
    SectionHeaderItem,
    TableItem,
    TitleItem,
    UnorderedList,
)

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
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        Field(
            default=_VERSION,
            alias=_KEY_VERSION,
        )
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


class TripletTableSerializer(BaseTableSerializer):
    """Triplet-based table item serializer."""

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
        parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            table_df = item.export_to_dataframe()
            if table_df.shape[0] >= 1 and table_df.shape[1] >= 2:

                # copy header as first row and shift all rows by one
                table_df.loc[-1] = table_df.columns  # type: ignore[call-overload]
                table_df.index = table_df.index + 1
                table_df = table_df.sort_index()

                rows = [str(item).strip() for item in table_df.iloc[:, 0].to_list()]
                cols = [str(item).strip() for item in table_df.iloc[0, :].to_list()]

                nrows = table_df.shape[0]
                ncols = table_df.shape[1]
                table_text_parts = [
                    f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
                    for i in range(1, nrows)
                    for j in range(1, ncols)
                ]
                table_text = ". ".join(table_text_parts)
                parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in parts])

        return create_ser_result(text=text_res, span_source=parts)


class ChunkingDocSerializer(MarkdownDocSerializer):
    """Doc serializer used for chunking purposes."""

    table_serializer: BaseTableSerializer = TripletTableSerializer()
    params: MarkdownParams = MarkdownParams(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        escape_underscores=False,
        escape_html=False,
    )


class ChunkingSerializerProvider(BaseSerializerProvider):
    """Serializer provider used for chunking purposes."""

    @override
    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        """Get the associated serializer."""
        return ChunkingDocSerializer(doc=doc)


class HierarchicalChunker(BaseChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()

    # deprecated:
    merge_list_items: Annotated[bool, Field(deprecated=True)] = True

    def chunk(
        self,
        dl_doc: DLDocument,
        **kwargs: Any,
    ) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        heading_by_level: dict[LevelNumber, str] = {}
        visited: set[str] = set()
        ser_res = create_ser_result()
        excluded_refs = my_doc_ser.get_excluded_refs(**kwargs)
        for item, level in dl_doc.iterate_items(with_groups=True):
            if item.self_ref in excluded_refs:
                continue
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                level = item.level if isinstance(item, SectionHeaderItem) else 0
                heading_by_level[level] = item.text

                # remove headings of higher level as they just went out of scope
                keys_to_del = [k for k in heading_by_level if k > level]
                for k in keys_to_del:
                    heading_by_level.pop(k, None)
                continue
            elif (
                isinstance(item, (OrderedList, UnorderedList, InlineGroup, DocItem))
                and item.self_ref not in visited
            ):
                ser_res = my_doc_ser.serialize(item=item, visited=visited)
            else:
                continue

            if not ser_res.text:
                continue
            if doc_items := [u.item for u in ser_res.spans]:
                c = DocChunk(
                    text=ser_res.text,
                    meta=DocMeta(
                        doc_items=doc_items,
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        origin=dl_doc.origin,
                    ),
                )
                yield c
