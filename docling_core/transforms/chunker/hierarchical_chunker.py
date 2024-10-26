#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar, Final, Iterator, Literal, Optional

from pandas import DataFrame
from pydantic import Field, StringConstraints, field_validator
from typing_extensions import Annotated

from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    DocItem,
    DocumentOrigin,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

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
    captions: Optional[list[str]] = Field(
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
    """Data model for Hierarchical Chunker chunks."""

    meta: DocMeta


class HierarchicalChunker(BaseChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    merge_list_items: bool = True
    delim: str = "\n"

    @classmethod
    def _triplet_serialize(cls, table_df: DataFrame) -> str:

        # copy header as first row and shift all rows by one
        table_df.loc[-1] = table_df.columns  # type: ignore[call-overload]
        table_df.index = table_df.index + 1
        table_df = table_df.sort_index()

        rows = [str(item).strip() for item in table_df.iloc[:, 0].to_list()]
        cols = [str(item).strip() for item in table_df.iloc[0, :].to_list()]

        nrows = table_df.shape[0]
        ncols = table_df.shape[1]
        texts = [
            f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
            for i in range(1, nrows)
            for j in range(1, ncols)
        ]
        output_text = ". ".join(texts)

        return output_text

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        for item, level in dl_doc.iterate_items():
            captions = None
            if isinstance(item, DocItem):

                # first handle any merging needed
                if self.merge_list_items:
                    if isinstance(
                        item, ListItem
                    ) or (  # TODO remove when all captured as ListItem:
                        isinstance(item, TextItem)
                        and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        continue
                    elif list_items:  # need to yield
                        yield DocChunk(
                            text=self.delim.join([i.text for i in list_items]),
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[
                                    heading_by_level[k]
                                    for k in sorted(heading_by_level)
                                ]
                                or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []  # reset

                if isinstance(
                    item, SectionHeaderItem
                ) or (  # TODO remove when all captured as SectionHeaderItem:
                    isinstance(item, TextItem)
                    and item.label == DocItemLabel.SECTION_HEADER
                ):
                    # TODO second branch not needed once cleanup above complete:
                    level = item.level if isinstance(item, SectionHeaderItem) else 1
                    heading_by_level[level] = item.text

                    # remove headings of higher level as they just went out of scope
                    keys_to_del = [k for k in heading_by_level if k > level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    continue

                if isinstance(item, TextItem) or (
                    (not self.merge_list_items) and isinstance(item, ListItem)
                ):
                    text = item.text
                elif isinstance(item, TableItem):
                    table_df = item.export_to_dataframe()
                    if table_df.shape[0] < 1 or table_df.shape[1] < 2:
                        # at least two cols needed, as first column contains row headers
                        continue
                    text = self._triplet_serialize(table_df=table_df)
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                else:
                    continue
                c = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

        if self.merge_list_items and list_items:  # need to yield
            yield DocChunk(
                text=self.delim.join([i.text for i in list_items]),
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                    or None,
                    origin=dl_doc.origin,
                ),
            )
