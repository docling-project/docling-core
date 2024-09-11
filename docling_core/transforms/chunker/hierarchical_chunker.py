#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Iterator, Optional, Union

import pandas as pd
from pydantic import BaseModel, PositiveInt

from docling_core.transforms.chunker import BaseChunker, Chunk, ChunkWithMetadata
from docling_core.types import BaseText
from docling_core.types import Document as DLDocument
from docling_core.types import Ref, Table

_logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """Chunker implementation leveraging the document layout."""

    include_metadata: bool = True
    min_chunk_len: PositiveInt = 64

    class _NodeType(str, Enum):
        PARAGRAPH = "paragraph"
        SUBTITLE_LEVEL_1 = "subtitle-level-1"
        TABLE = "table"
        CAPTION = "caption"

    class _NodeName(str, Enum):
        TITLE = "title"
        REFERENCE = "reference"
        LIST_ITEM = "list-item"
        SUBTITLE_LEVEL_1 = "subtitle-level-1"

    _allowed_types: list[str] = [
        _NodeType.PARAGRAPH,
        _NodeType.SUBTITLE_LEVEL_1,
        _NodeType.TABLE,
        _NodeType.CAPTION,
    ]
    _disallowed_names_by_type: dict[str, list[str]] = {
        _NodeType.PARAGRAPH: [
            _NodeName.REFERENCE,
        ],
    }

    @classmethod
    def _norm(cls, text: Optional[str]) -> Optional[str]:
        return text.lower() if text is not None else None

    @classmethod
    def _convert_table_to_dataframe(cls, table: Table) -> Optional[pd.DataFrame]:
        if table.data:
            table_content = [[cell.text for cell in row] for row in table.data]
            return pd.DataFrame(table_content)
        else:
            return None

    @classmethod
    def _triplet_serialize(cls, table) -> Optional[str]:
        output_text: Optional[str] = None
        table_df = cls._convert_table_to_dataframe(table)
        if table_df is not None and table_df.shape[0] > 1 and table_df.shape[1] > 1:
            rows = [item.strip() for item in table_df.iloc[:, 0].to_list()]
            cols = [item.strip() for item in table_df.iloc[0, :].to_list()]
            nrows = table_df.shape[0]
            ncols = table_df.shape[1]
            texts = [
                f"{rows[i]}, {cols[j]} = {table_df.iloc[i, j].strip()}"
                for i in range(1, nrows)
                for j in range(1, ncols)
            ]
            output_text = ". ".join(texts)

        return output_text

    @classmethod
    def _create_path(cls, pos: int, path_prefix: str = "main-text") -> str:
        return f"$.{path_prefix}[{pos}]"

    class _MainTextItemNode(BaseModel):
        parent: Optional[int] = None
        children: list[int] = []

    class _TitleInfo(BaseModel):
        text: str
        path_in_doc: str

    class _GlobalContext(BaseModel):
        title: Optional[_HC._TitleInfo] = None

    class _DocContext(BaseModel):
        dmap: dict[int, _HC._MainTextItemNode]  # main text element context
        glob: _HC._GlobalContext  # global context

        @classmethod
        def from_doc(cls, doc: DLDocument) -> _HC._DocContext:
            dmap: dict[int, _HC._MainTextItemNode] = {}
            glob: _HC._GlobalContext = _HC._GlobalContext()
            if doc.description.title:
                glob.title = _HC._TitleInfo(
                    text=doc.description.title,
                    path_in_doc="description.title",
                )

            parent = None
            if doc.main_text:
                idx = 0
                while idx < len(doc.main_text):
                    item = doc.main_text[idx]
                    if (
                        not glob.title
                        and isinstance(item, BaseText)
                        and _HC._norm(item.name) == _HC._NodeName.TITLE
                    ):
                        glob.title = _HC._TitleInfo(
                            text=item.text,
                            path_in_doc=_HC._create_path(idx),
                        )

                    # start of a subtitle-level-1 parent
                    if (
                        isinstance(item, BaseText)
                        and _HC._norm(item.obj_type) == _HC._NodeType.SUBTITLE_LEVEL_1
                    ):
                        dmap[idx] = _HC._MainTextItemNode(parent=None)
                        parent = idx
                        if not glob.title:
                            glob.title = _HC._TitleInfo(
                                text=item.text,
                                path_in_doc=_HC._create_path(idx),
                            )

                    # start of a list parent
                    elif (
                        isinstance(item, BaseText)
                        and _HC._norm(item.name) != _HC._NodeName.LIST_ITEM
                        and idx + 1 < len(doc.main_text)
                        and _HC._norm(doc.main_text[idx + 1].name)
                        == _HC._NodeName.LIST_ITEM
                    ):
                        if parent is not None:
                            dmap[parent].children.append(idx)
                        dmap[idx] = _HC._MainTextItemNode(parent=parent)

                        # have all children register locally
                        li = idx + 1
                        while (
                            li < len(doc.main_text)
                            and _HC._norm(doc.main_text[li].name)
                            == _HC._NodeName.LIST_ITEM
                        ):
                            dmap[idx].children.append(li)
                            dmap[li] = _HC._MainTextItemNode(parent=idx)
                            li += 1
                        idx = li
                        continue

                    # normal case
                    else:
                        if parent is not None:
                            dmap[parent].children.append(idx)
                        dmap[idx] = _HC._MainTextItemNode(parent=parent)

                    idx += 1
            else:
                pass
            return cls(
                dmap=dmap,
                glob=glob,
            )

    class _TextEntry(BaseModel):
        text: str
        path: str

    def _build_chunk_impl(
        self, doc: DLDocument, doc_map: _DocContext, idx: int, rec: bool = False
    ) -> list[_TextEntry]:
        if doc.main_text:
            item = doc.main_text[idx]
            item_type = _HC._norm(item.obj_type)
            item_name = _HC._norm(item.name)
            if (
                item_type not in self._allowed_types
                or item_name in self._disallowed_names_by_type.get(item_type, [])
            ):
                return []

            c2p = doc_map.dmap

            text_entries: list[_HC._TextEntry] = []
            if (
                isinstance(item, Ref)
                and item_type == _HC._NodeType.TABLE
                and doc.tables
            ):
                # resolve table reference
                ref_nr = int(item.ref.split("/")[2])  # e.g. '#/tables/0'
                table = doc.tables[ref_nr]
                ser_out = _HC._triplet_serialize(table)
                if table.data:
                    text_entries = (
                        [
                            self._TextEntry(
                                text=ser_out,
                                path=self._create_path(idx),
                            )
                        ]
                        if ser_out
                        else []
                    )
                else:
                    return []
            elif isinstance(item, BaseText):
                text_entries = [
                    self._TextEntry(
                        text=item.text,
                        path=self._create_path(idx),
                    )
                ]

            # squash in any children of type list-item
            if not rec:
                if (
                    c2p[idx].children
                    and _HC._norm(doc.main_text[c2p[idx].children[0]].name)
                    == _HC._NodeName.LIST_ITEM
                ):
                    text_entries = text_entries + [
                        self._TextEntry(
                            text=doc.main_text[c].text,  # type: ignore[union-attr]
                            path=self._create_path(c),
                        )
                        for c in c2p[idx].children
                        if isinstance(doc.main_text[c], BaseText)
                        and _HC._norm(doc.main_text[c].name) == _HC._NodeName.LIST_ITEM
                    ]
                elif item_name in [
                    _HC._NodeName.LIST_ITEM,
                    _HC._NodeName.SUBTITLE_LEVEL_1,
                ]:
                    return []

            if (parent := c2p[idx].parent) is not None:
                # prepend with ancestors
                return (
                    self._build_chunk_impl(
                        doc=doc, doc_map=doc_map, idx=parent, rec=True
                    )
                    + text_entries
                )
            else:
                # if root, augment with title (if available and different)
                return text_entries
        else:
            return []

    def _build_chunk(
        self,
        doc: DLDocument,
        doc_map: _DocContext,
        idx: int,
        delim: str,
        rec: bool = False,
    ) -> Optional[Chunk]:
        texts = self._build_chunk_impl(doc=doc, doc_map=doc_map, idx=idx, rec=rec)
        concat = delim.join([t.text for t in texts if t.text])
        assert doc.main_text is not None
        if len(concat) >= self.min_chunk_len:
            orig_item = doc.main_text[idx]
            item: Union[BaseText, Table]
            if isinstance(orig_item, Ref):
                if _HC._norm(orig_item.obj_type) == _HC._NodeType.TABLE and doc.tables:
                    pos = int(orig_item.ref.split("/")[2])
                    item = doc.tables[pos]
                    path = self._create_path(pos, path_prefix="tables")
                else:  # currently disregarding non-table references
                    return None
            else:
                item = orig_item
                path = self._create_path(idx)

            if self.include_metadata:
                return ChunkWithMetadata(
                    text=concat,
                    path=path,
                    page=item.prov[0].page if item.prov else None,
                    bbox=item.prov[0].bbox if item.prov else None,
                )
            else:
                return Chunk(
                    text=concat,
                    path=path,
                )
        else:
            return None

    def chunk(self, dl_doc: DLDocument, delim="\n", **kwargs: Any) -> Iterator[Chunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk
            delim (str, optional): delimiter to use when concatenating sub-items.
                Defaults to "\n".

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        if dl_doc.main_text:
            # extract doc structure incl. metadata for
            # each item (e.g. parent, children)
            doc_ctx = self._DocContext.from_doc(doc=dl_doc)
            _logger.debug(f"{doc_ctx.model_dump()=}")

            for i, item in enumerate(dl_doc.main_text):
                if (
                    isinstance(item, BaseText)
                    or _HC._norm(item.obj_type) == _HC._NodeType.TABLE
                ):
                    chunk = self._build_chunk(
                        doc=dl_doc, doc_map=doc_ctx, idx=i, delim=delim
                    )
                    if chunk:
                        _logger.info(f"{i=}, {chunk=}")
                        yield chunk


_HC = HierarchicalChunker
