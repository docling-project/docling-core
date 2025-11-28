"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from pydantic import ConfigDict, Field
from typing_extensions import Annotated, override

from docling_core.transforms.chunker import BaseChunk, BaseChunker
from docling_core.transforms.chunker.code_chunking.base_code_chunking_strategy import (
    BaseCodeChunkingStrategy,
)
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
    BaseTableSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    InlineGroup,
    LevelNumber,
    ListGroup,
    SectionHeaderItem,
    TableItem,
    TitleItem,
)

_logger = logging.getLogger(__name__)


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
            table_df = item.export_to_dataframe(doc)
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
        code_chunking_strategy (CodeChunkingStrategy): Optional strategy for chunking code items.
            If provided, code items will be processed using this strategy instead of being
            treated as regular text. Defaults to None (no special code processing).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()
    code_chunking_strategy: Optional[BaseCodeChunkingStrategy] = Field(default=None)

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
                isinstance(item, (ListGroup, InlineGroup, DocItem))
                and item.self_ref not in visited
            ):
                if self.code_chunking_strategy is not None and isinstance(
                    item, CodeItem
                ):
                    yield from self.code_chunking_strategy.chunk_code_item(
                        item=item,
                        doc=dl_doc,
                        doc_serializer=my_doc_ser,
                        visited=visited,
                        **kwargs,
                    )
                    continue

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
