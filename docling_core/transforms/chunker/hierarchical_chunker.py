"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Optional, Union

import pandas as pd
from pydantic import ConfigDict, Field
from typing_extensions import override

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
from docling_core.transforms.serializer.common import DocSerializer, create_ser_result
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
            table_df = item._export_to_dataframe_with_options(
                doc,
                doc_serializer=doc_serializer,
                **kwargs,
            )
            if table_df.shape[0] >= 1 and table_df.shape[1] >= 1:
                # Detect real headers from the table cell metadata
                # rather than assuming positional conventions.
                has_col_headers = not isinstance(table_df.columns, pd.RangeIndex)
                has_row_headers = any(cell.row_header for row in item.data.grid for cell in row)

                table_text_parts = self._build_triplets(
                    table_df,
                    has_col_headers=has_col_headers,
                    has_row_headers=has_row_headers,
                )
                table_text = ". ".join(table_text_parts)
                parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in parts])

        return create_ser_result(text=text_res, span_source=parts)

    @staticmethod
    def _build_triplets(
        table_df: pd.DataFrame,
        *,
        has_col_headers: bool,
        has_row_headers: bool,
    ) -> list[str]:
        table_df = table_df.copy()

        nrows, ncols = table_df.shape

        # Both row and column headers — use them directly.
        if has_col_headers and has_row_headers:
            row_headers = [str(v).strip() for v in table_df.iloc[:, 0]]
            col_headers = [str(c).strip() for c in table_df.columns]

            return [
                f"{row_headers[i]}, {col_headers[j]} = {str(table_df.iloc[i, j]).strip()}"
                for i in range(nrows)
                for j in range(1, ncols)
            ]

        # Column headers only — single column.
        if has_col_headers and ncols == 1:
            col_name = str(table_df.columns[0]).strip()
            return [f"{col_name} = {str(table_df.iloc[i, 0]).strip()}" for i in range(nrows)]

        # Column headers only — multi-column.
        # First column values serve as informal row identifiers.
        if has_col_headers:
            table_df.loc[-1] = list(table_df.columns)  # type: ignore[call-overload]
            table_df.index = table_df.index + 1
            table_df = table_df.sort_index()

            rows = [str(v).strip() for v in table_df.iloc[:, 0]]
            cols = [str(v).strip() for v in table_df.iloc[0, :]]

            nrows, ncols = table_df.shape

            return [
                f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
                for i in range(1, nrows)
                for j in range(1, ncols)
            ]

        # Row headers only — no column names available.
        if has_row_headers:
            rows = [str(v).strip() for v in table_df.iloc[:, 0]]
            return [
                f"{rows[i]}, Column {j} = {str(table_df.iloc[i, j]).strip()}"
                for i in range(nrows)
                for j in range(1, ncols)
            ]

        # No headers at all — positional notation.
        return [f"Row {i}, Column {j} = {str(table_df.iloc[i, j]).strip()}" for i in range(nrows) for j in range(ncols)]


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
        always_emit_headings (bool): Whether to emit headings even for empty sections. Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()
    code_chunking_strategy: Optional[BaseCodeChunkingStrategy] = Field(default=None)
    always_emit_headings: bool = False

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
        heading_by_level: dict[LevelNumber, Union[TitleItem, SectionHeaderItem]] = {}
        heading_emitted: set[str] = set()
        visited: set[str] = set()
        ser_res = create_ser_result()
        excluded_refs = my_doc_ser.get_excluded_refs(**kwargs)
        traverse_pictures = my_doc_ser.params.traverse_pictures if isinstance(my_doc_ser, DocSerializer) else False
        for item, level in dl_doc.iterate_items(
            with_groups=True,
            traverse_pictures=traverse_pictures,
        ):
            if item.self_ref in excluded_refs:
                continue
            if isinstance(item, TitleItem | SectionHeaderItem):
                level = item.level if isinstance(item, SectionHeaderItem) else 0

                # prepare to remove shadowed headings as they just went out of scope
                sorted_keys = sorted(heading_by_level)
                keys_to_del = [k for k in sorted_keys if k >= level]

                # before removing, check if headings need to be emitted
                if (
                    keys_to_del
                    and self.always_emit_headings
                    and (leaf_ref := heading_by_level[sorted_keys[-1]].self_ref) not in heading_emitted
                ):
                    yield DocChunk(
                        text="",
                        meta=DocMeta(
                            doc_items=[heading_by_level[k] for k in sorted_keys],
                            headings=[heading_by_level[k].text for k in sorted_keys],
                        ),
                    )
                    heading_emitted.add(leaf_ref)

                # actually remove shadowed headings
                for k in keys_to_del:
                    heading_by_level.pop(k, None)

                # capture current heading
                heading_by_level[level] = item

                continue
            elif isinstance(item, ListGroup | InlineGroup | DocItem) and item.self_ref not in visited:
                if self.code_chunking_strategy is not None and isinstance(item, CodeItem):
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
                sorted_keys = sorted(heading_by_level)
                headings = [heading_by_level[k].text for k in sorted_keys] or None
                c = DocChunk(
                    text=ser_res.text,
                    meta=DocMeta(
                        doc_items=doc_items,
                        headings=headings,
                        origin=dl_doc.origin,
                    ),
                )
                if self.always_emit_headings and headings:
                    leaf_ref = heading_by_level[sorted_keys[-1]].self_ref
                    heading_emitted.add(leaf_ref)
                yield c

        # if applicable, emit any remaining headings
        if (
            self.always_emit_headings
            and (sorted_keys := sorted(heading_by_level))
            and ((leaf_ref := heading_by_level[sorted_keys[-1]].self_ref) not in heading_emitted)
        ):
            yield DocChunk(
                text="",
                meta=DocMeta(
                    doc_items=[heading_by_level[k] for k in sorted_keys],
                    headings=[heading_by_level[k].text for k in sorted_keys],
                ),
            )
            heading_emitted.add(leaf_ref)
