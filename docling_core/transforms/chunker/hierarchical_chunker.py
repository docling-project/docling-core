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

        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result(
                text="\n\n".join([r.text for r in parts]),
                span_source=parts,
            )

        table_df = item._export_to_dataframe_with_options(
            doc,
            doc_serializer=doc_serializer,
            **kwargs,
        )

        if table_df.shape[0] == 0 or table_df.shape[1] == 0:
            return create_ser_result(
                text="\n\n".join([r.text for r in parts]),
                span_source=parts,
            )

        # Header Detection (metadata-first, fallback-safe)
        has_row_headers = any(getattr(cell, "row_header", False) for row in item.data.grid for cell in row)

        has_col_headers = any(getattr(cell, "col_header", False) for row in item.data.grid for cell in row)

        # Fallback if metadata missing
        if not has_col_headers:
            has_col_headers = not isinstance(table_df.columns, pd.RangeIndex) and any(
                str(c).strip() for c in table_df.columns
            )

        # Build Triplets
        triplets = self._build_triplets(
            table_df,
            has_row_headers=has_row_headers,
            has_col_headers=has_col_headers,
        )

        if triplets:
            table_text = ". ".join(triplets)
            parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in parts])

        return create_ser_result(text=text_res, span_source=parts)

    @staticmethod
    def _build_triplets(
        table_df: pd.DataFrame,
        *,
        has_row_headers: bool,
        has_col_headers: bool,
    ) -> list[str]:
        """
        Convert table into triplets of form:
            (row_label, col_label) = value

        Guarantees:
        - No mutation of input DataFrame
        - Consistent schema across all cases
        - No implicit structural assumptions
        """

        table_data_df = table_df.copy()
        nrows, ncols = table_data_df.shape

        # Extract headers safely
        row_headers: Optional[list[str]] = None
        if has_row_headers:
            row_headers = [str(v).strip() for v in table_data_df.iloc[:, 0]]

        col_headers: Optional[list[str]] = None
        data_start_row = 0

        if has_col_headers:
            if isinstance(table_data_df.columns, pd.RangeIndex) and nrows > 0:
                # Some exporters keep the header row in the table body when
                # DataFrame columns are still a RangeIndex.
                col_headers = [str(v).strip() for v in table_data_df.iloc[0, :]]
                data_start_row = 1
            else:
                col_headers = [str(c).strip() for c in table_data_df.columns]

        # Label helpers
        def get_row_label(i: int) -> str:
            if row_headers:
                return row_headers[i] or f"row_{i}"
            return f"row_{i}"

        def get_col_label(j: int) -> str:
            if col_headers:
                return col_headers[j] or f"col_{j}"
            return f"col_{j}"

        triplets: list[str] = []

        # For single-column tables, emit "column = value" only when a real
        # column header is detected; otherwise fall back to generic triplets.
        if ncols == 1 and nrows > 0 and has_col_headers and not has_row_headers:
            col_name = col_headers[0] if col_headers else "col_0"
            col_name = col_name or "col_0"
            # Header is sourced from DataFrame columns, so all rows are values.
            for value in table_data_df.iloc[:, 0].to_list():
                if pd.isna(value) or str(value).strip() == "":
                    continue
                triplets.append(f"{col_name} = {str(value).strip()}")
            if triplets:
                return triplets

        # Main extraction loop
        for i in range(data_start_row, nrows):
            for j in range(ncols):
                # Skip header cells themselves
                if has_row_headers and j == 0:
                    continue
                value = table_data_df.iloc[i, j]

                # Skip empty / NaN cells (important for RAG quality)
                if pd.isna(value) or str(value).strip() == "":
                    continue

                row_label = get_row_label(i)
                col_label = get_col_label(j)

                triplets.append(f"{row_label}, {col_label} = {str(value).strip()}")

        return triplets


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
