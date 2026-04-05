"""Hybrid chunker implementation leveraging both doc structure & token awareness."""

import warnings
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Any, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.types.doc.document import SectionHeaderItem, TableItem, TitleItem

try:
    import semchunk
    from transformers import PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Extra required by module: 'chunking' by default (or 'chunking-openai' if "
        "specifically using OpenAI tokenization); to install, run: "
        "`pip install 'docling-core[chunking]'` or "
        "`pip install 'docling-core[chunking-openai]'`"
    )

from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.transforms.serializer.base import (
    BaseSerializerProvider,
)
from docling_core.types import DoclingDocument


class HybridChunker(BaseChunker):
    r"""Chunker doing tokenization-aware refinements on top of document layout chunking.

    Args:
        tokenizer: The tokenizer to use; either instantiated object or name or path of
            respective pretrained model
        max_tokens: The maximum number of tokens per chunk. If not set, limit is
            resolved from the tokenizer
        repeat_table_headers: Whether to repeat a table header if the table is chunked
        merge_peers: Whether to merge undersized chunks sharing same relevant metadata
        always_emit_headings: Whether to emit headings even for empty sections
        omit_header_on_overflow: Only used when repeat_table_header is True. When True,
            omit table headers for rows that would overflow with them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: BaseTokenizer = Field(default_factory=get_default_tokenizer)
    repeat_table_header: bool = True
    merge_peers: bool = True
    omit_header_on_overflow: bool = False

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()
    always_emit_headings: bool = False

    @model_validator(mode="before")
    @classmethod
    def _patch(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tokenizer = data.get("tokenizer")
            max_tokens = data.get("max_tokens")
            if not isinstance(tokenizer, BaseTokenizer) and (
                # some legacy param passed:
                tokenizer is not None or max_tokens is not None
            ):
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )

                warnings.warn(
                    "Deprecated initialization parameter types for HybridChunker. "
                    "For updated usage check out "
                    "https://docling-project.github.io/docling/examples/hybrid_chunking/",
                    DeprecationWarning,
                )

                if isinstance(tokenizer, str):
                    data["tokenizer"] = HuggingFaceTokenizer.from_pretrained(
                        model_name=tokenizer,
                        max_tokens=max_tokens,
                    )
                elif tokenizer is None or isinstance(tokenizer, PreTrainedTokenizerBase):
                    kwargs = {"tokenizer": tokenizer or get_default_tokenizer().tokenizer}
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    data["tokenizer"] = HuggingFaceTokenizer(**kwargs)
        return data

    @property
    def max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        return self.tokenizer.get_max_tokens()

    @computed_field  # type: ignore[misc]
    @cached_property
    def _inner_chunker(self) -> HierarchicalChunker:
        return HierarchicalChunker(
            serializer_provider=self.serializer_provider,
            always_emit_headings=self.always_emit_headings,
        )

    def _count_text_tokens(self, text: Optional[Union[str, list[str]]]):
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self._count_text_tokens(t)
            return total
        return self.tokenizer.count_tokens(text=text)

    class _ChunkLengthInfo(BaseModel):
        total_len: int
        text_len: int
        other_len: int

    def _count_chunk_tokens(self, doc_chunk: DocChunk):
        ser_txt = self.contextualize(chunk=doc_chunk)
        return self.tokenizer.count_tokens(text=ser_txt)

    def _estimate_tokens_for_items(
        self,
        doc_items: list,
        doc_serializer: BaseDocSerializer,
    ) -> list[int]:
        """Pre-compute token counts for each doc item's serialized text.

        The sum of individual token counts is always >= the token count of the
        concatenated text (BPE can merge tokens across boundaries), so this
        provides a safe upper-bound estimate for fast-path acceptance.
        """
        counts = []
        for doc_item in doc_items:
            text = doc_serializer.serialize(item=doc_item).text
            if text and not isinstance(doc_item, (TitleItem, SectionHeaderItem)):
                counts.append(self.tokenizer.count_tokens(text=text))
            else:
                counts.append(0)
        return counts

    def _doc_chunk_length(self, doc_chunk: DocChunk):
        text_length = self._count_text_tokens(doc_chunk.text)
        total = self._count_chunk_tokens(doc_chunk=doc_chunk)
        return self._ChunkLengthInfo(
            total_len=total,
            text_len=text_length,
            other_len=total - text_length,
        )

    def _make_chunk_from_doc_items(
        self,
        doc_chunk: DocChunk,
        window_start: int,
        window_end: int,
        doc_serializer: BaseDocSerializer,
    ):
        doc_items = doc_chunk.meta.doc_items[window_start : window_end + 1]
        meta = DocMeta(
            doc_items=doc_items,
            headings=doc_chunk.meta.headings,
            origin=doc_chunk.meta.origin,
        )
        window_text = (
            doc_chunk.text
            if len(doc_chunk.meta.doc_items) == 1
            # TODO: merging should ideally be done by the serializer:
            else self.delim.join(
                [
                    res_text
                    for doc_item in doc_items
                    if (res_text := doc_serializer.serialize(item=doc_item).text)
                    and not isinstance(doc_item, TitleItem | SectionHeaderItem)
                ]
            )
        )
        new_chunk = DocChunk(text=window_text, meta=meta)
        return new_chunk

    def _split_by_doc_items(self, doc_chunk: DocChunk, doc_serializer: BaseDocSerializer) -> list[DocChunk]:
        chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_items = len(doc_chunk.meta.doc_items)

        # Pre-compute per-item token counts for fast-path estimation.
        # Since sum-of-parts >= exact tokenization (BPE merges across
        # boundaries), estimate <= budget implies exact <= budget.
        item_token_counts = self._estimate_tokens_for_items(
            doc_items=doc_chunk.meta.doc_items,
            doc_serializer=doc_serializer,
        )
        overhead_estimate = 0
        delim_tokens = 0
        if num_items > 0:
            probe_chunk = self._make_chunk_from_doc_items(
                doc_chunk=doc_chunk,
                window_start=0,
                window_end=0,
                doc_serializer=doc_serializer,
            )
            probe_total = self._count_chunk_tokens(doc_chunk=probe_chunk)
            overhead_estimate = probe_total - item_token_counts[0]
            delim_tokens = self.tokenizer.count_tokens(text=self.delim) if self.delim else 0
        running_text_tokens = 0

        while window_end < num_items:
            # Update incremental token estimate (always >= exact count)
            if window_end == window_start:
                running_text_tokens = item_token_counts[window_end]
            else:
                running_text_tokens += item_token_counts[window_end] + delim_tokens

            estimated_total = overhead_estimate + running_text_tokens

            # Fast path: estimate is an upper bound, so if estimate <= budget,
            # exact count is also <= budget. Safe to skip expensive tokenization.
            if estimated_total <= self.max_tokens and window_end < num_items - 1:
                window_end += 1
                continue

            # Near boundary or at end: use exact tokenization (original logic)
            new_chunk = self._make_chunk_from_doc_items(
                doc_chunk=doc_chunk,
                window_start=window_start,
                window_end=window_end,
                doc_serializer=doc_serializer,
            )
            if self._count_chunk_tokens(doc_chunk=new_chunk) <= self.max_tokens:
                if window_end < num_items - 1:
                    window_end += 1
                    # Still room left to add more to this chunk AND still at least one
                    # item left
                    continue
                else:
                    # All the items in the window fit into the chunk and there are no
                    # other items left
                    window_end = num_items  # signalizing the last loop
            elif window_start == window_end:
                # Only one item in the window and it doesn't fit into the chunk. So
                # we'll just make it a chunk for now and it will get split in the
                # plain text splitter.
                window_end += 1
                window_start = window_end
                running_text_tokens = 0
            else:
                # Multiple items in the window but they don't fit into the chunk.
                # However, the existing items must have fit or we wouldn't have
                # gotten here. So we put everything but the last item into the chunk
                # and then start a new window INCLUDING the current window end.
                new_chunk = self._make_chunk_from_doc_items(
                    doc_chunk=doc_chunk,
                    window_start=window_start,
                    window_end=window_end - 1,
                    doc_serializer=doc_serializer,
                )
                window_start = window_end
                running_text_tokens = 0
            chunks.append(new_chunk)
        return chunks

    def _split_using_plain_text(
        self,
        doc_chunk: DocChunk,
        doc_serializer: BaseDocSerializer,
    ) -> list[DocChunk]:
        lengths = self._doc_chunk_length(doc_chunk)
        if lengths.total_len <= self.max_tokens:
            return [DocChunk(**doc_chunk.export_json_dict())]
        else:
            # How much room is there for text after subtracting out the headers and
            # captions:
            available_length = self.max_tokens - lengths.other_len

            if available_length <= 0:
                warnings.warn(
                    "Headers and captions for this chunk are longer than the total "
                    "available size for the chunk, so they will be ignored: "
                    f"{doc_chunk.text=}, {doc_chunk.meta=}"
                )
                new_chunk = DocChunk(**doc_chunk.export_json_dict())
                new_chunk.meta.captions = None
                new_chunk.meta.headings = None
                return self._split_using_plain_text(doc_chunk=new_chunk, doc_serializer=doc_serializer)

            segments = self.segment(doc_chunk, available_length, doc_serializer)
            chunks = [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]
            return chunks

    def segment(self, doc_chunk: DocChunk, available_length: int, doc_serializer: BaseDocSerializer) -> list[str]:
        segments = []
        if (
            self.repeat_table_header
            and isinstance(doc_serializer, ChunkingDocSerializer)
            and len(doc_chunk.meta.doc_items) == 1
            and isinstance(doc_chunk.meta.doc_items[0], TableItem)
        ):
            header_lines, body_lines = doc_serializer.table_serializer.get_header_and_body_lines(
                table_text=doc_chunk.text
            )

            line_chunker = LineBasedTokenChunker(
                tokenizer=self.tokenizer,
                max_tokens=available_length,
                prefix="\n".join(header_lines),
                omit_prefix_on_overflow=self.omit_header_on_overflow,
                serializer_provider=self.serializer_provider,
            )
            segments = line_chunker.chunk_text(lines=body_lines)
        else:
            sem_chunker = semchunk.chunkerify(self.tokenizer.get_tokenizer(), chunk_size=available_length)
            sem_segments = sem_chunker(doc_chunk.text)
            segments = cast(list[str], sem_segments)
        return segments

    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]):
        output_chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_chunks = len(chunks)

        # Pre-compute per-chunk token counts for fast-path estimation
        chunk_token_counts = [self._count_chunk_tokens(doc_chunk=c) for c in chunks]
        delim_tokens = self.tokenizer.count_tokens(text=self.delim) if self.delim else 0
        running_tokens = 0

        while window_end < num_chunks:
            chunk = chunks[window_end]
            headings = chunk.meta.headings
            ready_to_append = False
            if window_start == window_end:
                current_headings = headings
                running_tokens = chunk_token_counts[window_end]
                window_end += 1
                first_chunk_of_window = chunk
            else:
                if headings != current_headings:
                    ready_to_append = True
                else:
                    # Incremental estimate (upper bound due to BPE)
                    estimated = running_tokens + chunk_token_counts[window_end] + delim_tokens

                    if estimated <= self.max_tokens:
                        # Fast path: estimate is upper bound, so exact also fits.
                        # Build the merged chunk without expensive tokenization.
                        chks = chunks[window_start : window_end + 1]
                        doc_items = [it for chk in chks for it in chk.meta.doc_items]
                        new_chunk = DocChunk(
                            text=self.delim.join([chk.text for chk in chks]),
                            meta=DocMeta(
                                doc_items=doc_items,
                                headings=current_headings,
                                origin=chunk.meta.origin,
                            ),
                        )
                        running_tokens = estimated  # safe upper bound for next iteration
                        window_end += 1
                    else:
                        # Estimate exceeds budget but exact might still fit —
                        # fall back to exact tokenization (original behavior)
                        chks = chunks[window_start : window_end + 1]
                        doc_items = [it for chk in chks for it in chk.meta.doc_items]
                        candidate = DocChunk(
                            # TODO: merging should ideally be done by the serializer:
                            text=self.delim.join([chk.text for chk in chks]),
                            meta=DocMeta(
                                doc_items=doc_items,
                                headings=current_headings,
                                origin=chunk.meta.origin,
                            ),
                        )
                        if self._count_chunk_tokens(doc_chunk=candidate) <= self.max_tokens:
                            window_end += 1
                            new_chunk = candidate
                            running_tokens = self._count_chunk_tokens(doc_chunk=candidate)
                        else:
                            ready_to_append = True
            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.  Either way, end the block
                # and use the current window_end as the start of a new block
                if window_start + 1 == window_end:
                    # just one chunk so use it as is
                    output_chunks.append(first_chunk_of_window)
                else:
                    output_chunks.append(new_chunk)
                # no need to reset window_text, etc. because that will be reset in the
                # next iteration in the if window_start == window_end block
                window_start = window_end
                running_tokens = 0

        return output_chunks

    def chunk(
        self,
        dl_doc: DoclingDocument,
        **kwargs: Any,
    ) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        res: Iterable[DocChunk]
        res = self._inner_chunker.chunk(
            dl_doc=dl_doc,
            doc_serializer=my_doc_ser,
            **kwargs,
        )  # type: ignore
        res = [x for c in res for x in self._split_by_doc_items(c, doc_serializer=my_doc_ser)]
        res = [x for c in res for x in self._split_using_plain_text(c, doc_serializer=my_doc_ser)]
        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)
