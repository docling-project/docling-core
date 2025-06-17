#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Hybrid chunker implementation leveraging both doc structure & token awareness."""
import warnings
from functools import cached_property
from typing import Any, Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from transformers import PreTrainedTokenizerBase

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

try:
    import semchunk
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
    BaseDocSerializer,
    BaseSerializerProvider,
)
from docling_core.types import DoclingDocument


def _get_default_tokenizer():
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    return HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


class HybridChunker(BaseChunker):
    r"""Chunker doing tokenization-aware refinements on top of document layout chunking.

    Args:
        tokenizer: The tokenizer to use; either instantiated object or name or path of
            respective pretrained model
        max_tokens: The maximum number of tokens per chunk. If not set, limit is
            resolved from the tokenizer
        merge_peers: Whether to merge undersized chunks sharing same relevant metadata
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: BaseTokenizer = Field(default_factory=_get_default_tokenizer)
    merge_peers: bool = True

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()

    @model_validator(mode="before")
    @classmethod
    def _patch(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tokenizer = data.get("tokenizer")
            max_tokens = data.get("max_tokens")
            if not isinstance(tokenizer, BaseTokenizer) and (
                # some legacy param passed:
                tokenizer is not None
                or max_tokens is not None
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
                elif tokenizer is None or isinstance(
                    tokenizer, PreTrainedTokenizerBase
                ):
                    kwargs = {
                        "tokenizer": tokenizer or _get_default_tokenizer().tokenizer
                    }
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
        return HierarchicalChunker(serializer_provider=self.serializer_provider)

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
                ]
            )
        )
        new_chunk = DocChunk(text=window_text, meta=meta)
        return new_chunk

    def _split_by_doc_items(
        self, doc_chunk: DocChunk, doc_serializer: BaseDocSerializer
    ) -> list[DocChunk]:
        chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_items = len(doc_chunk.meta.doc_items)
        while window_end < num_items:
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
            chunks.append(new_chunk)
        return chunks

    def _split_using_plain_text(
        self,
        doc_chunk: DocChunk,
    ) -> list[DocChunk]:
        lengths = self._doc_chunk_length(doc_chunk)
        if lengths.total_len <= self.max_tokens:
            return [DocChunk(**doc_chunk.export_json_dict())]
        else:
            # How much room is there for text after subtracting out the headers and
            # captions:
            available_length = self.max_tokens - lengths.other_len
            sem_chunker = semchunk.chunkerify(
                self.tokenizer.get_tokenizer(), chunk_size=available_length
            )
            if available_length <= 0:
                warnings.warn(
                    "Headers and captions for this chunk are longer than the total "
                    "available size for the chunk, so they will be ignored: "
                    f"{doc_chunk.text=}, {doc_chunk.meta=}"
                )
                new_chunk = DocChunk(**doc_chunk.export_json_dict())
                new_chunk.meta.captions = None
                new_chunk.meta.headings = None
                return self._split_using_plain_text(doc_chunk=new_chunk)
            text = doc_chunk.text
            segments = sem_chunker.chunk(text)
            chunks = [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]
            return chunks

    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]):
        output_chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_chunks = len(chunks)
        while window_end < num_chunks:
            chunk = chunks[window_end]
            headings = chunk.meta.headings
            ready_to_append = False
            if window_start == window_end:
                current_headings = headings
                window_end += 1
                first_chunk_of_window = chunk
            else:
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
                if (
                    headings == current_headings
                    and self._count_chunk_tokens(doc_chunk=candidate) <= self.max_tokens
                ):
                    # there is room to include the new chunk so add it to the window and
                    # continue
                    window_end += 1
                    new_chunk = candidate
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
        res = [
            x
            for c in res
            for x in self._split_by_doc_items(c, doc_serializer=my_doc_ser)
        ]
        res = [x for c in res for x in self._split_using_plain_text(c)]
        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)
