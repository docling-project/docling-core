#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Hybrid chunker implementation leveraging both doc structure & token awareness."""

import warnings
from typing import Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

try:
    import semchunk
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )

from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem


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

    tokenizer: Union[PreTrainedTokenizerBase, str] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    max_tokens: int = None  # type: ignore[assignment]
    merge_peers: bool = True

    _inner_chunker: HierarchicalChunker = HierarchicalChunker()

    @model_validator(mode="after")
    def _patch_tokenizer_and_max_tokens(self) -> Self:
        self._tokenizer = (
            self.tokenizer
            if isinstance(self.tokenizer, PreTrainedTokenizerBase)
            else AutoTokenizer.from_pretrained(self.tokenizer)
        )
        if self.max_tokens is None:
            self.max_tokens = TypeAdapter(PositiveInt).validate_python(
                self._tokenizer.model_max_length
            )
        return self

    def _count_tokens(self, text: Optional[Union[str, list[str]]]):
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self._count_tokens(t)
            return total
        return len(self._tokenizer.tokenize(text, max_length=None))

    class _ChunkLengthInfo(BaseModel):
        total_len: int
        text_len: int
        other_len: int

    def _doc_chunk_length(self, doc_chunk: DocChunk):
        text_length = self._count_tokens(doc_chunk.text)
        headings_length = self._count_tokens(doc_chunk.meta.headings)
        captions_length = self._count_tokens(doc_chunk.meta.captions)
        total = text_length + headings_length + captions_length
        return self._ChunkLengthInfo(
            total_len=total,
            text_len=text_length,
            other_len=total - text_length,
        )

    def _make_chunk_from_doc_items(
        self, doc_chunk: DocChunk, window_text: str, window_start: int, window_end: int
    ):
        meta = DocMeta(
            doc_items=doc_chunk.meta.doc_items[window_start : window_end + 1],
            headings=doc_chunk.meta.headings,
            captions=doc_chunk.meta.captions,
            origin=doc_chunk.meta.origin,
        )
        new_chunk = DocChunk(text=window_text, meta=meta)
        return new_chunk

    def _merge_text(self, t1, t2):
        if t1 == "":
            return t2
        elif t2 == "":
            return t1
        else:
            return f"{t1}{self.delim}{t2}"

    def _split_by_doc_items(self, doc_chunk: DocChunk) -> list[DocChunk]:
        if doc_chunk.meta.doc_items is None or len(doc_chunk.meta.doc_items) <= 1:
            return [doc_chunk]
        length = self._doc_chunk_length(doc_chunk)
        if length.total_len <= self.max_tokens:
            return [doc_chunk]
        else:
            chunks = []
            window_start = 0
            window_end = 0
            window_text = ""
            window_text_length = 0
            other_length = length.other_len
            num_items = len(doc_chunk.meta.doc_items)
            while window_end < num_items:
                doc_item = doc_chunk.meta.doc_items[window_end]
                if isinstance(doc_item, TextItem):
                    text = doc_item.text
                else:
                    raise RuntimeError("Non-TextItem split not implemented yet")
                text_length = self._count_tokens(text)
                if (
                    text_length + window_text_length + other_length < self.max_tokens
                    and window_end < num_items - 1
                ):
                    # Still room left to add more to this chunk AND still at least one
                    # item left
                    window_end += 1
                    window_text_length += text_length
                    window_text = self._merge_text(window_text, text)
                elif text_length + window_text_length + other_length < self.max_tokens:
                    # All the items in the window fit into the chunk and there are no
                    # other items left
                    window_text = self._merge_text(window_text, text)
                    new_chunk = self._make_chunk_from_doc_items(
                        doc_chunk, window_text, window_start, window_end
                    )
                    chunks.append(new_chunk)
                    window_end = num_items
                elif window_start == window_end:
                    # Only one item in the window and it doesn't fit into the chunk. So
                    # we'll just make it a chunk for now and it will get split in the
                    # plain text splitter.
                    window_text = self._merge_text(window_text, text)
                    new_chunk = self._make_chunk_from_doc_items(
                        doc_chunk, window_text, window_start, window_end
                    )
                    chunks.append(new_chunk)
                    window_start = window_end + 1
                    window_end = window_start
                    window_text = ""
                    window_text_length = 0
                else:
                    # Multiple items in the window but they don't fit into the chunk.
                    # However, the existing items must have fit or we wouldn't have
                    # gotten here. So we put everything but the last item into the chunk
                    # and then start a new window INCLUDING the current window end.
                    new_chunk = self._make_chunk_from_doc_items(
                        doc_chunk, window_text, window_start, window_end - 1
                    )
                    chunks.append(new_chunk)
                    window_start = window_end
                    window_text = ""
                    window_text_length = 0
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
                self._tokenizer, chunk_size=available_length
            )
            if available_length <= 0:
                warnings.warn(
                    f"Headers and captions for this chunk are longer than the total amount of size for the chunk, chunk will be ignored: {doc_chunk.text=}"  # noqa
                )
                return []
            text = doc_chunk.text
            segments = sem_chunker.chunk(text)
            chunks = [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]
            return chunks

    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]):
        output_chunks = []
        window_start = 0
        window_end = 0
        num_chunks = len(chunks)
        while window_end < num_chunks:
            chunk = chunks[window_end]
            lengths = self._doc_chunk_length(chunk)
            headings_and_captions = (chunk.meta.headings, chunk.meta.captions)
            ready_to_append = False
            if window_start == window_end:
                # starting a new block of chunks to potentially merge
                current_headings_and_captions = headings_and_captions
                window_text = chunk.text
                window_other_length = lengths.other_len
                window_text_length = lengths.text_len
                window_items = chunk.meta.doc_items
                window_end += 1
                first_chunk_of_window = chunk
            elif (
                headings_and_captions == current_headings_and_captions
                and window_text_length + window_other_length + lengths.text_len
                <= self.max_tokens
            ):
                # there is room to include the new chunk so add it to the window and
                # continue
                window_text = self._merge_text(window_text, chunk.text)
                window_text_length += lengths.text_len
                window_items = window_items + chunk.meta.doc_items
                window_end += 1
            else:
                ready_to_append = True

            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.  Either way, end the block
                # and use the current window_end as the start of a new block
                if window_start + 1 == window_end:
                    # just one chunk so use it as is
                    output_chunks.append(first_chunk_of_window)
                else:
                    new_meta = DocMeta(
                        doc_items=window_items,
                        headings=current_headings_and_captions[0],
                        captions=current_headings_and_captions[1],
                        origin=chunk.meta.origin,
                    )
                    new_chunk = DocChunk(
                        text=window_text,
                        meta=new_meta,
                    )
                    output_chunks.append(new_chunk)
                # no need to reset window_text, etc. because that will be reset in the
                # next iteration in the if window_start == window_end block
                window_start = window_end

        return output_chunks

    def chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        res: Iterable[DocChunk]
        res = self._inner_chunker.chunk(dl_doc=dl_doc, **kwargs)  # type: ignore
        res = [x for c in res for x in self._split_by_doc_items(c)]
        res = [x for c in res for x in self._split_using_plain_text(c)]
        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)
