import warnings
from collections.abc import Iterator
from functools import cached_property
from typing import Annotated, Any

from pydantic import ConfigDict, Field, computed_field

from docling_core.transforms.chunker import BaseChunk, BaseChunker, DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer
from docling_core.transforms.serializer.base import (
    BaseSerializerProvider,
)
from docling_core.types import DoclingDocument


class LineBasedTokenChunker(BaseChunker):
    """Tokenization-aware chunker that preserves line boundaries.

    This chunker serializes the document content into text and attempts to keep lines
    intact within chunks. It only splits a line if it exceeds the maximum token limit on
    its own. This is particularly useful for structured content like tables, code, or logs
    where line boundaries are semantically important.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Annotated[
        BaseTokenizer,
        Field(
            default_factory=get_default_tokenizer,
            description="The tokenizer to use; either instantiated object or name or path of respective pretrained model",
        ),
    ]

    prefix: Annotated[
        str,
        Field(
            default="",
            description="Text that appears at the beginning of each chunk. Useful for adding context like table headers",
        ),
    ]

    serializer_provider: Annotated[
        BaseSerializerProvider,
        Field(
            default_factory=ChunkingSerializerProvider,
            description="Provider for document serialization during chunking",
        ),
    ]

    @computed_field  # type: ignore[misc]
    @cached_property
    def prefix_len(self) -> int:
        """Cached token count of the prefix, computed during initialization."""
        token_count = self.tokenizer.count_tokens(self.prefix)
        if token_count >= self.max_tokens:
            warnings.warn(
                f"Chunks prefix: {self.prefix} is too long for chunk size {self.max_tokens} and will be ignored"
            )
            return 0
        return token_count

    @property
    def max_tokens(self) -> int:
        """Get maximum number of tokens allowed in a chunk. If not set, limit is resolved from the tokenizer."""
        return self.tokenizer.get_max_tokens()

    def model_post_init(self, __context) -> None:
        # Trigger computation of prefix_len to validate prefix length
        _ = self.prefix_len
        if self.prefix_len == 0 and self.prefix:
            # If prefix_len is 0 but prefix exists, it means prefix was too long
            self.prefix = ""

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """Chunk the provided document using line-based token-aware chunking.

        Args:
            dl_doc (DoclingDocument): document to chunk

        Yields:
            Iterator[BaseChunk]: iterator over extracted chunks
        """
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)

        # Serialize the entire document to get the text
        ser_res = my_doc_ser.serialize()

        if not ser_res.text:
            return

        # Use chunk_text to split the text into chunks
        text_chunks = self.chunk_text(lines=ser_res.text.splitlines(True))

        # Yield DocChunk objects for each text chunk
        for chunk_text in text_chunks:
            yield DocChunk(
                text=chunk_text,
                meta=DocMeta(
                    doc_items=ser_res.get_unique_doc_items(),
                    headings=None,
                    origin=dl_doc.origin,
                ),
            )

    def chunk_text(self, lines: list[str]) -> list[str]:
        chunks = []
        current = self.prefix
        current_len = self.prefix_len

        for line in lines:
            remaining = line

            while True:
                line_tokens = self.tokenizer.count_tokens(remaining)
                available = self.max_tokens - current_len

                # If the remaining part fits entirely into current chunk → append and stop
                if line_tokens <= available:
                    current += remaining
                    current_len += line_tokens
                    break

                # Remaining does NOT fit into current chunk.
                # If it CAN fit into a fresh chunk → flush current and start new one.
                if line_tokens + self.prefix_len <= self.max_tokens:
                    chunks.append(current)
                    current = self.prefix
                    current_len = self.prefix_len
                    # loop continues to retry fitting `remaining`
                    continue

                # Remaining is too large even for an empty chunk → split it.
                # Split off the first segment that fits into current.
                take, remaining = self.split_by_token_limit(remaining, available)

                # Add the taken part
                current += "\n" + take
                current_len += self.tokenizer.count_tokens(take)

                # flush the current chunk (full)
                chunks.append(current)
                current = self.prefix
                current_len = self.prefix_len

            # end while for this line

        # push final chunk if non-empty
        if current != self.prefix:
            chunks.append(current)

        return chunks

    def split_by_token_limit(
        self,
        text: str,
        token_limit: int,
        prefer_word_boundary: bool = True,
    ) -> tuple[str, str]:
        """
        Split `text` into (head, tail) where `head` has at most `token_limit` tokens,
        and `tail` is the remainder. Uses binary search on character indices to minimize
        calls to `count_tokens`.

        Parameters
        ----------
        text : str
            Input string to split.
        token_limit: int
            Maximum number of tokens allowed in the head.
        prefer_word_boundary : bool
            If True, try to end the head on a whitespace boundary (without violating
            the token limit). If no boundary exists in range, fall back to the
            exact max index found by search.

        Returns
        -------
        (head, tail) : tuple[str, str]
            `head` contains at most `token_limit` tokens, `tail` is the remaining suffix.
            If `token_limit <= 0`, returns ("", text).
        """
        if token_limit <= 0 or not text:
            return "", text

        # if the whole text already fits, return as is.
        if self.tokenizer.count_tokens(text) <= token_limit:
            return text, ""

        # Binary search over character indices [0, len(text)]
        lo, hi = 0, len(text)
        best_idx: int | None = None

        while lo <= hi:
            mid = (lo + hi) // 2
            head = text[:mid]
            tok_count = self.tokenizer.count_tokens(head)

            if tok_count <= token_limit:
                best_idx = mid  # feasible; try to extend
                lo = mid + 1
            else:
                hi = mid - 1

        if best_idx is None or best_idx <= 0:
            # Even the first character exceeds the limit (e.g., tokenizer behavior).
            # Return nothing in head, everything in tail.
            return "", text

        # Optionally adjust to a previous whitespace boundary without violating the limit
        if prefer_word_boundary:
            # Search backwards from best_idx to find whitespace; keep within token limit.

            last_space_index = text[:best_idx].rfind(" ")
            if last_space_index >= 0:
                best_idx = last_space_index

        head, tail = text[:best_idx], text[best_idx:]
        return head, tail
