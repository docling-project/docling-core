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

    omit_prefix_on_overflow: Annotated[
        bool,
        Field(default=False, description="When True, omit prefix for lines that would overflow with it. "),
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
    def prefix_chunks(self) -> list[str]:
        """Cached list of prefix chunks, computed during initialization.

        If the prefix is larger than max_tokens, it will be split into multiple chunks.
        """
        if not self.prefix:
            return []

        token_count = self.tokenizer.count_tokens(self.prefix)
        if token_count >= self.max_tokens:
            warnings.warn(
                f"Chunks prefix is too long ({token_count} tokens) for chunk size {self.max_tokens}. "
                f"It will be split into multiple chunks and only included in the first chunk(s). "
                f"Consider increasing max_tokens to accommodate the full prefix in each chunk."
            )
            # Split the prefix into chunks using the generalized chunking logic
            return self._chunk_text_by_tokens(self.prefix, max_tokens=self.max_tokens)

        return [self.prefix]

    @computed_field  # type: ignore[misc]
    @cached_property
    def prefix_len(self) -> int:
        """Cached token count of the prefix that fits in a single chunk.

        Returns 0 if prefix needs to be split into multiple chunks.
        """
        if not self.prefix:
            return 0

        token_count = self.tokenizer.count_tokens(self.prefix)
        if token_count >= self.max_tokens:
            return 0
        return token_count

    @property
    def max_tokens(self) -> int:
        """Get maximum number of tokens allowed in a chunk. If not set, limit is resolved from the tokenizer."""
        return self.tokenizer.get_max_tokens()

    def model_post_init(self, __context) -> None:
        # Trigger computation of prefix_chunks to validate prefix length
        _ = self.prefix_chunks

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

    def _chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Generalized function to chunk text by token limit.

        This function splits text into chunks where each chunk respects the token limit.
        It attempts to split on line boundaries first, then falls back to splitting
        within lines if necessary.

        Parameters
        ----------
        text : str
            The text to chunk.
        max_tokens : int
            Maximum number of tokens per chunk.

        Returns
        -------
        list[str]
            List of text chunks, each respecting the token limit.
        """
        if not text:
            return []

        chunks = []
        lines = text.splitlines(True)
        current = ""
        current_len = 0

        for line in lines:
            remaining = line

            while True:
                line_tokens = self.tokenizer.count_tokens(remaining)
                available = max_tokens - current_len

                # If the remaining part fits entirely into current chunk → append and stop
                if line_tokens <= available:
                    current += remaining
                    current_len += line_tokens
                    break

                # Remaining does NOT fit into current chunk.
                # If it CAN fit into a fresh chunk → flush current and start new one.
                if line_tokens <= max_tokens:
                    if current:
                        chunks.append(current)
                    current = ""
                    current_len = 0
                    # loop continues to retry fitting `remaining`
                    continue

                # Remaining is too large even for an empty chunk → split it.
                # Split off the first segment that fits into current.
                take, remaining = self.split_by_token_limit(remaining, available)

                # Zero-progress detection: if take is empty, force character-level split
                if not take:
                    # Fallback: take at least one character to ensure progress
                    if remaining:
                        take = remaining[0]
                        remaining = remaining[1:]
                    else:
                        # Should not happen, but break to prevent infinite loop
                        break

                # Add the taken part
                if current:
                    current += "\n" + take
                else:
                    current = take
                current_len = self.tokenizer.count_tokens(current)

                # flush the current chunk (full)
                if current:
                    chunks.append(current)
                current = ""
                current_len = 0

        # push final chunk if non-empty
        if current:
            chunks.append(current)

        return chunks

    def chunk_text(self, lines: list[str]) -> list[str]:
        chunks = []

        # Handle prefix chunks - add them first if prefix is too large
        if self.prefix_chunks and self.prefix_len == 0:
            # Prefix is too large, add prefix chunks first
            chunks.extend(self.prefix_chunks)
            # Continue with regular chunking without prefix
            current = ""
            current_len = 0
        else:
            # Normal case: prefix fits in a single chunk or no prefix
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
                    # Only add prefix to new chunks if it fits (prefix_len > 0)
                    if self.prefix_len > 0:
                        current = self.prefix
                        current_len = self.prefix_len
                    else:
                        current = ""
                        current_len = 0
                    # loop continues to retry fitting `remaining`
                    continue

                # Check if omitting prefix would allow the line to fit
                # (only if line itself is not larger than max_tokens)
                if self.omit_prefix_on_overflow and line_tokens <= self.max_tokens and self.prefix_len > 0:
                    # Omit prefix for this line to make it fit
                    # Only append current if it contains more than just the prefix
                    if current and current != self.prefix:
                        chunks.append(current)
                    current = ""
                    current_len = 0
                    # loop continues to retry fitting `remaining` without prefix
                    continue

                # Remaining is too large even for an empty chunk → split it.
                # Split off the first segment that fits into current.
                take, remaining = self.split_by_token_limit(remaining, available)

                # Zero-progress detection: if take is empty, force character-level split
                if not take:
                    # Fallback: take at least one character to ensure progress
                    if remaining:
                        take = remaining[0]
                        remaining = remaining[1:]
                    else:
                        # Should not happen, but break to prevent infinite loop
                        break

                # Add the taken part
                current += "\n" + take
                current_len += self.tokenizer.count_tokens(take)

                # flush the current chunk (full)
                chunks.append(current)

                # Determine whether to add prefix to the next chunk
                # If omit_prefix_on_overflow is True, don't add prefix for overflow chunks
                if self.prefix_len > 0 and not self.omit_prefix_on_overflow:
                    current = self.prefix
                    current_len = self.prefix_len
                else:
                    current = ""
                    current_len = 0

            # end while for this line

        # push final chunk if non-empty
        # Check against both empty string and prefix (for when prefix fits)
        if current and (self.prefix_len == 0 or current != self.prefix):
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
            # Only snap back if it produces a non-empty head (last_space_index > 0)
            last_space_index = text[:best_idx].rfind(" ")
            if last_space_index > 0:
                best_idx = last_space_index

        head, tail = text[:best_idx], text[best_idx:]
        return head, tail
