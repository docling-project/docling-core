import re
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
    SerializationResult,
)
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocItem, TextItem

_ItemRange = tuple[int, int, DocItem]
_PAGE_BREAK_MARKER = re.compile(r"#_#_DOCLING_DOC_PAGE_BREAK_\d+_\d+_#_#")


def _find_line_end(text: str, needle: str, start: int, end: int) -> int:
    """Find ``needle`` in ``text[start:end]`` where it ends a line; -1 if absent.

    Requiring the line end skips a match inside a list marker or another item,
    such as the "1" of "1. 1".
    """
    pos = text.find(needle, start, end)
    while pos >= 0:
        stop = pos + len(needle)
        if stop == end or text[stop] == "\n":
            return pos
        pos = text.find(needle, pos + 1, end)
    return -1


def _locate_items(text: str, items: list[DocItem], start: int, end: int) -> list[_ItemRange]:
    """Find the range of each item of a part, in order, within ``text[start:end]``.

    An item whose text is not found as such, for example a table or a run with
    Markdown formatting, covers the whole part.
    """
    if len(items) == 1:
        return [(start, end, items[0])]
    ranges: list[_ItemRange] = []
    cursor = start
    for item in items:
        needle = item.text if isinstance(item, TextItem) else ""
        pos = _find_line_end(text, needle, cursor, end) if needle else -1
        if pos >= 0:
            cursor = pos + len(needle)
            ranges.append((pos, cursor, item))
        else:
            ranges.append((start, end, item))
    return ranges


def _find_part(text: str, part_text: str, cursor: int) -> tuple[int, int] | None:
    """Find a part in the serialized text from ``cursor``; None if it is not there.

    Serializers write a page break inside a part as a marker that they replace
    once the parts are joined (see ``DocSerializer._create_page_break``), so the
    text around each marker is found on its own.
    """
    pieces = [p for p in _PAGE_BREAK_MARKER.split(part_text) if p]
    start = end = -1
    for piece in pieces:
        pos = text.find(piece, cursor)
        if pos < 0:
            return None
        if start < 0:
            start = pos
        end = cursor = pos + len(piece)
    return (start, end) if start >= 0 else None


def _locate_parts(text: str, parts: list[SerializationResult]) -> list[tuple[int, int, list[_ItemRange]]] | None:
    """Find the range of each part in the serialized text, and of each item in it.

    Returns None if a part with doc items is not found, for example when a custom
    serializer rewrites the parts as it joins them.
    """
    ranges: list[tuple[int, int, list[_ItemRange]]] = []
    cursor = 0
    for part in parts:
        if not part.text or not part.spans:
            continue
        found = _find_part(text, part.text, cursor)
        if found is None:
            return None
        start, cursor = found
        ranges.append((start, cursor, _locate_items(text, part.get_unique_doc_items(), start, cursor)))
    return ranges


class LineBasedTokenChunker(BaseChunker):
    """Tokenization-aware chunker that preserves line boundaries.

    This chunker serializes the document content into text and attempts to keep lines
    intact within chunks. It only splits a line if it exceeds the maximum token limit on
    its own. This is particularly useful for structured content like tables, code, or logs
    where line boundaries are semantically important.

    The chunker supports adding a repeated prefix to each chunk, which can be useful for
    keeping context like headers or metadata, while also offering flexibility when unusually
    long lines appear. If a line is too large to fit alongside the prefix, the chunker
    can either split the line across multiple prefixed chunks to maintain a consistent
    format, or temporarily drop the prefix for that line to preserve the line's
    integrity.

    Note:
        If the prefix itself exceeds max_tokens, it will be split into multiple
        standalone chunks and only included at the beginning of the output.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Annotated[
        BaseTokenizer,
        Field(
            default_factory=get_default_tokenizer,
            description="The tokenizer to use; either instantiated object or name or path of respective pretrained model",
        ),
    ]

    max_tokens_override: Annotated[
        int | None,
        Field(
            gt=0, description="Optional token limit override to constrain chunk size independently of the tokenizer."
        ),
    ] = None

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
            # Split the prefix into chunks using a temporary chunker with no prefix
            temp_chunker = LineBasedTokenChunker(
                tokenizer=self.tokenizer,
                max_tokens_override=self.max_tokens_override,
                prefix="",
                omit_prefix_on_overflow=False,
                serializer_provider=self.serializer_provider,
            )
            return temp_chunker.chunk_text([self.prefix])

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
        """Maximum number of tokens allowed in a chunk.

        Returns max_tokens_override if set, otherwise the maximum tokens
        reported by the tokenizer.
        """
        return self.max_tokens_override or self.tokenizer.get_max_tokens()

    def model_post_init(self, __context) -> None:
        # Trigger computation of prefix_chunks to validate prefix length
        _ = self.prefix_chunks
        # Track whether we've warned about prefix omission
        self._prefix_omitted_warned = False

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """Chunk the provided document using line-based token-aware chunking.

        Args:
            dl_doc (DoclingDocument): document to chunk

        Yields:
            Iterator[BaseChunk]: iterator over extracted chunks. Each chunk's
            ``meta.doc_items`` holds the items whose serialized text it contains.
        """
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        ser_res = my_doc_ser.serialize()

        if not ser_res.text:
            return

        text_chunks, chunk_ends = self._chunk_lines(lines=ser_res.text.splitlines(True))
        part_ranges = _locate_parts(ser_res.text, my_doc_ser.get_parts())

        start = 0
        first_part = 0
        for chunk_text, end in zip(text_chunks, chunk_ends):
            if part_ranges is None:
                doc_items = ser_res.get_unique_doc_items()
            else:
                # Chunks and parts both run forward through the text.
                while first_part < len(part_ranges) and part_ranges[first_part][1] <= start:
                    first_part += 1
                overlapping: list[DocItem] = []
                part_items: list[DocItem] = []
                for part_start, _, item_ranges in part_ranges[first_part:]:
                    if part_start >= end:
                        break
                    for item_start, item_end, item in item_ranges:
                        part_items.append(item)
                        if item_start < end and start < item_end:
                            overlapping.append(item)
                # A chunk of list markers or blank lines overlaps a part but none of
                # its items; one of prefix or document-level meta text overlaps no
                # part. DocMeta needs at least one item.
                items = overlapping or part_items or ser_res.get_unique_doc_items()
                doc_items = list({item.self_ref: item for item in items}.values())
            yield DocChunk(
                text=chunk_text,
                meta=DocMeta(
                    doc_items=doc_items,
                    headings=None,
                    origin=dl_doc.origin,
                ),
            )
            start = end

    def chunk_text(self, lines: list[str]) -> list[str]:
        return self._chunk_lines(lines)[0]

    def _chunk_lines(self, lines: list[str]) -> tuple[list[str], list[int]]:
        """Chunk lines, and give the end offset of each chunk in ``"".join(lines)``.

        A chunk holds the prefix, if any, and the text after the previous chunk's
        end; a chunk that holds only the prefix ends where the previous one did.
        """
        chunks: list[str] = []
        ends: list[int] = []
        pos = 0  # characters of the joined lines placed in chunks so far

        def flush(text: str) -> None:
            chunks.append(text)
            ends.append(pos)

        # Handle prefix chunks - add them first if prefix is too large
        if self.prefix_chunks and self.prefix_len == 0:
            # Prefix is too large, add prefix chunks first
            for prefix_chunk in self.prefix_chunks:
                flush(prefix_chunk)
            # Continue with regular chunking without prefix
            current = ""
            current_len = 0
        else:
            # Check if first line would overflow with prefix when omit_prefix_on_overflow=True
            # If yes, add prefix as a standalone chunk first to ensure it's visible
            if self.omit_prefix_on_overflow and self.prefix_len > 0 and lines:
                # If first line would overflow with prefix, add prefix as standalone chunk
                if self.tokenizer.count_tokens(self.prefix + lines[0]) > self.max_tokens:
                    flush(self.prefix)
                    current = ""
                    current_len = 0
                else:
                    # First line fits with prefix, use normal flow
                    current = self.prefix
                    current_len = self.prefix_len
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
                if self.tokenizer.count_tokens(current + remaining) <= self.max_tokens:
                    current += remaining
                    pos += len(remaining)
                    current_len = self.tokenizer.count_tokens(current)
                    break

                # Remaining does NOT fit into current chunk.
                # If it CAN fit into a fresh chunk → flush current and start new one.
                fresh_prefix = self.prefix if self.prefix_len > 0 else ""
                if self.tokenizer.count_tokens(fresh_prefix + remaining) <= self.max_tokens:
                    if current:
                        flush(current)
                    # Only add prefix to new chunks if it fits (prefix_len > 0)
                    current = fresh_prefix
                    current_len = self.prefix_len
                    # loop continues to retry fitting `remaining`
                    continue

                # Check if omitting prefix would allow the line to fit
                # (only if line itself is not larger than max_tokens)
                if self.omit_prefix_on_overflow and line_tokens <= self.max_tokens and self.prefix_len > 0:
                    # Warn once when prefix is actually omitted
                    if not self._prefix_omitted_warned:
                        warnings.warn(
                            f"Prefix omitted for at least one line due to omit_prefix_on_overflow=True. "
                            f"Line would overflow with prefix ({self.prefix_len} tokens) but fits without it "
                            f"within max_tokens ({self.max_tokens}). This may result in inconsistent chunk formatting.",
                            UserWarning,
                            stacklevel=4,
                        )
                        self._prefix_omitted_warned = True

                    # Omit prefix for this line to make it fit
                    # Only append current if it contains more than just the prefix
                    if current and current != self.prefix:
                        flush(current)
                    current = ""
                    current_len = 0
                    # loop continues to retry fitting `remaining` without prefix
                    continue

                # Remaining is too large even for an empty chunk → split it.
                # Split off the first segment that fits into current.
                take, remaining = self.split_by_token_limit(remaining, available)

                while take and self.tokenizer.count_tokens(current + take) > self.max_tokens:
                    take_limit = self.tokenizer.count_tokens(take) - 1
                    take, returned = self.split_by_token_limit(take, take_limit)
                    remaining = returned + remaining

                if not take:
                    take, remaining = self._split_by_token_limit(
                        remaining,
                        token_limit=self.max_tokens,
                        prefer_word_boundary=True,
                        prefix=current,
                    )

                if not take:
                    if current:
                        flush(current)
                        if self.prefix_len > 0 and not self.omit_prefix_on_overflow:
                            current = self.prefix
                            current_len = self.prefix_len
                        else:
                            current = ""
                            current_len = 0
                        continue
                    raise ValueError(
                        f"Character {remaining[0]!r} cannot fit within the token budget of {self.max_tokens}"
                    )

                # Add the taken part
                current += take
                pos += len(take)
                current_len = self.tokenizer.count_tokens(current)

                # flush the current chunk (full)
                flush(current)

                # Determine whether to add prefix to the next chunk
                # If omit_prefix_on_overflow is True, don't add prefix for overflow chunks
                if self.prefix_len > 0 and not self.omit_prefix_on_overflow:
                    current = self.prefix
                    current_len = self.prefix_len
                else:
                    # Warn once when prefix is omitted for overflow chunks
                    if self.omit_prefix_on_overflow and self.prefix_len > 0 and not self._prefix_omitted_warned:
                        warnings.warn(
                            f"Prefix omitted for at least one line due to omit_prefix_on_overflow=True. "
                            f"Line would overflow with prefix ({self.prefix_len} tokens) but fits without it "
                            f"within max_tokens ({self.max_tokens}). This may result in inconsistent chunk formatting.",
                            UserWarning,
                            stacklevel=4,
                        )
                        self._prefix_omitted_warned = True
                    current = ""
                    current_len = 0

            # end while for this line

        # push final chunk if non-empty
        # Check against both empty string and prefix (for when prefix fits)
        if current and (self.prefix_len == 0 or current != self.prefix):
            flush(current)

        return chunks, ends

    def split_by_token_limit(
        self,
        text: str,
        token_limit: int,
        prefer_word_boundary: bool = True,
    ) -> tuple[str, str]:
        """Split `text` into (head, tail) where `head` has at most `token_limit` tokens,
        and `tail` is the remainder. Uses binary search on character indices to minimize
        calls to `count_tokens`.

        Args:
            text: Input string to split.
            token_limit: Maximum number of tokens allowed in the head.
            prefer_word_boundary: If True, try to end the head on a whitespace boundary (without violating
                the token limit). If no boundary exists in range, fall back to the
                exact max index found by search.

        Returns:
           (head, tail) where `head` contains at most `token_limit` tokens, `tail` is the remaining suffix. If `token_limit <= 0`, returns ("", text).
        """
        return self._split_by_token_limit(
            text,
            token_limit=token_limit,
            prefer_word_boundary=prefer_word_boundary,
            prefix="",
        )

    def _split_by_token_limit(
        self,
        text: str,
        token_limit: int,
        prefer_word_boundary: bool,
        prefix: str,
    ) -> tuple[str, str]:
        if token_limit <= 0 or not text:
            return "", text

        # if the whole text already fits, return as is.
        if self.tokenizer.count_tokens(prefix + text) <= token_limit:
            return text, ""

        # Binary search over character indices [0, len(text)]
        lo, hi = 0, len(text)
        best_idx: int | None = None

        while lo <= hi:
            mid = (lo + hi) // 2
            head = text[:mid]
            tok_count = self.tokenizer.count_tokens(prefix + head)

            if tok_count <= token_limit:
                best_idx = mid  # feasible; try to extend
                lo = mid + 1
            else:
                hi = mid - 1

        if best_idx is None or best_idx <= 0:
            best_idx = next(
                (
                    index
                    for index in range(len(text) - 1, 0, -1)
                    if self.tokenizer.count_tokens(prefix + text[:index]) <= token_limit
                ),
                None,
            )
            if best_idx is None:
                return "", text

        # Optionally adjust to a previous whitespace boundary without violating the limit
        if prefer_word_boundary:
            # Search backwards from best_idx to find whitespace; keep within token limit.
            # Only snap back if it produces a non-empty head (last_space_index > 0)
            last_space_index = text[:best_idx].rfind(" ")
            if last_space_index > 0 and self.tokenizer.count_tokens(prefix + text[:last_space_index]) <= token_limit:
                best_idx = last_space_index

        head, tail = text[:best_idx], text[best_idx:]
        return head, tail
