"""Token-aware semantic text splitter.

This module is vendored and adapted from ``semchunk`` (version 3.2.5,
https://github.com/isaacus-dev/semchunk), Copyright (c) 2024-2025 Isaacus Pty Ltd
and Umar Butler, distributed under the MIT License. It is reduced to the
single-text, no-overlap code path used by docling-core's chunkers and accepts a
plain token-counting callable instead of deriving one from a tokenizer object.

The MIT License text accompanying the original work is reproduced in
``docling_core/transforms/chunker/LICENSE-semchunk``.
"""

# The splitter table below intentionally contains Unicode punctuation that ruff would
# otherwise flag as ambiguous (RUF001); these characters are meaningful split points.
# ruff: noqa: RUF001

import re
from collections.abc import Callable
from functools import lru_cache
from itertools import accumulate

_NON_WHITESPACE_SEMANTIC_SPLITTERS = (
    # Sentence terminators.
    ".",
    "?",
    "!",
    "*",
    # Clause separators.
    ";",
    ",",
    "(",
    ")",
    "[",
    "]",
    "“",
    "”",
    "‘",
    "’",
    "'",
    '"',
    "`",
    # Sentence interrupters.
    ":",
    "—",
    "…",
    # Word joiners.
    "/",
    "\\",
    "–",
    "&",
    "-",
)
"""Semantically meaningful non-whitespace splitters, ordered most to least desirable."""

_REGEX_ESCAPED_NON_WHITESPACE_SEMANTIC_SPLITTERS = tuple(
    re.escape(splitter) for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS
)


def _split_text(text: str) -> tuple[str, bool, list[str]]:
    """Split text using the most semantically meaningful splitter possible."""
    splitter_is_whitespace = True

    # Try splitting at, in order of most desirable to least desirable:
    # - The largest sequence of newlines and/or carriage returns;
    # - The largest sequence of tabs;
    # - The largest sequence of whitespace characters or, if the largest such sequence
    #   is only a single character and there exists a whitespace character preceded by a
    #   semantically meaningful non-whitespace splitter, then that whitespace character;
    # - A semantically meaningful non-whitespace splitter.
    if "\n" in text or "\r" in text:
        splitter = max(re.findall(r"[\r\n]+", text), key=len)

    elif "\t" in text:
        splitter = max(re.findall(r"\t+", text), key=len)

    elif re.search(r"\s", text):
        splitter = max(re.findall(r"\s+", text), key=len)

        # If the splitter is only a single character, see if we can target whitespace
        # characters that are preceded by semantically meaningful non-whitespace
        # splitters to avoid splitting in the middle of sentences.
        if len(splitter) == 1:
            for escaped_preceder in _REGEX_ESCAPED_NON_WHITESPACE_SEMANTIC_SPLITTERS:
                if whitespace_preceded_by_preceder := re.search(rf"{escaped_preceder}(\s)", text):
                    splitter = whitespace_preceded_by_preceder.group(1)
                    escaped_splitter = re.escape(splitter)

                    return (
                        splitter,
                        splitter_is_whitespace,
                        re.split(rf"(?<={escaped_preceder}){escaped_splitter}", text),
                    )

    else:
        # Identify the most desirable semantically meaningful non-whitespace splitter
        # present in the text.
        for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS:
            if splitter in text:
                splitter_is_whitespace = False
                break

        # If no semantically meaningful splitter is present in the text, return an empty
        # string as the splitter and the text as a list of characters.
        else:  # NOTE only reached if the for loop completes without breaking.
            return "", splitter_is_whitespace, list(text)

    # Return the splitter and the split text.
    return splitter, splitter_is_whitespace, text.split(splitter)


def _bisect_left(seq: list[int], target: float, low: int, high: int) -> int:
    """Return the leftmost index at which ``target`` can be inserted to keep order."""
    while low < high:
        mid = (low + high) // 2

        if seq[mid] < target:
            low = mid + 1

        else:
            high = mid

    return low


def _merge_splits(
    splits: list[str],
    cum_lens: list[int],
    chunk_size: int,
    splitter: str,
    token_counter: Callable[[str], int],
    start: int,
    high: int,
) -> tuple[int, str]:
    """Merge splits until the chunk size is reached.

    Returns the index of the last split included in the merged chunk along with the
    merged chunk itself.
    """
    average = 0.2
    low = start

    offset = cum_lens[start]
    target = offset + (chunk_size * average)

    while low < high:
        i = _bisect_left(cum_lens, target, low=low, high=high)
        midpoint = min(i, high - 1)

        tokens = token_counter(splitter.join(splits[start:midpoint]))

        local_cum = cum_lens[midpoint] - offset

        if local_cum and tokens > 0:
            average = local_cum / tokens
            target = offset + (chunk_size * average)

        if tokens > chunk_size:
            high = midpoint

        else:
            low = midpoint + 1

    end = low - 1
    return end, splitter.join(splits[start:end])


def _chunk(
    text: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
    _start: int = 0,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Recursively split a text into chunks, returning the chunks and their offsets."""
    # Split the text using the most semantically meaningful splitter possible.
    splitter, splitter_is_whitespace, splits = _split_text(text)

    offsets: list[tuple[int, int]] = []
    splitter_len = len(splitter)
    split_lens = [len(split) for split in splits]
    cum_lens = list(accumulate(split_lens, initial=0))
    split_start_iter = accumulate([0] + [split_len + splitter_len for split_len in split_lens])
    split_starts = [start + _start for start in split_start_iter]
    num_splits_plus_one = len(splits) + 1

    chunks: list[str] = []
    skips: set[int] = set()
    """Indices of splits to skip because they have already been added to a chunk."""

    # Iterate through the splits.
    for i, (split, split_start) in enumerate(zip(splits, split_starts)):
        # Skip the split if it has already been added to a chunk.
        if i in skips:
            continue

        # If the split is over the chunk size, recursively chunk it.
        if token_counter(split) > chunk_size:
            new_chunks, new_offsets = _chunk(
                text=split,
                chunk_size=chunk_size,
                token_counter=token_counter,
                _start=split_start,
            )

            chunks.extend(new_chunks)
            offsets.extend(new_offsets)

        # If the split is equal to or under the chunk size, add it and any subsequent
        # splits to a new chunk until the chunk size is reached.
        else:
            final_split_in_chunk_i, new_chunk = _merge_splits(
                splits=splits,
                cum_lens=cum_lens,
                chunk_size=chunk_size,
                splitter=splitter,
                token_counter=token_counter,
                start=i,
                high=num_splits_plus_one,
            )

            # Mark any splits included in the new chunk for exclusion from future chunks.
            skips.update(range(i + 1, final_split_in_chunk_i))

            # Add the chunk.
            chunks.append(new_chunk)

            # Add the chunk's offsets.
            split_end = split_starts[final_split_in_chunk_i] - splitter_len
            offsets.append((split_start, split_end))

        # If the splitter is not whitespace and the split is not the last split, add the
        # splitter to the end of the latest chunk if doing so would not cause it to
        # exceed the chunk size otherwise add the splitter as a new chunk.
        if not splitter_is_whitespace and not (
            i == len(splits) - 1 or all(j in skips for j in range(i + 1, len(splits)))
        ):
            if token_counter(last_chunk_with_splitter := chunks[-1] + splitter) <= chunk_size:
                chunks[-1] = last_chunk_with_splitter
                start, end = offsets[-1]
                offsets[-1] = (start, end + splitter_len)

            else:
                start = offsets[-1][1] if offsets else split_start

                chunks.append(splitter)
                offsets.append((start, start + splitter_len))

    return chunks, offsets


def chunk_text(text: str, chunk_size: int, token_counter: Callable[[str], int]) -> list[str]:
    """Split a text into semantically meaningful chunks of at most ``chunk_size`` tokens.

    Args:
        text: The text to be chunked.
        chunk_size: The maximum number of tokens a chunk may contain.
        token_counter: A callable returning the number of tokens in a given string.

    Returns:
        A list of chunks, each at most ``chunk_size`` tokens long, with any whitespace
        used to split the text removed.
    """
    # Memoize the token counter for the duration of this call: the recursive splitting
    # and merging count tokens for many overlapping substrings.
    memoized_token_counter = lru_cache(maxsize=None)(token_counter)

    chunks, _offsets = _chunk(text=text, chunk_size=chunk_size, token_counter=memoized_token_counter)

    # Remove empty chunks as well as chunks comprised entirely of whitespace.
    return [chunk for chunk in chunks if chunk and not chunk.isspace()]
