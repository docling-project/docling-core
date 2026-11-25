"""Progress guarantees when a mandatory prefix leaves insufficient token space."""

import pytest
import tiktoken

from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer


@pytest.mark.parametrize(
    "prefix,limit,lines",
    [
        ("a", 2, ["🙂"]),
        ("Heading: a", 4, ["🙂"]),
        ("Context:" + " category" * 61, 64, ["🙂"]),
        ("a", 2, ["b", "🙂"]),
    ],
)
def test_mandatory_prefix_rejects_content_that_cannot_fit(prefix, limit, lines):
    tokenizer = OpenAITokenizer(tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=limit)
    chunker = LineBasedTokenChunker(tokenizer=tokenizer, prefix=prefix)

    assert tokenizer.count_tokens(prefix) < limit
    assert tokenizer.count_tokens(prefix + "🙂") > limit
    with pytest.raises(ValueError, match=r"cannot fit within the token budget.*with prefix"):
        chunker.chunk_text(lines)


@pytest.mark.parametrize(
    "prefix,limit,omit_prefix,lines,expected",
    [
        ("a", 2, True, ["🙂"], ["a", "🙂"]),
        ("", 2, False, ["🙂"], ["🙂"]),
        ("a", 3, False, ["🙂🙂"], ["a🙂", "a🙂"]),
        ("a", 2, False, ["b", "cd"], ["abcd"]),
        ("!", 1, False, ["!", "hello world"], ["!", "!", "hello", " world"]),
    ],
)
def test_prefix_chunking_controls(prefix, limit, omit_prefix, lines, expected):
    tokenizer = OpenAITokenizer(tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=limit)
    chunker = LineBasedTokenChunker(tokenizer=tokenizer, prefix=prefix, omit_prefix_on_overflow=omit_prefix)

    chunks = chunker.chunk_text(lines)

    assert chunks == expected
    assert all(tokenizer.count_tokens(chunk) <= limit for chunk in chunks)
