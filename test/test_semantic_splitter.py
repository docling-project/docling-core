"""Tests for the vendored token-aware semantic text splitter.

These exercise the splitter's contract directly with deterministic token counters,
so they run without downloading any tokenizer model.
"""

from docling_core.transforms.chunker._semantic_splitter import chunk_text


def _word_counter(text: str) -> int:
    return len(text.split())


def _char_counter(text: str) -> int:
    return len(text)


SAMPLE = (
    "Docling parses documents into a unified representation. "
    "It supports PDF, DOCX, HTML, and Markdown inputs. "
    "Chunking splits long passages into smaller, token-bounded segments "
    "so that downstream retrieval and generation stay within model limits."
)


def test_short_text_is_returned_as_single_chunk():
    text = "A short sentence."
    chunks = chunk_text(text, chunk_size=100, token_counter=_word_counter)
    assert chunks == [text]


def test_every_chunk_respects_the_token_budget():
    chunk_size = 5
    chunks = chunk_text(SAMPLE, chunk_size=chunk_size, token_counter=_word_counter)
    assert len(chunks) > 1  # the sample must actually get split
    assert all(_word_counter(chunk) <= chunk_size for chunk in chunks)


def test_no_empty_or_whitespace_only_chunks():
    chunks = chunk_text(SAMPLE, chunk_size=4, token_counter=_word_counter)
    assert chunks
    assert all(chunk and not chunk.isspace() for chunk in chunks)


def test_content_is_preserved_in_order():
    chunks = chunk_text(SAMPLE, chunk_size=3, token_counter=_word_counter)
    # Whitespace used as a split point is dropped, but the words themselves and their
    # order must be preserved across the concatenation of chunks.
    assert " ".join(chunks).split() == SAMPLE.split()


def test_recurses_into_oversized_splits_with_character_budget():
    # A single long token forces the recursive sub-word splitting path.
    word = "supercalifragilisticexpialidocious"
    chunk_size = 8
    chunks = chunk_text(word, chunk_size=chunk_size, token_counter=_char_counter)
    assert len(chunks) > 1
    assert all(_char_counter(chunk) <= chunk_size for chunk in chunks)
    assert "".join(chunks) == word
