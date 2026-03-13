import json

import pytest
from transformers import AutoTokenizer

from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument as DLDocument
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 25
INNER_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)


@pytest.fixture(scope="module")
def default_tokenizer():
    """Fixture providing a default HuggingFaceTokenizer for tests."""
    return HuggingFaceTokenizer(
        tokenizer=INNER_TOKENIZER,
        max_tokens=MAX_TOKENS,
    )


def _process(act_data, exp_path_str):
    """Helper function to either generate or compare test data."""
    if GEN_TEST_DATA:
        with open(exp_path_str, mode="w", encoding="utf-8") as f:
            json.dump(act_data, fp=f, indent=4)
            f.write("\n")
    else:
        with open(exp_path_str, encoding="utf-8") as f:
            exp_data = json.load(fp=f)
        assert exp_data == act_data


def test_chunk_text_with_prefix(default_tokenizer):
    """Test text chunking with a prefix."""
    prefix = "Context: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
    )

    lines = ["Line 1\n", "Line 2\n", "Line 3"]
    chunks = chunker.chunk_text(lines)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # Each chunk should start with the prefix
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk.startswith(prefix)


def test_chunk_text_long_prefix_warning(default_tokenizer):
    """Test that a warning is issued when prefix is too long."""
    # Create a very long prefix that exceeds max_tokens
    long_prefix = "This is a very long prefix " * 50

    with pytest.warns(UserWarning, match="too long.*will be split into multiple chunks"):
        chunker = LineBasedTokenChunker(
            tokenizer=default_tokenizer,
            prefix=long_prefix,
        )

    # Prefix should be kept but split into chunks
    assert chunker.prefix == long_prefix
    assert chunker.prefix_len == 0  # Returns 0 when prefix is too large
    assert len(chunker.prefix_chunks) > 1  # Should be split into multiple chunks
    
    # Test that chunking works with large prefix
    lines = ["Line 1", "Line 2", "Line 3"]
    chunks = chunker.chunk_text(lines)
    
    # First chunk(s) should contain the prefix chunks
    assert len(chunks) > 0
    # The prefix chunks should be at the beginning
    for i, prefix_chunk in enumerate(chunker.prefix_chunks):
        if i < len(chunks):
            assert prefix_chunk in chunks[i]


def test_chunk_text_single_long_line(default_tokenizer):
    """Test chunking when a single line exceeds max_tokens."""

    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    # Create a very long line
    long_line = "word " * MAX_TOKENS * 5
    lines = [long_line]
    chunks = chunker.chunk_text(lines)

    assert len(chunks) > 1
    # Verify each chunk respects token limit
    for chunk in chunks:
        token_count = chunker.tokenizer.count_tokens(chunk)
        assert token_count <= MAX_TOKENS


def test_chunk_text_empty_string(default_tokenizer):
    """Test chunking an empty list."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    chunks = chunker.chunk_text([])
    assert len(chunks) == 0


def test_chunk_text_single_line(default_tokenizer):
    """Test chunking a single line that fits in one chunk."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    text = "This is a single short line.\n"
    lines = [text]
    chunks = chunker.chunk_text(lines)

    assert len(chunks) == 1
    assert chunks[0] == text
    # newline should be preserved
    assert "\n" in chunks[0]


def test_split_by_token_limit(default_tokenizer):
    """Test the split_by_token_limit method."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    available = 10
    text = "This is a test sentence with multiple words that should be split."
    head, tail = chunker.split_by_token_limit(text, token_limit=available)

    assert len(head) > 0
    assert len(tail) > 0
    assert chunker.tokenizer.count_tokens(head) <= available
    assert head + tail == text


def test_split_by_token_limit_zero_limit(default_tokenizer):
    """Test split_by_token_limit with zero token limit."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    text = "Some text"
    head, tail = chunker.split_by_token_limit(text, token_limit=0)

    assert head == ""
    assert tail == text


def test_split_by_token_limit_fits_entirely(default_tokenizer):
    """Test split_by_token_limit when text fits within limit."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    text = "Short text"
    head, tail = chunker.split_by_token_limit(text, token_limit=100)

    assert head == text
    assert tail == ""


def test_split_by_token_limit_word_boundary(default_tokenizer):
    """Test that split_by_token_limit prefers word boundaries."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    text = "word1 word2 word3 word4 word5"
    head, tail = chunker.split_by_token_limit(text, token_limit=5, prefer_word_boundary=True)

    # Head should end at a word boundary (space)
    if len(head) > 0 and len(tail) > 0:
        # Either head ends with a space or tail starts with a space
        assert head[-1].isspace() or tail[0].isspace() or not head[-1].isalnum()


def test_chunk_text_with_prefix_and_long_lines(default_tokenizer):
    """Test chunking with prefix when lines are long."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
    )

    long_line = "This is a long line that will need to be split " * 3
    lines = [long_line]
    chunks = chunker.chunk_text(lines)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.startswith(prefix)
        token_count = chunker.tokenizer.count_tokens(chunk)
        assert token_count <= MAX_TOKENS



def test_chunk_document(default_tokenizer):
    """Test the chunk() method with a DoclingDocument."""
    # Create a simple DoclingDocument
    doc = DLDocument(name="test_doc")
    paragraphs = ["This is the first paragraph with some content.",
    "This is the second paragraph with more content",
    "This is the third paragraph with even more content."]

    # Add some text items to the document
    for t in paragraphs:
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=t)

    # Create chunker
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    # Chunk the document
    chunks = list(chunker.chunk(doc))

    # Verify chunks were created
    assert len(chunks) > 0

    # Verify each chunk is a DocChunk with proper structure
    for chunk in chunks:
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "meta")
        assert isinstance(chunk.text, str)
        assert len(chunk.text) > 0

        # Verify token count is within limit
        token_count = chunker.tokenizer.count_tokens(chunk.text)
        assert token_count <= MAX_TOKENS

     # Verify each paragraph resides fully in a chunk
    for t in paragraphs:
        assert any(t in c.text for c in chunks)


def test_chunk_empty_document(default_tokenizer):
    """Test the chunk() method with an empty document."""
    # Create an empty DoclingDocument
    doc = DLDocument(name="empty_doc")

    # Create chunker
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    # Chunk the document
    chunks = list(chunker.chunk(doc))

    # Should return no chunks for empty document
    assert len(chunks) == 0


def test_chunk_document_with_long_content(default_tokenizer):
    """Test the chunk() method with long content that requires multiple chunks."""
    # Create a DoclingDocument with long content
    doc = DLDocument(name="long_doc")
    prefix = "Document: "

    # Add a very long paragraph
    long_text = "This is a sentence with multiple words. " * 50
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=long_text)

    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix
    )

    # Chunk the document
    chunks = list(chunker.chunk(doc))

    # Should create multiple chunks
    assert len(chunks) > 1

    # Verify each chunk respects token limit
    for chunk in chunks:
        assert chunk.text.startswith(prefix)
        token_count = chunker.tokenizer.count_tokens(chunk.text)
        assert token_count <= MAX_TOKENS


def test_infinite_loop_regression_long_unbreakable_token(default_tokenizer):
    """
    Regression test for infinite loop bug when processing text with a long
    unbreakable token sequence preceded by a space.

    This test reproduces the issue where LineBasedTokenChunker.chunk_text()
    would enter an infinite loop when the prefer_word_boundary logic in
    split_by_token_limit() snapped best_idx back to 0, producing an empty
    head and returning the tail unchanged.

    The fix ensures:
    1. split_by_token_limit only snaps to word boundary if it produces non-empty head
    2. chunk_text detects zero-progress and forces character-level splitting as fallback
    """
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    # Create text with leading space followed by long unbreakable token
    long_word = "a" * 200
    text = "Header " + long_word + " Footer\n"
    token_count = chunker.tokenizer.count_tokens(text)
    assert token_count <= MAX_TOKENS

    # This should complete without hanging
    result = chunker.chunk_text(lines=[text])

    assert len(result) == 1
    assert result[0] == text


def test_split_by_token_limit_leading_space_regression(default_tokenizer):
    """
    Test that split_by_token_limit handles text with leading space correctly.

    Previously, when text started with a space followed by unbreakable content,
    the word boundary snap-back could produce an empty head, causing infinite loops.
    """
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    # Text with leading space then long unbreakable sequence
    text = " " + "a" * 200

    # Split with small token limit
    head, tail = chunker.split_by_token_limit(text, token_limit=10, prefer_word_boundary=True)

    # Head should not be empty (must make progress)
    assert len(head) > 0, "split_by_token_limit must make progress and return non-empty head"

    # Tail should be shorter than original
    assert len(tail) < len(text), "split_by_token_limit must consume some input"

    # Verify head + tail reconstructs original
    assert head + tail == text

    # Verify head respects token limit
    head_tokens = chunker.tokenizer.count_tokens(head)
    assert head_tokens <= 10


def test_character_level_fallback_on_zero_available(default_tokenizer):
    """
    Test that chunk_text uses character-level fallback when available space is 0.

    This test demonstrates a real scenario where the fallback is needed:
    1. Current chunk is exactly at max_tokens (available = 0)
    2. Remaining text is too long to fit in a fresh chunk (exceeds max_tokens)
    3. split_by_token_limit is called with token_limit=0
    4. It returns ("", text) in split_by_token_limit of line_chunker.py
    5. The fallback takes 1 character to ensure progress
    """
    # Use a very small max_tokens to make it easier to create the scenario
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
    )

    # First line: 8 tokens (leaves room for 2 more)
    first_line = "word " * (MAX_TOKENS - 2)

    # Second line: more than 10 tokens (so it can't fit in a fresh chunk)
    second_line = "x" * 100

    # Verify second line is indeed > max_tokens
    second_line_tokens = chunker.tokenizer.count_tokens(second_line)
    assert second_line_tokens > MAX_TOKENS, (
        f"Second line must exceed max_tokens for test to work: "
        f"{second_line_tokens} <= {MAX_TOKENS}"
    )

    lines = [first_line, second_line]
    result = chunker.chunk_text(lines=lines)

    # Verify we got multiple chunks (the long second line should be split)
    assert len(result) > 1, f"Should have multiple chunks, got {len(result)}"

    # Verify each chunk respects token limit (allow small overflow due to newline addition)
    for i, chunk in enumerate(result):
        token_count = chunker.tokenizer.count_tokens(chunk)
        assert token_count <= MAX_TOKENS, f"Chunk {i} exceeds token limit: {token_count} > {MAX_TOKENS}"

    # Verify all content is preserved
    combined = "".join(result)
    # First line should be in first chunk
    assert first_line in result[0], "First line should be in first chunk"
    # Second line should be split across remaining chunks
    combined_without_newlines = combined.replace("\n", "")
    assert second_line in combined_without_newlines, "Second line should be preserved (possibly split)"


def test_omit_prefix_on_overflow_false(default_tokenizer):
    """Test default behavior when omit_prefix_on_overflow is False (default)."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=False,
    )
    
    # Create a line that fits without prefix but not with prefix
    # prefix_len is around 2 tokens, create a line that's just 1 token short
    line = "word " * (MAX_TOKENS - 1) 
    line_tokens = chunker.tokenizer.count_tokens(line)
    
    # Verify test setup: line should fit alone but not with prefix
    assert line_tokens <= MAX_TOKENS, f"Line should fit in chunk: {line_tokens} <= {MAX_TOKENS}"
    assert line_tokens + chunker.prefix_len > MAX_TOKENS, (
        f"Line with prefix should NOT fit: {line_tokens} + {chunker.prefix_len} > {MAX_TOKENS}"
    )
    
    lines = [line]
    chunks = chunker.chunk_text(lines)
    
    # With omit_prefix_on_overflow=False, the line should be split
    # and each chunk should have the prefix
    assert len(chunks) > 1, "Line should be split into multiple chunks"
    for chunk in chunks:
        assert chunk.startswith(prefix), "Each chunk should start with prefix"


def test_omit_prefix_on_overflow_true(default_tokenizer):
    """Test that prefix is added as standalone chunk when first line would overflow with it."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Create a line that fits without prefix but not with prefix
    line = "word " * (MAX_TOKENS - 1)
    line_tokens = chunker.tokenizer.count_tokens(line)
    
    # Verify our test setup
    assert line_tokens <= MAX_TOKENS, f"Line should fit in chunk: {line_tokens} <= {MAX_TOKENS}"
    assert line_tokens + chunker.prefix_len > MAX_TOKENS, (
        f"Line with prefix should NOT fit: {line_tokens} + {chunker.prefix_len} > {MAX_TOKENS}"
    )
    
    lines = [line]
    chunks = chunker.chunk_text(lines)
    
    # With omit_prefix_on_overflow=True and first line overflowing:
    # - First chunk should be the prefix alone (to ensure it's visible)
    # - Second chunk should be the line content without prefix
    assert len(chunks) == 2, "Should have 2 chunks: prefix chunk + content chunk"
    assert chunks[0] == prefix, "First chunk should be the prefix alone"
    assert not chunks[1].startswith(prefix), "Second chunk should NOT start with prefix"
    assert line.strip() in chunks[1], "Line content should be in second chunk"


def test_omit_prefix_on_overflow_line_too_large(default_tokenizer):
    """Test that line is still split when it exceeds max_tokens, regardless of omit_prefix_on_overflow."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Create a line that's larger than max_tokens even without prefix
    long_line = "word " * (MAX_TOKENS * 2)
    line_tokens = chunker.tokenizer.count_tokens(long_line)
    
    # Verify line is too large
    assert line_tokens > MAX_TOKENS, f"Line should exceed max_tokens: {line_tokens} > {MAX_TOKENS}"
    
    lines = [long_line]
    chunks = chunker.chunk_text(lines)
    
    # Line should be split regardless of omit_prefix_on_overflow
    assert len(chunks) > 1, "Line should be split when it exceeds max_tokens"
    
    # Verify each chunk respects token limit
    for chunk in chunks:
        token_count = chunker.tokenizer.count_tokens(chunk)
        assert token_count <= MAX_TOKENS, f"Chunk should respect token limit: {token_count} <= {MAX_TOKENS}"


def test_omit_prefix_on_overflow_multiple_lines(default_tokenizer):
    """Test omit_prefix_on_overflow with multiple lines of varying sizes."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Line 1: Small, fits with prefix
    line1 = "Small line\n"
    # Line 2: Medium, fits without prefix but not with prefix
    line2 = "word " * (MAX_TOKENS - 1) + "\n"
    # Line 3: Small again, fits with prefix
    line3 = "Another small line\n"
    
    lines = [line1, line2, line3]
    chunks = chunker.chunk_text(lines)
    
    # Verify we have chunks
    assert len(chunks) > 0
    
    # First chunk should have prefix and line1
    assert chunks[0].startswith(prefix)
    assert line1.strip() in chunks[0]
    
    # line2 should be in a chunk without prefix (due to overflow)
    line2_found = False
    for chunk in chunks:
        if line2.strip() in chunk:
            line2_found = True
            # This chunk should NOT start with prefix
            assert not chunk.startswith(prefix), "Line2 chunk should not have prefix due to overflow"
            break
    assert line2_found, "Line2 should be found in chunks"
    
    # line3 should be in a chunk with prefix (it's small enough)
    line3_found = False
    for chunk in chunks:
        if line3.strip() in chunk:
            line3_found = True
            # This chunk should start with prefix
            assert chunk.startswith(prefix), "Line3 chunk should have prefix"
            break
    assert line3_found, "Line3 should be found in chunks"


def test_omit_prefix_on_overflow_no_prefix(default_tokenizer):
    """Test that omit_prefix_on_overflow has no effect when there's no prefix."""
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix="",
        omit_prefix_on_overflow=True,
    )
    
    line = "word " * (MAX_TOKENS - 1)
    lines = [line]
    chunks = chunker.chunk_text(lines)
    
    # Should work normally without prefix
    assert len(chunks) == 1
    assert line in chunks[0]


def test_omit_prefix_on_overflow_with_line_splitting(default_tokenizer):
    """Test that overflow chunks from split lines don't have prefix when omit_prefix_on_overflow=True."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Create a line that's too long even without prefix (will be split)
    long_line = "word " * (MAX_TOKENS * 2)
    line_tokens = chunker.tokenizer.count_tokens(long_line)
    
    # Verify line is too large even without prefix
    assert line_tokens > MAX_TOKENS, f"Line should exceed max_tokens: {line_tokens} > {MAX_TOKENS}"
    
    lines = [long_line]
    chunks = chunker.chunk_text(lines)
    
    # Line should be split into multiple chunks
    assert len(chunks) > 1, "Line should be split when it exceeds max_tokens"
    
    # With omit_prefix_on_overflow=True:
    # - First chunk may have prefix (if it was already in current when line started)
    # - Subsequent overflow chunks should NOT have the prefix
    for i, chunk in enumerate(chunks):
        token_count = chunker.tokenizer.count_tokens(chunk)
        assert token_count <= MAX_TOKENS, f"Chunk {i} should respect token limit: {token_count} <= {MAX_TOKENS}"
    
    # At least the overflow chunks (after the first) should not have prefix
    if len(chunks) > 1:
        for i in range(1, len(chunks)):
            assert not chunks[i].startswith(prefix), f"Overflow chunk {i} should NOT have prefix with omit_prefix_on_overflow=True"


def test_omit_prefix_on_overflow_false_with_line_splitting(default_tokenizer):
    """Test that overflow chunks from split lines DO have prefix when omit_prefix_on_overflow=False."""
    prefix = "PREFIX: "
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=False,  # Default behavior
    )
    
    # Create a line that's too long even without prefix (will be split)
    long_line = "word " * (MAX_TOKENS * 2)
    line_tokens = chunker.tokenizer.count_tokens(long_line)
    
    # Verify line is too large even without prefix
    assert line_tokens > MAX_TOKENS, f"Line should exceed max_tokens: {line_tokens} > {MAX_TOKENS}"
    
    lines = [long_line]
    chunks = chunker.chunk_text(lines)
    
    # Line should be split into multiple chunks
    assert len(chunks) > 1, "Line should be split when it exceeds max_tokens"
    
    # With omit_prefix_on_overflow=False, all chunks should have the prefix
    for i, chunk in enumerate(chunks):
        token_count = chunker.tokenizer.count_tokens(chunk)
        assert token_count <= MAX_TOKENS, f"Chunk {i} should respect token limit: {token_count} <= {MAX_TOKENS}"
        
        # All chunks should have the prefix when omit_prefix_on_overflow=False
        assert chunk.startswith(prefix), f"Chunk {i} should have prefix with omit_prefix_on_overflow=False"

def test_omit_prefix_on_overflow_warning(default_tokenizer):
    """Test that a warning is issued once when prefix is actually omitted."""
    prefix = "PREFIX: "
    
    # Create chunker with omit_prefix_on_overflow=True
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Create a line that fits without prefix but not with prefix
    line = "word " * (MAX_TOKENS - 1)
    line_tokens = chunker.tokenizer.count_tokens(line)
    
    # Verify test setup
    assert line_tokens <= MAX_TOKENS
    assert line_tokens + chunker.prefix_len > MAX_TOKENS
    
    # Should warn once when prefix is omitted
    with pytest.warns(UserWarning, match="Prefix omitted for at least one line"):
        lines = [line, line, line]  # Multiple lines that would cause omission
        chunks = chunker.chunk_text(lines)
    
    # Verify chunks were created
    assert len(chunks) > 0


def test_omit_prefix_on_overflow_no_warning_when_not_omitted(default_tokenizer):
    """Test that no warning is issued when prefix is never omitted."""
    prefix = "PREFIX: "
    
    # Create chunker with omit_prefix_on_overflow=True
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Create lines that fit with prefix (no omission needed)
    small_line = "Small line\n"
    
    # Should NOT warn since prefix is never omitted
    import warnings as warnings_module
    with warnings_module.catch_warnings(record=True) as warning_list:
        warnings_module.simplefilter("always")
        lines = [small_line, small_line, small_line]
        chunks = chunker.chunk_text(lines)
        
        # Filter for UserWarnings about prefix omission
        prefix_warnings = [w for w in warning_list if "Prefix omitted" in str(w.message)]
    
    # Verify no prefix omission warnings were issued
    assert len(prefix_warnings) == 0
    
    # Verify chunks were created with prefix
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.startswith(prefix)


def test_omit_prefix_on_overflow_warning_on_split_line(default_tokenizer):
    """Test that warning is issued when prefix is omitted for overflow chunks from split lines."""
    prefix = "PREFIX: "
    
    # Create chunker with omit_prefix_on_overflow=True
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    # Create a line that's too long even without prefix (will be split)
    long_line = "word " * (MAX_TOKENS * 2)
    line_tokens = chunker.tokenizer.count_tokens(long_line)
    
    # Verify line is too large even without prefix
    assert line_tokens > MAX_TOKENS
    
    # Should warn once when prefix is omitted for overflow chunks
    with pytest.warns(UserWarning, match="Prefix omitted for at least one line"):
        lines = [long_line]
        chunks = chunker.chunk_text(lines)
    
    # Verify line was split into multiple chunks
    assert len(chunks) > 1
    
    # With omit_prefix_on_overflow=True, overflow chunks should NOT have prefix
    # (first chunk may or may not have prefix depending on initial state)
    for i in range(1, len(chunks)):
        assert not chunks[i].startswith(prefix), f"Overflow chunk {i} should not have prefix"


def test_omit_prefix_on_overflow_all_lines_overflow(default_tokenizer):
    """
    Test edge case where omit_prefix_on_overflow=True and ALL lines overflow with prefix.
    """
    prefix = "HEADER: Column1 | Column2 | Column3\n"
    
    chunker = LineBasedTokenChunker(
        tokenizer=default_tokenizer,
        prefix=prefix,
        omit_prefix_on_overflow=True,
    )
    
    prefix_tokens = chunker.tokenizer.count_tokens(prefix)
    assert prefix_tokens < MAX_TOKENS, f"Prefix should fit in chunk: {prefix_tokens} < {MAX_TOKENS}"
    assert chunker.prefix_len > 0, "prefix_len should be > 0 when prefix fits"
    
    # Each line should be close to max_tokens so that prefix + line > max_tokens
    line1 = "word " * (MAX_TOKENS - chunker.prefix_len + 1) + "\n"
    line2 = "data " * (MAX_TOKENS - chunker.prefix_len + 1) + "\n"
    line3 = "text " * (MAX_TOKENS - chunker.prefix_len + 1) + "\n"
    
    for i, line in enumerate([line1, line2, line3], 1):
        line_tokens = chunker.tokenizer.count_tokens(line)
        assert line_tokens <= MAX_TOKENS, f"Line {i} should fit alone: {line_tokens} <= {MAX_TOKENS}"
        assert line_tokens + chunker.prefix_len > MAX_TOKENS, (
            f"Line {i} with prefix should overflow: {line_tokens} + {chunker.prefix_len} > {MAX_TOKENS}"
        )
    
    lines = [line1, line2, line3]
    
    # Chunk the text
    with pytest.warns(UserWarning, match="Prefix omitted for at least one line"):
        chunks = chunker.chunk_text(lines)
    
    assert len(chunks) > 0, "Should create at least one chunk"
    
    # Check if prefix appears in any chunk
    prefix_found = any(prefix.strip() in chunk for chunk in chunks)
    
    # This assertion will FAIL with the current implementation, demonstrating the bug
    assert prefix_found, (
        "Prefix should appear in at least one chunk even when all lines omit it. "
        "This ensures headers/context are visible. "
        f"Chunks: {chunks}"
    )
    
    # Additionally verify that content lines are present (without prefix)
    for line in lines:
        line_found = any(line.strip() in chunk for chunk in chunks)
        assert line_found, f"Line content should be preserved: {line.strip()}"

