from pathlib import Path

import pytest

from docling_core.transforms.chunker import (
    CodeLanguageLabel,
    HierarchicalChunker,
    HybridChunker,
    StandardCodeChunkingStrategy,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.base import Size
from docling_core.types.doc.document import DoclingDocument, DocumentOrigin
from docling_core.types.doc.labels import DocItemLabel


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data" / "repo_chunking"


def test_code_chunking_strategies():
    """Test different code chunking strategies."""
    python_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n-1)
'''

    doc = DoclingDocument(name="test")
    code_item = doc.add_code(text=python_code, code_language=CodeLanguageLabel.PYTHON)
    strategy = StandardCodeChunkingStrategy(min_chunk_size=10, max_tokens=100)
    doc_ser = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            format_code_blocks=False,
        ),
    )
    chunks = list(strategy.chunk_code_item(item=code_item, doc=doc, doc_serializer=doc_ser))

    assert len(chunks) > 0
    for chunk in chunks:
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "meta")
        assert hasattr(chunk.meta, "chunk_type")


def test_hierarchical_chunker_integration():
    """Test HierarchicalChunker with and without code chunking strategy."""
    python_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

    doc = DoclingDocument(name="test")
    doc.add_page(page_no=0, size=Size(width=612.0, height=792.0))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Here's some Python code:",
        orig="Here's some Python code:",
    )
    doc.add_code(text=python_code, code_language=CodeLanguageLabel.PYTHON, orig=python_code)
    doc.origin = DocumentOrigin(filename="test.py", mimetype="text/x-python", binary_hash=12345)

    strategy = StandardCodeChunkingStrategy(min_chunk_size=50, max_tokens=1000)
    chunker_with_strategy = HierarchicalChunker(code_chunking_strategy=strategy)
    chunks_with_strategy = list(chunker_with_strategy.chunk(doc))

    assert len(chunks_with_strategy) > 0
    for chunk in chunks_with_strategy:
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "meta")

    chunker_without_strategy = HierarchicalChunker()
    chunks_without_strategy = list(chunker_without_strategy.chunk(doc))

    assert len(chunks_without_strategy) > 0
    for chunk in chunks_without_strategy:
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "meta")


def test_hybrid_chunker_with_code_files(test_data_dir):
    """Test that HybridChunker can process code files."""
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=512
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    python_file = test_data_dir / "sample.py"
    if not python_file.exists():
        pytest.skip("Python test file not found")

    doc = DoclingDocument(name="sample.py")
    doc.origin = DocumentOrigin(filename="sample.py", mimetype="text/x-python", binary_hash=12345)

    with open(python_file, "r", encoding="utf-8") as f:
        content = f.read()
    doc.add_code(text=content)

    chunks = list(chunker.chunk(dl_doc=doc))

    assert len(chunks) > 0
    for chunk in chunks:
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "meta")
        assert len(chunk.text) > 0


def test_unsupported_language_fallback(test_data_dir):
    """Test that unsupported languages fall back to regular text chunking."""
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=512
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    go_file = test_data_dir / "sample.go"
    if go_file.exists():
        doc = DoclingDocument(name="sample.go")
        doc.origin = DocumentOrigin(filename="sample.go", mimetype="text/plain", binary_hash=12345)

        with open(go_file, "r", encoding="utf-8") as f:
            content = f.read()
        doc.add_code(text=content)

        assert len(doc.texts) == 1
        assert doc.texts[0].code_language == CodeLanguageLabel.UNKNOWN

        chunks = list(chunker.chunk(dl_doc=doc))
        assert len(chunks) > 0

        all_text = " ".join(chunk.text for chunk in chunks)
        assert "package main" in all_text
        assert "func fibonacci" in all_text

    md_file = test_data_dir / "sample.md"
    if md_file.exists():
        doc = DoclingDocument(name="sample.md")
        doc.origin = DocumentOrigin(filename="sample.md", mimetype="text/plain", binary_hash=12345)

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        doc.add_code(text=content)

        assert len(doc.texts) == 1
        assert doc.texts[0].code_language == CodeLanguageLabel.UNKNOWN

        chunks = list(chunker.chunk(dl_doc=doc))
        assert len(chunks) > 0

        all_text = " ".join(chunk.text for chunk in chunks)
        assert "Sample Markdown File" in all_text
        assert "def hello()" in all_text


def test_repository_processing(test_data_dir):
    """Test processing multiple files from a repository."""
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=512
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    all_chunks = []
    for file_path in test_data_dir.glob("sample.*"):
        doc = DoclingDocument(name=file_path.name)
        doc.origin = DocumentOrigin(filename=file_path.name, mimetype="text/plain", binary_hash=12345)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        doc.add_code(text=content)

        chunks = list(chunker.chunk(dl_doc=doc))
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0

    for chunk in all_chunks:
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "meta")
        assert len(chunk.text) > 0
