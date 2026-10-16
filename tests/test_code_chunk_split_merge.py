"""Regression tests for code chunk splitting / tail merging."""

from types import SimpleNamespace

from docling_core.transforms.chunker.code_chunking._language_code_chunkers import (
    _ChunkSizeProcessor,
    _PythonFunctionChunker,
)
from docling_core.transforms.chunker.code_chunking.code_chunk import (
    CodeChunk,
    CodeChunkType,
    CodeDocMeta,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import CodeLanguageLabel


class _FakeTokenizer(BaseTokenizer):
    """Deterministic offline tokenizer (char/4) to avoid network downloads."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_max_tokens(self) -> int:
        return 200

    def get_tokenizer(self):
        return None


def test_split_function_tail_merge_does_not_strip_body_chars():
    body_lines = [f"\tfinal_{i} = process({i})" for i in range(40)] + ["\teffort = final_39"]
    code = "def handle_many(items):\n" + "\n".join(body_lines) + "\n"

    doc = DoclingDocument(name="t")
    doc.add_code(text=code, code_language=CodeLanguageLabel.PYTHON, orig=code)

    # max_tokens small -> the function is split; min_chunk_size large enough to
    # force the small trailing part to be merged back into the previous chunk.
    chunker = _PythonFunctionChunker(max_tokens=200, min_chunk_size=15, tokenizer=_FakeTokenizer())
    chunks = list(chunker.chunk(doc))

    merged = "\n".join(c.text for c in chunks)
    # The tail line must survive intact (indented "effort = ..."). Previously
    # lstrip() with the signature string stripped leading body chars ("eff"),
    # leaving a non-indented "ort = ...".
    assert "\teffort = final_39" in merged
    assert "\nort = final_39" not in merged


def test_split_function_tail_merge_keeps_trailing_brace_of_body():
    # Brace-language chunkers use chunk_prefix=" {" and chunk_suffix="\n}".
    # When a small trailing part is merged, the previous chunk's suffix must be
    # stripped exactly. Previously rstrip("\n}") used a character set, so when
    # the penultimate chunk's last body line ended with "}" (a nested block),
    # that brace was stripped too.
    body_lines = [f"  v{i} = c({i});" for i in range(12)] + [
        "  if (x) { a(); }",
        "  last = 1;",
    ]
    code = "function f() {\n" + "\n".join(body_lines) + "\n"

    doc = DoclingDocument(name="t")
    item = doc.add_code(text=code, code_language=CodeLanguageLabel.JAVASCRIPT, orig=code)

    chunk = CodeChunk(
        text=code,
        meta=CodeDocMeta(
            doc_items=[item],
            sha256=1,
            part_name="f",
            chunk_type=CodeChunkType.FUNCTION,
        ),
    )

    brace_chunker = SimpleNamespace(chunk_prefix=" {", chunk_suffix="\n}")
    processor = _ChunkSizeProcessor(
        tokenizer=_FakeTokenizer(),
        max_tokens=40,
        min_chunk_size=30,
        chunker=brace_chunker,
    )
    chunks = [c for c, _ in processor._split_function_chunk(chunk, ranges=[])]
    merged = "\n".join(c.text for c in chunks)

    # The nested block's closing brace must survive the merge.
    assert "if (x) { a(); }" in merged
    assert "if (x) { a(); \n" not in merged
