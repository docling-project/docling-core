"""Code chunking strategy implementations for different programming languages."""

from typing import Any, Dict, Iterator, Optional

from typing_extensions import override

from docling_core.transforms.chunker.code_chunking._language_code_chunkers import (
    _CFunctionChunker,
    _CodeChunker,
    _JavaFunctionChunker,
    _JavaScriptFunctionChunker,
    _new_hash,
    _PythonFunctionChunker,
    _TypeScriptFunctionChunker,
)
from docling_core.transforms.chunker.code_chunking.code_chunk import (
    CodeChunk,
    CodeChunkType,
    CodeDocMeta,
)
from docling_core.transforms.chunker.hierarchical_chunker import (
    BaseCodeChunkingStrategy,
)
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.types.doc.document import CodeItem, DoclingDocument
from docling_core.types.doc.labels import CodeLanguageLabel

_INNER_CHUNKERS_BY_LANG: dict[CodeLanguageLabel, type[_CodeChunker]] = {
    CodeLanguageLabel.PYTHON: _PythonFunctionChunker,
    CodeLanguageLabel.TYPESCRIPT: _TypeScriptFunctionChunker,
    CodeLanguageLabel.JAVASCRIPT: _JavaScriptFunctionChunker,
    CodeLanguageLabel.C: _CFunctionChunker,
    CodeLanguageLabel.JAVA: _JavaFunctionChunker,
}


class StandardCodeChunkingStrategy(BaseCodeChunkingStrategy):
    """Standard implementation of CodeChunkingStrategy that uses appropriate chunkers."""

    def __init__(self, **chunker_kwargs: Any):
        """Initialize the strategy with optional chunker parameters."""
        self.chunker_kwargs = chunker_kwargs
        self._chunker_cache: Dict[CodeLanguageLabel, _CodeChunker] = {}

    def _get_chunker(self, language: CodeLanguageLabel) -> Optional[_CodeChunker]:
        """Get or create a chunker for the given language."""
        if chunker_instance := self._chunker_cache.get(language):
            return chunker_instance
        elif chunker_class := _INNER_CHUNKERS_BY_LANG.get(language):
            self._chunker_cache[language] = chunker_class(**self.chunker_kwargs)
            return self._chunker_cache[language]
        else:
            return None

    @override
    def chunk_code_item(
        self,
        *,
        item: CodeItem,
        doc: DoclingDocument,
        doc_serializer: Optional[BaseDocSerializer] = None,
        visited: Optional[set[str]] = None,
        **kwargs: Any,
    ) -> Iterator[CodeChunk]:
        """Chunk a single code item using the appropriate language chunker."""
        code_text = (
            doc_serializer.serialize(
                item=item,
                format_code_blocks=False,
                visited=visited,
            ).text
            if doc_serializer
            else item.text
        )
        if not code_text.strip():
            return

        if chunker := self._get_chunker(item.code_language):
            doc = DoclingDocument(name="", origin=doc.origin)
            doc.add_code(
                text=code_text, code_language=item.code_language, orig=code_text
            )
            yield from chunker.chunk(doc, **kwargs)
        else:  # if no inner chunker available for language, fall back to yielding a single code block chunk
            yield CodeChunk(
                text=code_text,
                meta=CodeDocMeta(
                    doc_items=[item],
                    origin=doc.origin,
                    start_line=1,
                    end_line=len(code_text.splitlines()),
                    sha256=_new_hash(code_text),
                    chunk_type=CodeChunkType.CODE_BLOCK,
                ),
            )
