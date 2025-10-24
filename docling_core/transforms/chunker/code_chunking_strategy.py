"""Code chunking strategy implementations for different programming languages."""

from typing import Any, Dict, Iterator, Optional

from docling_core.transforms.chunker.base_code_chunker import _CodeChunker
from docling_core.transforms.chunker.code_chunk_utils.utils import Language
from docling_core.transforms.chunker.hierarchical_chunker import (
    CodeChunk,
    CodeChunkType,
    CodeDocMeta,
)
from docling_core.transforms.chunker.language_code_chunkers import (
    _CFunctionChunker,
    _JavaFunctionChunker,
    _JavaScriptFunctionChunker,
    _PythonFunctionChunker,
    _TypeScriptFunctionChunker,
)
from docling_core.types.doc.base import Size
from docling_core.types.doc.document import (
    CodeItem,
    DoclingDocument,
    DocumentOrigin,
    PageItem,
)
from docling_core.utils.legacy import _create_hash


class LanguageDetector:
    """Utility class for detecting programming languages from code content and file extensions."""

    @staticmethod
    def detect_from_extension(filename: Optional[str]) -> Optional[Language]:
        """Detect language from file extension."""
        if not filename:
            return None

        filename_lower = filename.lower()

        for language in Language:
            for ext in language.file_extensions():
                if filename_lower.endswith(ext):
                    return language
        return None

    @staticmethod
    def detect_from_content(code_text: str) -> Optional[Language]:
        """Detect language from code content using heuristics."""
        if not code_text:
            return None

        code_lower = code_text.lower().strip()

        if any(
            pattern in code_lower
            for pattern in [
                "def ",
                "import ",
                "from ",
                'if __name__ == "__main__"',
                "print(",
                "lambda ",
                "yield ",
                "async def",
            ]
        ) and not any(
            pattern in code_lower
            for pattern in [
                "public class",
                "private ",
                "protected ",
                "package ",
                "package main",
                "func main()",
                'import "fmt"',
                "chan ",
                "interface{}",
                "go func",
                "defer ",
                ":= ",
            ]
        ):
            return Language.PYTHON

        if any(
            pattern in code_lower
            for pattern in [
                "package main",
                "func main()",
                'import "fmt"',
                'import "os"',
                "chan ",
                "interface{}",
                "go func",
                "defer ",
                ":= ",
            ]
        ) and not any(
            pattern in code_lower
            for pattern in [
                "public class",
                "import java.",
                "System.out.println",
                "extends ",
                "implements ",
            ]
        ):
            return None

        if any(
            pattern in code_lower
            for pattern in [
                "public class",
                "package ",
                "import java.",
                "public static void main",
                "extends ",
                "implements ",
                "String[]",
                "System.out.println",
            ]
        ) and not any(
            pattern in code_lower
            for pattern in ["package main", "func main()", "chan ", "interface{}"]
        ):
            return Language.JAVA

        if any(
            pattern in code_lower
            for pattern in [
                ": string",
                ": number",
                ": boolean",
                "interface ",
                "type ",
                "enum ",
                "public ",
                "private ",
                "protected ",
            ]
        ):
            return Language.TYPESCRIPT

        if any(
            pattern in code_lower
            for pattern in [
                "function ",
                "const ",
                "let ",
                "var ",
                "=>",
                "require(",
                "module.exports",
                "export ",
                "import ",
                "console.log",
            ]
        ):
            return Language.JAVASCRIPT

        if any(
            pattern in code_lower
            for pattern in [
                "#include",
                "int main(",
                "void ",
                "char ",
                "float ",
                "double ",
                "struct ",
                "#define",
                "printf(",
                "scanf(",
            ]
        ):
            return Language.C

        return None

    @staticmethod
    def detect_language(
        code_text: str, filename: Optional[str] = None
    ) -> Optional[Language]:
        """Detect language from both filename and content."""
        if filename:
            lang = LanguageDetector.detect_from_extension(filename)
            if lang:
                return lang
            return None

        return LanguageDetector.detect_from_content(code_text)


class CodeChunkingStrategyFactory:
    """Factory for creating language-specific code chunking strategies."""

    @staticmethod
    def create_chunker(language: Language, **kwargs: Any) -> _CodeChunker:
        """Create a language-specific code chunker."""
        chunker_map = {
            Language.PYTHON: _PythonFunctionChunker,
            Language.TYPESCRIPT: _TypeScriptFunctionChunker,
            Language.JAVASCRIPT: _JavaScriptFunctionChunker,
            Language.C: _CFunctionChunker,
            Language.JAVA: _JavaFunctionChunker,
        }

        chunker_class = chunker_map.get(language)
        if not chunker_class:
            raise ValueError(f"No chunker available for language: {language}")

        return chunker_class(**kwargs)


class DefaultCodeChunkingStrategy:
    """Default implementation of CodeChunkingStrategy that uses language detection and appropriate chunkers."""

    def __init__(self, **chunker_kwargs: Any):
        """Initialize the strategy with optional chunker parameters."""
        self.chunker_kwargs = chunker_kwargs
        self._chunker_cache: Dict[Language, _CodeChunker] = {}

    def _get_chunker(self, language: Language) -> _CodeChunker:
        """Get or create a chunker for the given language."""
        if language not in self._chunker_cache:
            self._chunker_cache[language] = CodeChunkingStrategyFactory.create_chunker(
                language, **self.chunker_kwargs
            )
        return self._chunker_cache[language]

    def chunk_code_item(
        self,
        code_text: str,
        language: Language,
        original_doc=None,
        original_item=None,
        **kwargs: Any,
    ) -> Iterator[CodeChunk]:
        """Chunk a single code item using the appropriate language chunker."""
        if not code_text.strip():
            return

        chunker = self._get_chunker(language)

        if original_doc and original_doc.origin:
            filename = original_doc.origin.filename or "code_chunk"
            mimetype = original_doc.origin.mimetype or "text/plain"
            binary_hash = _create_hash(code_text)
            uri = getattr(original_doc.origin, "uri", None)
        else:
            filename = "code_chunk"
            mimetype = "text/plain"
            binary_hash = _create_hash(code_text)
            uri = None

        if original_item and hasattr(original_item, "self_ref"):
            self_ref = original_item.self_ref
        else:
            self_ref = "#/texts/0"

        code_item = CodeItem(text=code_text, self_ref=self_ref, orig=code_text)

        doc = DoclingDocument(
            name=filename,
            texts=[code_item],
            pages={0: PageItem(page_no=0, size=Size(width=612.0, height=792.0))},
            origin=DocumentOrigin(
                filename=filename, mimetype=mimetype, binary_hash=binary_hash, uri=uri
            ),
        )

        yield from chunker.chunk(doc, **kwargs)


class NoOpCodeChunkingStrategy:
    """No-operation code chunking strategy that returns the original code as a single chunk."""

    def chunk_code_item(
        self,
        code_text: str,
        language: Language,
        original_doc=None,
        original_item=None,
        **kwargs: Any,
    ) -> Iterator[CodeChunk]:
        """Return the code as a single chunk without further processing."""
        if not code_text.strip():
            return

        meta = CodeDocMeta(
            chunk_type=CodeChunkType.CODE_BLOCK,
            start_line=1,
            end_line=len(code_text.splitlines()),
        )

        yield CodeChunk(text=code_text, meta=meta)
