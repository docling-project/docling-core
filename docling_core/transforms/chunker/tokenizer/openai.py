"""OpenAI tokenization."""

from typing import Any

from pydantic import ConfigDict, model_validator

from docling_core.transforms.chunker.hybrid_chunker import BaseTokenizer

_TIKTOKEN_AVAILABLE: bool = False
_TIKTOKEN_IMPORT_ERROR: ImportError | None = None
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError as e:
    _TIKTOKEN_IMPORT_ERROR = e

_INSTALL_HINT = (
    "The 'tiktoken' package is required by OpenAITokenizer. "
    "Install it with `pip install 'docling-core[chunking-openai]'`."
)


class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: "tiktoken.Encoding"
    max_tokens: int

    @model_validator(mode="before")
    @classmethod
    def _check_deps(cls, data: Any) -> Any:
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _TIKTOKEN_IMPORT_ERROR
        return data

    def count_tokens(self, text: str) -> int:
        """Get number of tokens for given text."""
        return len(self.tokenizer.encode(text=text))

    def get_max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        return self.max_tokens

    def get_tokenizer(self) -> "tiktoken.Encoding":
        """Get underlying tokenizer object."""
        return self.tokenizer
