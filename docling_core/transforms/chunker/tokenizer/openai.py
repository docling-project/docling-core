"""OpenAI tokenization."""

from pydantic import ConfigDict

from docling_core.transforms.chunker.hybrid_chunker import BaseTokenizer

try:
    import tiktoken
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking-openai' extra; to install, run: "
        "`pip install 'docling-core[chunking-openai]'`"
    )


class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: tiktoken.Encoding
    max_tokens: int

    def count_tokens(self, text: str) -> int:
        """Get number of tokens for given text."""
        return len(self.tokenizer.encode(text=text))

    def get_max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        return self.max_tokens

    def get_tokenizer(self) -> tiktoken.Encoding:
        """Get underlying tokenizer object."""
        return self.tokenizer
