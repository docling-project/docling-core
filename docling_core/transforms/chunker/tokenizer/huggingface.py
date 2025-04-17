"""HuggingFace tokenization."""

import sys
from os import PathLike
from typing import Optional, Union

from pydantic import ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )


class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace tokenizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: PreTrainedTokenizerBase
    max_tokens: int = None  # type: ignore[assignment]

    @model_validator(mode="after")
    def _patch(self) -> Self:
        if hasattr(self.tokenizer, "model_max_length"):
            model_max_tokens: PositiveInt = TypeAdapter(PositiveInt).validate_python(
                self.tokenizer.model_max_length
            )
            user_max_tokens = self.max_tokens or sys.maxsize
            self.max_tokens = min(model_max_tokens, user_max_tokens)
        elif self.max_tokens is None:
            raise ValueError(
                "max_tokens must be defined as model does not define model_max_length"
            )
        return self

    def count_tokens(self, text: str):
        """Get number of tokens for given text."""
        return len(self.tokenizer.tokenize(text=text))

    def get_max_tokens(self):
        """Get maximum number of tokens allowed."""
        return self.max_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name: Union[str, PathLike],
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Self:
        """Create tokenizer from model name."""
        my_kwargs = {
            "tokenizer": AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name, **kwargs
            ),
        }
        if max_tokens is not None:
            my_kwargs["max_tokens"] = max_tokens
        return cls(**my_kwargs)

    def get_tokenizer(self):
        """Get underlying tokenizer object."""
        return self.tokenizer
