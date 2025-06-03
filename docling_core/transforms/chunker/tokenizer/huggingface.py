"""HuggingFace tokenization."""

import json
from os import PathLike
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from pydantic import ConfigDict, model_validator
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
        if self.max_tokens is None:
            try:
                # try to use SentenceTransformers-specific config as that seems to be
                # reliable (whenever available)
                config_name = "sentence_bert_config.json"
                config_path = hf_hub_download(
                    repo_id=self.tokenizer.name_or_path,
                    filename=config_name,
                )
                with open(config_path) as f:
                    data = json.load(f)
                self.max_tokens = int(data["max_seq_length"])
            except Exception as e:
                raise RuntimeError(
                    "max_tokens could not be determined automatically; please set "
                    "explicitly."
                ) from e
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
