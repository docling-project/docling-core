"""Define base classes for tokenization."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseTokenizer(BaseModel, ABC):
    """Base tokenizer class."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Get number of tokens for given text."""
        ...

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        ...

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Get underlying tokenizer object."""
        ...
