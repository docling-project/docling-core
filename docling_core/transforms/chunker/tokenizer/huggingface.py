"""HuggingFace tokenization."""

import json
import logging
import re
from os import PathLike
from pathlib import Path
from typing import Any, Final

from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.utils.settings import settings

_TRANSFORMERS_AVAILABLE: bool = False
_TRANSFORMERS_IMPORT_ERROR: ImportError | None = None
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    _TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    _TRANSFORMERS_IMPORT_ERROR = e

_INSTALL_HINT = (
    "The 'transformers' package is required by HuggingFaceTokenizer. "
    "Install it with `pip install 'docling-core[chunking]'`."
)

_logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME: Final = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE_BERT_CONFIG: Final = "sentence_bert_config.json"

# A Hugging Face repo id is `name` or `namespace/name`. Each part starts with an
# alphanumeric character, so a part can never be `.` or `..`.
_REPO_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")

# transformers sets `model_max_length` to a very large integer when the
# tokenizer files do not declare a limit.
_MAX_PLAUSIBLE_MODEL_LENGTH: Final = 1_000_000

# The files that a tokenizer needs. Model weights are not in this list.
_TOKENIZER_FILE_PATTERNS: Final = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "*.model",
    _SENTENCE_BERT_CONFIG,
)


def _is_repo_id(name: str) -> bool:
    return _REPO_ID_RE.fullmatch(name) is not None and ".." not in name


def get_tokenizer_folder(repo_id: str) -> str:
    """Get the folder name of a tokenizer inside an artifacts directory.

    The name follows the convention that docling uses for Hugging Face models,
    so that one artifacts directory can hold the models and the tokenizers.

    Args:
        repo_id: The Hugging Face repo id, for example
            `sentence-transformers/all-MiniLM-L6-v2`.

    Returns:
        The folder name, for example `sentence-transformers--all-MiniLM-L6-v2`.

    Raises:
        ValueError: If `repo_id` is not a valid Hugging Face repo id.
    """
    if not _is_repo_id(repo_id):
        raise ValueError(f"Not a valid Hugging Face repo id: {repo_id!r}")
    return repo_id.replace("/", "--")


def download_tokenizer(repo_id: str, output_dir: Path, revision: str | None = None) -> Path:
    """Download the tokenizer files of a Hugging Face repo into an artifacts directory.

    Only the tokenizer and configuration files are downloaded, not the model
    weights. The result can be loaded with
    `HuggingFaceTokenizer.from_pretrained(repo_id, artifacts_path=output_dir)`.

    Args:
        repo_id: The Hugging Face repo id.
        output_dir: The artifacts directory.
        revision: The revision to download. Defaults to the main revision.

    Returns:
        The folder that contains the tokenizer files.

    Raises:
        ValueError: If `repo_id` is not a valid Hugging Face repo id.
    """
    local_dir = Path(output_dir) / get_tokenizer_folder(repo_id)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=list(_TOKENIZER_FILE_PATTERNS),
    )
    return local_dir


def _resolve_max_tokens(tokenizer: "PreTrainedTokenizerBase") -> int:
    name_or_path = str(tokenizer.name_or_path)
    local_dir = Path(name_or_path)
    if local_dir.is_dir():
        # The tokenizer comes from a local folder: do not use the network.
        config_path = local_dir / _SENTENCE_BERT_CONFIG
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                return int(json.load(f)["max_seq_length"])
        model_max_length = int(tokenizer.model_max_length)
        if 0 < model_max_length <= _MAX_PLAUSIBLE_MODEL_LENGTH:
            return model_max_length
        raise ValueError(f"'{local_dir}' does not declare a maximum sequence length")

    # try to use SentenceTransformers-specific config as that seems to be
    # reliable (whenever available)
    config_path = Path(hf_hub_download(repo_id=name_or_path, filename=_SENTENCE_BERT_CONFIG))
    with config_path.open(encoding="utf-8") as f:
        return int(json.load(f)["max_seq_length"])


class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace tokenizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: "PreTrainedTokenizerBase"
    max_tokens: int = None  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def _check_deps(cls, data: Any) -> Any:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _TRANSFORMERS_IMPORT_ERROR
        return data

    @model_validator(mode="after")
    def _patch(self) -> Self:
        if self.max_tokens is None:
            try:
                self.max_tokens = _resolve_max_tokens(self.tokenizer)
            except Exception as e:
                raise RuntimeError("max_tokens could not be determined automatically; please set explicitly.") from e
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
        model_name: str | PathLike,
        max_tokens: int | None = None,
        artifacts_path: Path | None = None,
        **kwargs,
    ) -> Self:
        """Create tokenizer from model name.

        Args:
            model_name: A Hugging Face repo id or a path to a local tokenizer folder.
            max_tokens: The maximum number of tokens. If not set, the limit is
                resolved from the tokenizer files.
            artifacts_path: A directory with tokenizers that were downloaded
                before, see `download_tokenizer`. Defaults to
                `settings.artifacts_path`. If the directory contains the
                tokenizer of `model_name`, the tokenizer is loaded from there
                and the network is not used. Otherwise a warning is logged and
                `model_name` is loaded in the usual way.
            **kwargs: Additional arguments for `AutoTokenizer.from_pretrained`.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _TRANSFORMERS_IMPORT_ERROR
        name_or_path: str | PathLike = model_name
        if artifacts_path is None:
            artifacts_path = settings.artifacts_path
        # Only a repo id is looked up in the artifacts directory. Any other
        # value, such as `../x` or an absolute path, is never joined to it.
        if artifacts_path is not None and isinstance(model_name, str) and _is_repo_id(model_name):
            local_dir = Path(artifacts_path) / get_tokenizer_folder(model_name)
            if local_dir.is_dir():
                name_or_path = local_dir
                kwargs["local_files_only"] = True
            else:
                _logger.warning(
                    "Tokenizer '%s' was not found in the artifacts path '%s'. "
                    "Loading it from the Hugging Face Hub or its local cache instead. "
                    "To make it available offline, run "
                    "docling_core.transforms.chunker.tokenizer.huggingface."
                    "download_tokenizer(repo_id='%s', output_dir='%s').",
                    model_name,
                    artifacts_path,
                    model_name,
                    artifacts_path,
                )
        my_kwargs: dict[str, Any] = {
            "tokenizer": AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name_or_path, **kwargs),
        }
        if max_tokens is not None:
            my_kwargs["max_tokens"] = max_tokens
        return cls(**my_kwargs)

    def get_tokenizer(self):
        """Get underlying tokenizer object."""
        return self.tokenizer


def get_default_tokenizer(artifacts_path: Path | None = None) -> HuggingFaceTokenizer:
    """Get default tokenizer instance.

    Args:
        artifacts_path: A directory with tokenizers that were downloaded before.
            Defaults to `settings.artifacts_path`.
    """
    return HuggingFaceTokenizer.from_pretrained(model_name=_DEFAULT_MODEL_NAME, artifacts_path=artifacts_path)
