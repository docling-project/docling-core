"""HuggingFace tokenization."""

import json
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Final

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
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

# transformers sets `model_max_length` to a very large integer when the
# tokenizer files do not declare a limit.
_MAX_PLAUSIBLE_MODEL_LENGTH: Final = 1_000_000

# The files that the common tokenizers need. Model weights are not in this list.
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
    "tokenizer.model.v*",
    "*.tiktoken",
    "chat_template*",
    _SENTENCE_BERT_CONFIG,
)

# The (artifacts path, model name) pairs for which the warning was logged.
_warned_missing: set[tuple[str, str]] = set()


def _is_repo_id(name: str) -> bool:
    # The Hugging Face rules also reject `..`, `--`, an absolute path and more
    # than one `/`, so a valid repo id maps to exactly one folder name.
    try:
        validate_repo_id(name)
    except HFValidationError:
        return False
    return True


def _warn_missing_once(model_name: str, artifacts_path: Path) -> None:
    key = (str(artifacts_path), model_name)
    if key in _warned_missing:
        return
    _warned_missing.add(key)
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


def download_tokenizer(repo_id: str, output_dir: str | PathLike[str], revision: str | None = None) -> Path:
    """Download the tokenizer files of a Hugging Face repo into an artifacts directory.

    Only the tokenizer and configuration files are downloaded, not the model
    weights. The result can be loaded with
    `HuggingFaceTokenizer.from_pretrained(repo_id, artifacts_path=output_dir)`.

    The list of downloaded files is a best effort for the common tokenizers. If
    a tokenizer needs a file that is not in the list, copy that file from the
    repo into the returned folder.

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


def _read_max_seq_length(config_path: Path) -> int:
    with config_path.open(encoding="utf-8") as f:
        return int(json.load(f)["max_seq_length"])


def _resolve_local_max_tokens(local_dir: Path, tokenizer: "PreTrainedTokenizerBase") -> int:
    """Resolve the maximum number of tokens from a local tokenizer folder."""
    config_path = local_dir / _SENTENCE_BERT_CONFIG
    if config_path.is_file():
        return _read_max_seq_length(config_path)
    model_max_length = int(tokenizer.model_max_length)
    if 0 < model_max_length <= _MAX_PLAUSIBLE_MODEL_LENGTH:
        return model_max_length
    raise ValueError(f"'{local_dir}' does not declare a maximum sequence length")


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
                # try to use SentenceTransformers-specific config as that seems to be
                # reliable (whenever available)
                config_path = hf_hub_download(
                    repo_id=self.tokenizer.name_or_path,
                    filename=_SENTENCE_BERT_CONFIG,
                )
                self.max_tokens = _read_max_seq_length(Path(config_path))
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
        artifacts_path: str | PathLike[str] | None = None,
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
                with `local_files_only=True`, also when the caller passes a
                different value for `local_files_only`. If the tokenizer folder
                is missing or empty, a warning is logged one time and
                `model_name` is loaded in the usual way.
            **kwargs: Additional arguments for `AutoTokenizer.from_pretrained`.

        Raises:
            RuntimeError: If the tokenizer folder in `artifacts_path` has files
                but cannot be loaded, or if `max_tokens` cannot be resolved.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _TRANSFORMERS_IMPORT_ERROR
        if artifacts_path is None:
            artifacts_path = settings.artifacts_path

        # The folder that the tokenizer is loaded from, when this is known
        # without a guess. It is then also the source of `max_tokens`.
        local_dir: Path | None = None
        if not isinstance(model_name, str):
            # A path object always names a local folder.
            local_dir = Path(model_name)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, **kwargs)
        elif (artifacts_dir := cls._find_in_artifacts(model_name, artifacts_path)) is not None:
            local_dir = artifacts_dir
            kwargs["local_files_only"] = True
            try:
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=artifacts_dir, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"The tokenizer folder '{artifacts_dir}' could not be loaded. It can be incomplete. "
                    "Delete it and download it again with docling_core.transforms.chunker.tokenizer."
                    f"huggingface.download_tokenizer(repo_id='{model_name}', output_dir='{artifacts_path}')."
                ) from e
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, **kwargs)

        if max_tokens is None and local_dir is not None:
            try:
                max_tokens = _resolve_local_max_tokens(local_dir, tokenizer)
            except Exception as e:
                raise RuntimeError("max_tokens could not be determined automatically; please set explicitly.") from e
        my_kwargs: dict[str, Any] = {"tokenizer": tokenizer}
        if max_tokens is not None:
            my_kwargs["max_tokens"] = max_tokens
        return cls(**my_kwargs)

    @staticmethod
    def _find_in_artifacts(model_name: str, artifacts_path: str | PathLike[str] | None) -> Path | None:
        """Get the folder of `model_name` in the artifacts directory, if it has files."""
        # Only a repo id is looked up in the artifacts directory. Any other
        # value, such as `../x` or an absolute path, is never joined to it.
        if artifacts_path is None or not _is_repo_id(model_name):
            return None
        local_dir = Path(artifacts_path) / get_tokenizer_folder(model_name)
        # An empty folder is what an interrupted download can leave behind.
        if local_dir.is_dir() and any(local_dir.iterdir()):
            return local_dir
        _warn_missing_once(model_name, Path(artifacts_path))
        return None

    def get_tokenizer(self):
        """Get underlying tokenizer object."""
        return self.tokenizer


def get_default_tokenizer(artifacts_path: str | PathLike[str] | None = None) -> HuggingFaceTokenizer:
    """Get default tokenizer instance.

    Args:
        artifacts_path: A directory with tokenizers that were downloaded before.
            Defaults to `settings.artifacts_path`.
    """
    return HuggingFaceTokenizer.from_pretrained(model_name=_DEFAULT_MODEL_NAME, artifacts_path=artifacts_path)
