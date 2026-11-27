"""Tests for loading chunking tokenizers from an artifacts directory."""

import json
import logging
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from docling_core.transforms.chunker.tokenizer import huggingface as hf_mod
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
    download_tokenizer,
    get_tokenizer_folder,
)
from docling_core.utils.settings import CoreSettings, settings

REPO_ID = "acme/tiny-tokenizer"


def _save_tokenizer(folder: Path, max_seq_length: int | None = None, **kwargs) -> None:
    inner = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]", **kwargs).save_pretrained(folder)
    if max_seq_length is not None:
        (folder / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": max_seq_length}))


@pytest.fixture
def no_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the code asks the Hugging Face Hub for a file."""

    def _fail(*args, **kwargs):
        raise AssertionError("unexpected access to the Hugging Face Hub")

    monkeypatch.setattr(hf_mod, "hf_hub_download", _fail)
    monkeypatch.setattr(hf_mod, "snapshot_download", _fail)


@pytest.fixture
def loaded_names(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the arguments that reach `AutoTokenizer.from_pretrained`."""
    calls: list[dict] = []

    def _record(**kwargs):
        calls.append(kwargs)
        return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))

    monkeypatch.setattr(hf_mod.AutoTokenizer, "from_pretrained", _record)
    return calls


def test_loads_from_artifacts_path_without_hub(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / "acme--tiny-tokenizer", max_seq_length=128)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    assert tokenizer.get_max_tokens() == 128
    assert tokenizer.count_tokens("hello world") == 2


def test_uses_artifacts_path_from_settings(tmp_path: Path, no_hub: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_tokenizer(tmp_path / "acme--tiny-tokenizer", max_seq_length=64)
    monkeypatch.setattr(settings, "artifacts_path", tmp_path)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID)

    assert tokenizer.get_max_tokens() == 64


def test_settings_share_the_docling_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCLINGCORE_ARTIFACTS_PATH", raising=False)
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path / "shared"))
    assert CoreSettings().artifacts_path == tmp_path / "shared"

    monkeypatch.setenv("DOCLINGCORE_ARTIFACTS_PATH", str(tmp_path / "core"))
    assert CoreSettings().artifacts_path == tmp_path / "core"


def test_max_tokens_falls_back_to_model_max_length(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / "acme--tiny-tokenizer", model_max_length=256)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    assert tokenizer.get_max_tokens() == 256


def test_local_tokenizer_without_declared_limit_needs_max_tokens(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / "acme--tiny-tokenizer")

    with pytest.raises(RuntimeError, match="max_tokens could not be determined"):
        HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path, max_tokens=32)
    assert tokenizer.get_max_tokens() == 32


def test_missing_tokenizer_warns_and_falls_back(
    tmp_path: Path, loaded_names: list[dict], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=hf_mod.__name__):
        HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path, max_tokens=8)

    assert loaded_names == [{"pretrained_model_name_or_path": REPO_ID}]
    assert "was not found in the artifacts path" in caplog.text
    assert "download_tokenizer" in caplog.text


@pytest.mark.parametrize("hostile_name", ["../secret", "acme/../../secret", "secret/..", ".."])
def test_name_is_never_joined_outside_the_artifacts_path(
    tmp_path: Path, loaded_names: list[dict], hostile_name: str
) -> None:
    artifacts_path = tmp_path / "artifacts"
    artifacts_path.mkdir()
    # A tokenizer next to the artifacts directory: a naive join would load it.
    _save_tokenizer(tmp_path / "secret", max_seq_length=128)

    HuggingFaceTokenizer.from_pretrained(model_name=hostile_name, artifacts_path=artifacts_path, max_tokens=8)

    assert loaded_names == [{"pretrained_model_name_or_path": hostile_name}]


@pytest.mark.parametrize("hostile_name", ["../secret", "/etc", "a/b/c", ""])
def test_download_rejects_an_invalid_repo_id(tmp_path: Path, no_hub: None, hostile_name: str) -> None:
    with pytest.raises(ValueError, match="Not a valid Hugging Face repo id"):
        download_tokenizer(repo_id=hostile_name, output_dir=tmp_path)
    assert not any(tmp_path.iterdir())


def test_tokenizer_folder_matches_the_docling_convention() -> None:
    assert get_tokenizer_folder("sentence-transformers/all-MiniLM-L6-v2") == "sentence-transformers--all-MiniLM-L6-v2"
