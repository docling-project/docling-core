"""Tests for loading chunking tokenizers from an artifacts directory."""

import json
import logging
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from docling_core.transforms.chunker.code_chunking import _utils as code_utils
from docling_core.transforms.chunker.tokenizer import huggingface as hf_mod
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
    download_tokenizer,
    get_default_tokenizer,
)
from docling_core.utils.settings import CoreSettings, settings

REPO_ID = "acme/tiny-tokenizer"
REPO_FOLDER = "acme--tiny-tokenizer"
DEFAULT_FOLDER = "sentence-transformers--all-MiniLM-L6-v2"
MISSING_WARNING = "was not found in the artifacts path"


def _save_tokenizer(folder: Path, max_seq_length: int | None = None, **kwargs) -> None:
    inner = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]", **kwargs).save_pretrained(folder)
    if max_seq_length is not None:
        (folder / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": max_seq_length}))


@pytest.fixture(autouse=True)
def _reset_warned_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_mod, "_warned_missing", set())


@pytest.fixture
def no_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if this module asks the Hugging Face Hub for a file."""

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


def test_loads_repo_id_from_artifacts_path(tmp_path: Path, no_hub: None) -> None:
    # REPO_ID is not on the Hub: the load can succeed only through the artifacts path.
    _save_tokenizer(tmp_path / REPO_FOLDER, max_seq_length=128)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    assert tokenizer.get_max_tokens() == 128
    assert tokenizer.count_tokens("hello world") == 2


def test_artifacts_load_is_local_files_only(tmp_path: Path, loaded_names: list[dict]) -> None:
    _save_tokenizer(tmp_path / REPO_FOLDER)

    # The caller cannot switch the network back on for an artifacts load.
    HuggingFaceTokenizer.from_pretrained(
        model_name=REPO_ID, artifacts_path=str(tmp_path), max_tokens=8, local_files_only=False
    )

    assert loaded_names == [{"pretrained_model_name_or_path": tmp_path / REPO_FOLDER, "local_files_only": True}]


def test_uses_artifacts_path_from_settings(tmp_path: Path, no_hub: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_tokenizer(tmp_path / REPO_FOLDER, max_seq_length=64)
    monkeypatch.setattr(settings, "artifacts_path", tmp_path)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID)

    assert tokenizer.get_max_tokens() == 64


def test_settings_share_the_docling_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCLINGCORE_ARTIFACTS_PATH", raising=False)
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path / "shared"))
    assert CoreSettings().artifacts_path == tmp_path / "shared"

    monkeypatch.setenv("DOCLINGCORE_ARTIFACTS_PATH", str(tmp_path / "core"))
    assert CoreSettings().artifacts_path == tmp_path / "core"


def test_default_tokenizer_uses_artifacts_path(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / DEFAULT_FOLDER, max_seq_length=96)

    assert get_default_tokenizer(artifacts_path=tmp_path).get_max_tokens() == 96


def test_code_chunkers_use_artifacts_path_from_settings(
    tmp_path: Path, no_hub: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_tokenizer(tmp_path / DEFAULT_FOLDER, max_seq_length=96)
    monkeypatch.setattr(settings, "artifacts_path", tmp_path)

    assert code_utils._get_default_tokenizer().get_max_tokens() == 96


def test_max_tokens_falls_back_to_model_max_length(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / REPO_FOLDER, model_max_length=256)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    assert tokenizer.get_max_tokens() == 256


def test_local_tokenizer_without_declared_limit_needs_max_tokens(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / REPO_FOLDER)

    with pytest.raises(RuntimeError, match="max_tokens could not be determined"):
        HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path, max_tokens=32)
    assert tokenizer.get_max_tokens() == 32


def test_path_object_resolves_max_tokens_from_its_folder(tmp_path: Path, no_hub: None) -> None:
    _save_tokenizer(tmp_path / "my-tokenizer", max_seq_length=48)

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=tmp_path / "my-tokenizer")

    assert tokenizer.get_max_tokens() == 48


@pytest.mark.parametrize("empty_folder", [False, True])
def test_missing_tokenizer_warns_one_time_and_falls_back(
    tmp_path: Path, loaded_names: list[dict], caplog: pytest.LogCaptureFixture, empty_folder: bool
) -> None:
    if empty_folder:
        # An interrupted download can leave an empty folder behind.
        (tmp_path / REPO_FOLDER).mkdir()

    with caplog.at_level(logging.WARNING, logger=hf_mod.__name__):
        HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path, max_tokens=8)
        HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path, max_tokens=8)

    assert loaded_names == [{"pretrained_model_name_or_path": REPO_ID}] * 2
    warnings = [r for r in caplog.records if MISSING_WARNING in r.getMessage()]
    assert len(warnings) == 1
    assert "download_tokenizer" in warnings[0].getMessage()


def test_incomplete_tokenizer_folder_gives_a_clear_error(tmp_path: Path, no_hub: None) -> None:
    folder = tmp_path / REPO_FOLDER
    folder.mkdir()
    (folder / "special_tokens_map.json").write_text("{}")

    with pytest.raises(RuntimeError, match="could not be loaded") as exc_info:
        HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)

    assert str(folder) in str(exc_info.value)
    assert "download_tokenizer" in str(exc_info.value)


@pytest.mark.parametrize("hostile_name", ["../secret", "acme/../../secret", "secret/..", "..", "acme--tiny-tokenizer"])
def test_name_is_never_joined_outside_the_artifacts_path(
    tmp_path: Path, loaded_names: list[dict], hostile_name: str
) -> None:
    artifacts_path = tmp_path / "artifacts"
    # A tokenizer next to the artifacts directory, and one inside it under the
    # folder name of another repo: a naive join would load one of them.
    _save_tokenizer(tmp_path / "secret", max_seq_length=128)
    _save_tokenizer(artifacts_path / REPO_FOLDER, max_seq_length=128)

    HuggingFaceTokenizer.from_pretrained(model_name=hostile_name, artifacts_path=artifacts_path, max_tokens=8)

    assert loaded_names == [{"pretrained_model_name_or_path": hostile_name}]


def test_hostile_name_does_not_load_from_next_to_the_artifacts_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_path = tmp_path / "artifacts"
    artifacts_path.mkdir()
    _save_tokenizer(tmp_path / "secret", max_seq_length=128)
    # Run from a directory where `../secret` does not exist, so that only a
    # join with the artifacts path could find the tokenizer.
    workdir = tmp_path / "work" / "cwd"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    # No mock: transformers gets the name as given and cannot resolve it.
    with pytest.raises((OSError, ValueError)):
        HuggingFaceTokenizer.from_pretrained(model_name="../secret", artifacts_path=artifacts_path)


def test_download_then_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def _fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        _save_tokenizer(Path(kwargs["local_dir"]), max_seq_length=128)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(hf_mod, "snapshot_download", _fake_snapshot_download)

    folder = download_tokenizer(repo_id=REPO_ID, output_dir=str(tmp_path), revision="abc123")

    assert folder == tmp_path / REPO_FOLDER
    assert len(calls) == 1
    assert calls[0]["repo_id"] == REPO_ID
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["local_dir"] == tmp_path / REPO_FOLDER
    patterns = calls[0]["allow_patterns"]
    assert {"tokenizer.json", "tokenizer_config.json", "sentence_bert_config.json"} <= set(patterns)
    assert not any(p.endswith((".bin", ".safetensors", ".onnx")) or p == "*" for p in patterns)

    # What the download wrote is what `from_pretrained` looks for.
    monkeypatch.setattr(hf_mod, "snapshot_download", None)
    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=REPO_ID, artifacts_path=tmp_path)
    assert tokenizer.get_max_tokens() == 128


@pytest.mark.parametrize("hostile_name", ["../secret", "/etc", "a/b/c", "a--b", ""])
def test_download_rejects_an_invalid_repo_id(tmp_path: Path, no_hub: None, hostile_name: str) -> None:
    with pytest.raises(ValueError, match="Not a valid Hugging Face repo id"):
        download_tokenizer(repo_id=hostile_name, output_dir=tmp_path)
    assert not any(tmp_path.iterdir())
