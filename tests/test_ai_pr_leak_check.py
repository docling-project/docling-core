# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("ai_pr_leak_check", SCRIPTS_DIR / "ai_pr_leak_check.py")
assert SPEC is not None and SPEC.loader is not None
leak = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = leak
SPEC.loader.exec_module(leak)

SAMPLE_KEY = "bobk_Zx81QmN3-aTe7vUy_P0LrWc9Hd"


@pytest.mark.parametrize(
    "payload",
    [
        SAMPLE_KEY,
        SAMPLE_KEY[::-1],
        SAMPLE_KEY.encode().hex(),
        SAMPLE_KEY.encode().hex().upper(),
        # Base64 with padding
        "Ym9ia19aeDgxUW1OMy1hVGU3dlV5X1AwTHJXYzlIZA==",
        # URL-safe base64
        "Ym9ia19aeDgxUW1OMy1hVGU3dlV5X1AwTHJXYzlIZA",
        # Split by punctuation, space, and newline
        "bobk _ Zx81QmN3 - aTe7vUy _ P0LrWc9Hd",
        "bobk\nZx81QmN3\naTe7vUy\nP0LrWc9Hd",
        "bobk.Zx81QmN3.aTe7vUy.P0LrWc9Hd",
        # Inside a JSON field
        f'{{"summary": "the result", "key": "{SAMPLE_KEY}"}}',
    ],
)
def test_plain_and_encoded_copies_are_found(payload: str) -> None:
    assert leak.contains_secret(payload, SAMPLE_KEY)
    assert leak.contains_secret(f"preamble {payload} postamble", SAMPLE_KEY)


def test_normal_text_does_not_match() -> None:
    text = "The quick brown fox jumps over the lazy dog. 1234567890."
    assert not leak.contains_secret(text, SAMPLE_KEY)


def test_a_short_key_is_ignored() -> None:
    # A short value would produce too many false positives.
    assert not leak.contains_secret("short", "short")


def test_main_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file = tmp_path / "result.json"
    file.write_text(f'{{"key": "{SAMPLE_KEY}"}}', encoding="utf-8")
    monkeypatch.setenv("BOB_API_KEY", SAMPLE_KEY)
    assert leak.main(["--file", str(file)]) == 1

    clean_file = tmp_path / "clean.json"
    clean_file.write_text('{"summary": "all good"}', encoding="utf-8")
    assert leak.main(["--file", str(clean_file)]) == 0

    monkeypatch.delenv("BOB_API_KEY")
    assert leak.main(["--file", str(file)]) == 0
