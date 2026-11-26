# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Check that a model answer does not contain the Bob API key.

A literal search finds only a plain copy of the key. A model that a prompt
injection steers can also write the key as base64, hex, reversed, or split by
spaces or punctuation. This check looks for those forms. It is a defense in
depth, not a guarantee: the key must also be limited in scope and cost.

The key is read from the environment, never from an argument, so it does not
show in a process list.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path

# A shorter value gives too many false matches to be useful.
MIN_KEY_LENGTH = 8
_SEPARATORS = re.compile(r"[\s\-_.,:;|/\\'\"`*+=]+")
_HEX = re.compile(r"[0-9a-fA-F]+")


def _base64_needles(data: bytes) -> set[str]:
    """Return the base64 text of ``data`` as it appears inside a longer text.

    Base64 encodes 3 bytes as 4 characters, so the text of ``data`` depends on
    its offset in the longer text. This encodes it at the three offsets and
    keeps only the characters that depend on ``data`` alone: the leading
    characters that mix in the bytes before it, and the last character of a
    partial group, are dropped.
    """
    needles: set[str] = set()
    for offset in range(3):
        encoded = base64.b64encode(bytes(offset) + data).decode("ascii").rstrip("=")
        # The first character that depends only on `data`: 0, 2, or 3.
        start = (0, 2, 3)[offset]
        end = len(encoded) - (1 if (offset + len(data)) % 3 else 0)
        needle = encoded[start:end]
        needles.add(needle)
        needles.add(needle.replace("+", "-").replace("/", "_"))
    return needles


def secret_needles(key: str) -> set[str]:
    """Return the plain, reversed, hex, and base64 forms of the key."""
    needles: set[str] = set()
    for value in (key, key[::-1]):
        data = value.encode("utf-8")
        needles |= {value, data.hex(), *_base64_needles(data)}
    return {needle for needle in needles if len(needle) >= MIN_KEY_LENGTH}


def contains_secret(text: str, key: str) -> bool:
    """True when the text holds the key in a plain, encoded, or split form."""
    if len(key) < MIN_KEY_LENGTH:
        return False
    needles = secret_needles(key)
    # The compact view removes whitespace and punctuation, so a copy that the
    # model split into parts or wrapped over several lines still matches.
    compact_text = _SEPARATORS.sub("", text)
    compact_needles = {_SEPARATORS.sub("", needle) for needle in needles}
    for haystack in (text, compact_text):
        if any(needle in haystack for needle in needles | compact_needles):
            return True
    # Hex is case-insensitive.
    hex_needles = {needle.lower() for needle in needles if _HEX.fullmatch(needle)}
    return any(needle in compact_text.lower() for needle in hex_needles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args(argv)

    key = os.environ.get("BOB_API_KEY", "")
    if not key or not args.file.is_file():
        return 0
    if contains_secret(args.file.read_text("utf-8", errors="replace"), key):
        print("::error::The answer contains the API key, plain or encoded. It is discarded.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
