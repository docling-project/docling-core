#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utils for document types."""

import unicodedata
from pathlib import Path


def relative_path(src: Path, target: Path) -> Path:
    """Compute the relative path from `src` to `target`.

    Args:
        src (str | Path): The source directory or file path (must be absolute).
        target (str | Path): The target directory or file path (must be absolute).

    Returns:
        Path: The relative path from `src` to `target`.

    Raises:
        ValueError: If either `src` or `target` is not an absolute path.
    """
    src = Path(src).resolve()
    target = Path(target).resolve()

    # Ensure both paths are absolute
    if not src.is_absolute():
        raise ValueError(f"The source path must be absolute: {src}")
    if not target.is_absolute():
        raise ValueError(f"The target path must be absolute: {target}")

    # Find the common ancestor
    common_parts = []
    for src_part, target_part in zip(src.parts, target.parts):
        if src_part == target_part:
            common_parts.append(src_part)
        else:
            break

    # Determine the path to go up from src to the common ancestor
    up_segments = [".."] * (len(src.parts) - len(common_parts))

    # Add the path from the common ancestor to the target
    down_segments = target.parts[len(common_parts) :]

    # Combine and return the result
    return Path(*up_segments, *down_segments)


def get_html_tag_with_text_direction(html_tag: str, text: str) -> str:
    """Form the HTML element with tag, text, and optional dir attribute."""
    text_dir = get_text_direction(text)

    if text_dir == "ltr":
        return f"<{html_tag}>{text}</{html_tag}>"
    else:
        return f'<{html_tag} dir="{text_dir}">{text}</{html_tag}>'


def get_text_direction(text: str) -> str:
    """Determine the text direction of a given string as LTR or RTL script."""
    if not text:
        return "ltr"  # Default for empty input

    rtl_scripts = {"R", "AL"}
    rtl_chars = sum(unicodedata.bidirectional(c) in rtl_scripts for c in text)

    return (
        "rtl"
        if unicodedata.bidirectional(text[0]) in rtl_scripts
        or rtl_chars > len(text) / 2
        else "ltr"
    )
