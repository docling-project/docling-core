#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utils for document types."""

import html
import itertools
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from docling_core.types.doc.tokens import _LOC_PREFIX, DocumentToken, TableToken

if TYPE_CHECKING:
    from docling_core.types.doc.document import TableCell, TableData


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


def get_html_tag_with_text_direction(
    html_tag: str, text: str, attrs: Optional[dict] = None
) -> str:
    """Form the HTML element with tag, text, and optional dir attribute."""
    my_attrs = attrs or {}
    if (dir := my_attrs.get("dir")) is not None and dir != "ltr":
        my_attrs["dir"] = get_text_direction(text)
    pieces: list[str] = [html_tag]
    if my_attrs:
        attrs_str = " ".join(
            [
                f'{html.escape(k, quote=False)}="{html.escape(my_attrs[k], quote=False)}"'
                for k in my_attrs
            ]
        )
        pieces.append(attrs_str)
    return f"<{' '.join(pieces)}>{text}</{html_tag}>"


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


def otsl_extract_tokens_and_text(s: str) -> Tuple[List[str], List[str]]:
    """Extract OTSL tokens and text from an OTSL string."""
    # Pattern to match anything enclosed by < >
    # (including the angle brackets themselves)
    pattern = r"(<[^>]+>)"
    # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
    tokens = re.findall(pattern, s)
    # Remove any tokens that start with "<loc_"
    tokens = [
        token
        for token in tokens
        if not (
            token.startswith(rf"<{_LOC_PREFIX}")
            or token
            in [
                rf"<{DocumentToken.OTSL.value}>",
                rf"</{DocumentToken.OTSL.value}>",
            ]
        )
    ]
    # Split the string by those tokens to get the in-between text
    text_parts = re.split(pattern, s)
    text_parts = [
        token
        for token in text_parts
        if not (
            token.startswith(rf"<{_LOC_PREFIX}")
            or token
            in [
                rf"<{DocumentToken.OTSL.value}>",
                rf"</{DocumentToken.OTSL.value}>",
            ]
        )
    ]
    # Remove any empty or purely whitespace strings from text_parts
    text_parts = [part for part in text_parts if part.strip()]

    return tokens, text_parts


def otsl_parse_texts(
    texts: List[str], tokens: List[str]
) -> Tuple[List["TableCell"], List[List[str]]]:
    """Parse OTSL texts and tokens into table cells."""
    from docling_core.types.doc.document import TableCell

    split_word = TableToken.OTSL_NL.value
    # CLEAN tokens from extra tags, only structural OTSL allowed
    clean_tokens = []
    for t in tokens:
        if t in [
            TableToken.OTSL_ECEL.value,
            TableToken.OTSL_FCEL.value,
            TableToken.OTSL_LCEL.value,
            TableToken.OTSL_UCEL.value,
            TableToken.OTSL_XCEL.value,
            TableToken.OTSL_NL.value,
            TableToken.OTSL_CHED.value,
            TableToken.OTSL_RHED.value,
            TableToken.OTSL_SROW.value,
        ]:
            clean_tokens.append(t)
    tokens = clean_tokens
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]

    table_cells = []
    r_idx = 0
    c_idx = 0

    def count_right(
        tokens: List[List[str]], c_idx: int, r_idx: int, which_tokens: List[str]
    ) -> int:
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(
        tokens: List[List[str]], c_idx: int, r_idx: int, which_tokens: List[str]
    ) -> int:
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [
            TableToken.OTSL_FCEL.value,
            TableToken.OTSL_ECEL.value,
            TableToken.OTSL_CHED.value,
            TableToken.OTSL_RHED.value,
            TableToken.OTSL_SROW.value,
        ]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != TableToken.OTSL_ECEL.value:
                cell_text = texts[i + 1]
                right_offset = 2

            # Check next element(s) for lcel / ucel / xcel,
            # set properly row_span, col_span
            next_right_cell = ""
            if i + right_offset < len(texts):
                next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [
                TableToken.OTSL_LCEL.value,
                TableToken.OTSL_XCEL.value,
            ]:
                # we have horizontal spanning cell or 2d spanning cell
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [TableToken.OTSL_LCEL.value, TableToken.OTSL_XCEL.value],
                )
            if next_bottom_cell in [
                TableToken.OTSL_UCEL.value,
                TableToken.OTSL_XCEL.value,
            ]:
                # we have a vertical spanning cell or 2d spanning cell
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [TableToken.OTSL_UCEL.value, TableToken.OTSL_XCEL.value],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [
            TableToken.OTSL_FCEL.value,
            TableToken.OTSL_ECEL.value,
            TableToken.OTSL_CHED.value,
            TableToken.OTSL_RHED.value,
            TableToken.OTSL_SROW.value,
            TableToken.OTSL_LCEL.value,
            TableToken.OTSL_UCEL.value,
            TableToken.OTSL_XCEL.value,
        ]:
            c_idx += 1
        if text == TableToken.OTSL_NL.value:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def parse_otsl_table_content(otsl_content: str) -> "TableData":
    """Parse OTSL content into TableData."""
    from docling_core.types.doc.document import TableData

    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    return TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )
