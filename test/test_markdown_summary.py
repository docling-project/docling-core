"""Tests for MarkdownSummarySerializer (document outline)."""

from pathlib import Path

import pytest

from docling_core.experimental.serializer.markdown_summary import (
    MarkdownSummaryMode,
    MarkdownSummaryParams,
    MarkdownSummarySerializer,
)

from .test_data_gen_flag import GEN_TEST_DATA
from .test_docling_doc import _construct_doc


def verify(exp_file: Path, actual: str):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{actual}\n")
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read().rstrip()
        assert expected == actual


@pytest.mark.parametrize(
    "mode",
    [
        MarkdownSummaryMode.OUTLINE,
        MarkdownSummaryMode.TABLE_OF_CONTENTS,
    ],
)
@pytest.mark.parametrize("use_md_headers", [False, True])
@pytest.mark.parametrize("indent_by_section_level", [False, True])
def test_markdown_summary_outline(
    mode: MarkdownSummaryMode, use_md_headers: bool, indent_by_section_level: bool
):
    # Build a representative document with title, headers, text, lists, table, and pictures
    doc = _construct_doc()

    ser = MarkdownSummarySerializer(
        doc=doc,
        params=MarkdownSummaryParams(
            use_markdown_headers=use_md_headers,
            mode=mode,
            indent_by_section_level=indent_by_section_level,
        ),
    )

    outline = ser.serialize().text

    # Compare with or generate ground-truth output
    root_dir = Path("./test/data/doc")
    exp_path = (
        root_dir
        / f"constructed_mdsum_{mode.value}_mdhdr_{str(use_md_headers).lower()}_indent_{str(indent_by_section_level).lower()}.gt.md"
    )
    verify(exp_file=exp_path, actual=outline)


@pytest.mark.parametrize("use_md_headers", [False, True])
def test_markdown_summary_indentation(use_md_headers: bool):
    # Build a representative document
    doc = _construct_doc()

    ser = MarkdownSummarySerializer(
        doc=doc,
        params=MarkdownSummaryParams(
            use_markdown_headers=use_md_headers,
            indent_by_section_level=True,
            indent_size=2,
        ),
    )

    outline = ser.serialize().text

    # Compare with or generate ground-truth output for indentation-specific case
    root_dir = Path("./test/data/doc")
    exp_path = (
        root_dir
        / f"constructed_mdsum_indent_mdhdr_{str(use_md_headers).lower()}_indent_true_size_2.gt.md"
    )
    verify(exp_file=exp_path, actual=outline)
