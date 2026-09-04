"""Test LaTeX serialization.

These tests leverage DOCLING_GEN_TEST_DATA via GEN_TEST_DATA to optionally
generate or update the golden files. Additionally, if a golden file does not
exist, the test will create it to bootstrap the baseline.
"""

from pathlib import Path

import yaml

from docling_core.transforms.serializer.latex import LaTeXDocSerializer, LaTeXParams
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument, Formatting

from .test_data_gen_flag import GEN_TEST_DATA


def verify_or_update(exp_file: Path, actual: str):
    exp_file.parent.mkdir(parents=True, exist_ok=True)
    # If GEN_TEST_DATA is enabled or the expected file is missing, write/update it
    if GEN_TEST_DATA or not exp_file.exists():
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"{actual}\n")
    else:
        with open(exp_file, encoding="utf-8") as f:
            expected = f.read().rstrip()
        assert expected == actual


def test_latex_basic_activities():
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            # Do not add page-break replacements by default
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_latex_inline_and_formatting():
    src = Path("./test/data/doc/inline_and_formatting.yaml")
    doc = DoclingDocument.load_from_yaml(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_dummy_doc():
    src = Path("test/data/doc/dummy_doc.yaml")

    # Read YAML file of manual reference doc
    with open(src, encoding="utf-8") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_constructed_doc(sample_doc: DoclingDocument):
    doc = sample_doc

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    src = Path("test/data/doc/construct_doc.yaml")
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_constructed_rich_table_doc(rich_table_doc: DoclingDocument):
    doc = rich_table_doc

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    src = Path("test/data/doc/construct_rich_table_doc.yaml")
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_latex_paper():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_latex_nested_lists():
    src = Path("./test/data/doc/polymers.json")
    doc = DoclingDocument.load_from_json(src)

    ser = LaTeXDocSerializer(
        doc=doc,
        params=LaTeXParams(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_command=None,
        ),
    )
    actual = ser.serialize().text
    verify_or_update(exp_file=src.with_suffix(".gt.tex"), actual=actual)


def test_inline_group_no_duplication():
    r"""Regression test: inline-group content must not appear twice.

    When a SectionHeaderItem, TitleItem or ListItem delegates its text to a
    child InlineGroup (the shape produced by Markdown/DOCX/HTML import for any
    formatted heading or list item), the serialized text must appear exactly
    once — inside the LaTeX command (\section{}, \item, …) and *not* again
    as a standalone paragraph immediately after it.
    """
    doc = DoclingDocument(name="t")

    # Formatted section heading via InlineGroup
    heading = doc.add_heading(text="", level=1)
    ig = doc.add_inline_group(parent=heading)
    doc.add_text(
        label="text",
        text="Partially formatted",
        formatting=Formatting(italic=True),
        parent=ig,
    )
    doc.add_text(label="text", text="heading", parent=ig)

    # Formatted list item via InlineGroup
    lg = doc.add_list_group()
    li = doc.add_list_item(text="", parent=lg)
    ig2 = doc.add_inline_group(parent=li)
    doc.add_text(label="text", text="Term", formatting=Formatting(bold=True), parent=ig2)
    doc.add_text(label="text", text=": definition", parent=ig2)

    body = LaTeXDocSerializer(doc=doc).serialize().text.split("\\begin{document}")[1]

    # Each piece of content must appear exactly once in the body
    assert body.count("\\textit{Partially formatted} heading") == 1
    assert body.count("\\textbf{Term} : definition") == 1

    # The content must be inside the LaTeX command, not on a standalone line
    assert "\\section{\\textit{Partially formatted} heading}" in body
    assert "\\item \\textbf{Term} : definition" in body
