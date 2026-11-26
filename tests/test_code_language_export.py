# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

"""Code block exports retain stored language labels without changing inline code."""

import pytest

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import CodeLanguageLabel, DocItemLabel, DoclingDocument


@pytest.mark.parametrize(
    ("language", "tag"),
    [
        (CodeLanguageLabel.PYTHON, "python"),
        (CodeLanguageLabel.C_PLUS_PLUS, "c++"),
        (CodeLanguageLabel.C_SHARP, "c#"),
        (CodeLanguageLabel.UNKNOWN, ""),
    ],
)
@pytest.mark.parametrize("output_format", ["markdown", "html"])
def test_code_block_exports_keep_language(language: CodeLanguageLabel, tag: str, output_format: str) -> None:
    doc = DoclingDocument(name="code-language")
    doc.add_code(text="value = '<tag>'", code_language=language)

    if output_format == "markdown":
        assert doc.export_to_markdown() == f"```{tag}\nvalue = '<tag>'\n```"
    else:
        code_class = f' class="language-{tag}"' if tag else ""
        assert f"<pre><code{code_class}>value = '&lt;tag&gt;'</code></pre>" in doc.export_to_html()


def test_language_label_keeps_long_fence_for_literal_backticks() -> None:
    doc = DoclingDocument(name="backticks")
    doc.add_code(text="literal = '```'", code_language=CodeLanguageLabel.PYTHON)

    assert doc.export_to_markdown() == "````python\nliteral = '```'\n````"


def test_language_label_does_not_turn_inline_code_into_a_block() -> None:
    doc = DoclingDocument(name="inline-code")
    group = doc.add_inline_group()
    doc.add_text(label=DocItemLabel.TEXT, text="Run ", parent=group)
    doc.add_code(text="print(1)", code_language=CodeLanguageLabel.PYTHON, parent=group)

    assert doc.export_to_markdown() == "Run `print(1)`"
    assert "<code>print(1)</code>" in doc.export_to_html()
    assert (
        MarkdownDocSerializer(doc=doc, params=MarkdownParams(format_code_blocks=False)).serialize().text
        == "Run print(1)"
    )
