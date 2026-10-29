# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Rich headings must keep their inline content before the section's body."""

import json
import re

import pytest

from docling_core.experimental.serializer.outline import OutlineDocSerializer, OutlineFormat, OutlineMode, OutlineParams
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk, HierarchicalChunker
from docling_core.transforms.serializer.plain_text import PlainTextDocSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument


@pytest.mark.parametrize("level", [None, 1, 3])
@pytest.mark.parametrize("following", ["none", "paragraph", "subsection"])
def test_rich_heading_keeps_its_inline_content(level: int | None, following: str) -> None:
    doc = DoclingDocument(name="rich heading")
    heading = doc.add_title(text="") if level is None else doc.add_heading(text="", level=level)
    inline = doc.add_inline_group(parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="Heading eq", parent=inline)
    doc.add_text(label=DocItemLabel.FORMULA, text="y=2", parent=inline)
    if following == "paragraph":
        doc.add_text(label=DocItemLabel.TEXT, text="Body under it.", parent=heading)
    elif following == "subsection":
        subsection = doc.add_heading(text="Subsection", level=4, parent=heading)
        doc.add_text(label=DocItemLabel.TEXT, text="Body under it.", parent=subsection)

    doc = DoclingDocument.model_validate(doc.export_to_dict())
    markdown = doc.export_to_markdown()
    depth = 1 if level is None else level + 1
    assert markdown.splitlines()[0] == f"{'#' * depth} Heading eq $y=2$"
    assert markdown.count("Heading eq") == markdown.count("$y=2$") == 1
    html = doc.export_to_html(formula_to_mathml=False)
    heading_html = html.split(f"<h{depth}>", 1)[1].split(f"</h{depth}>", 1)[0]
    assert "Heading eq" in heading_html and "y=2" in heading_html
    assert "Body under it." not in heading_html
    assert html.count("Heading eq") == html.count("y=2") == 1
    if following != "none":
        assert markdown.count("Body under it.") == html.count("Body under it.") == 1
        assert markdown.index("Body under it.") > markdown.index("$y=2$")
        assert html.index("Body under it.") > html.index(f"</h{depth}>")
    if following == "subsection":
        assert "##### Subsection" in markdown
        assert html.index("<h5>Subsection</h5>") > html.index(f"</h{depth}>")
    plain = PlainTextDocSerializer(doc=doc).serialize().text
    assert plain.splitlines()[0] == "Heading eq $y=2$"
    assert plain.count("Heading eq") == plain.count("y=2") == 1
    tokens = doc.export_to_doctags()
    token_heading = re.search(r"<(title|section_header_level_\d+)>(.*?)</\1>", tokens, re.DOTALL)
    assert token_heading is not None
    assert "Heading eq" in token_heading[2] and "y=2" in token_heading[2]
    assert "Body under it." not in token_heading[2]
    assert tokens.count("Heading eq") == tokens.count("y=2") == 1


@pytest.mark.parametrize("level", [None, 1])
def test_inline_content_after_a_body_item_is_not_a_heading(level: int | None) -> None:
    doc = DoclingDocument(name="body inline content")
    heading = doc.add_title(text="") if level is None else doc.add_heading(text="", level=level)
    doc.add_text(label=DocItemLabel.TEXT, text="First body.", parent=heading)
    inline = doc.add_inline_group(parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="Later body.", parent=inline)

    depth = 1 if level is None else level + 1
    markdown = doc.export_to_markdown()
    assert markdown.splitlines()[0].strip() == "#" * depth
    assert markdown.index("First body.") < markdown.index("Later body.")
    assert f"<h{depth}></h{depth}>" in doc.export_to_html()


@pytest.mark.parametrize("level", [None, 1, 3])
@pytest.mark.parametrize("following", ["none", "paragraph", "subsection"])
def test_rich_heading_chunk_context(level: int | None, following: str) -> None:
    doc = DoclingDocument(name="rich heading chunks")
    heading = doc.add_title(text="") if level is None else doc.add_heading(text="", level=level)
    inline = doc.add_inline_group(parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="Heading eq", parent=inline)
    doc.add_text(label=DocItemLabel.FORMULA, text="y=2", parent=inline)
    parent = heading
    if following == "subsection":
        parent = doc.add_heading(text="Subsection", level=4, parent=heading)
    if following != "none":
        doc.add_text(label=DocItemLabel.TEXT, text="Body under it.", parent=parent)

    chunks = [DocChunk.model_validate(chunk) for chunk in HierarchicalChunker(always_emit_headings=True).chunk(doc)]
    assert len(chunks) == 1
    assert chunks[0].text == ("" if following == "none" else "Body under it.")
    assert chunks[0].meta.headings == ["Heading eq $y=2$"] + (["Subsection"] if following == "subsection" else [])


@pytest.mark.parametrize("level", [None, 1, 3])
@pytest.mark.parametrize("format", list(OutlineFormat))
def test_rich_heading_table_of_contents(level: int | None, format: OutlineFormat) -> None:
    doc = DoclingDocument(name="rich heading outline")
    heading = doc.add_title(text="") if level is None else doc.add_heading(text="", level=level)
    inline = doc.add_inline_group(parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="Heading eq", parent=inline)
    doc.add_text(label=DocItemLabel.FORMULA, text="y=2", parent=inline)
    doc.add_text(label=DocItemLabel.TEXT, text="Body under it.", parent=heading)
    outline = (
        OutlineDocSerializer(
            doc=doc, params=OutlineParams(mode=OutlineMode.TABLE_OF_CONTENTS, format=format, include_non_meta=True)
        )
        .serialize()
        .text
    )
    assert outline.count("Heading eq") == outline.count("y=2") == 1
    assert "Body under it." not in outline
    if format == OutlineFormat.JSON:
        data = json.loads(outline)
        assert len(data) == 1
        assert data[0]["ref"] == heading.self_ref
        assert data[0]["title"] == "Heading eq $y=2$"
