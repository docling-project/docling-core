# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

"""Regression tests for Markdown list continuation blocks."""

import pytest

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import Formatting
from docling_core.types.doc.labels import DocItemLabel, GroupLabel


@pytest.mark.parametrize("marker", ["-", "*", "1.", "10."])
@pytest.mark.parametrize("nested", [False, True])
def test_markdown_loose_list_blocks(marker, nested):
    doc = DoclingDocument(name="loose list")
    group = doc.add_list_group()
    if nested:
        outer = doc.add_list_item(text="Outer", parent=group)
        group = doc.add_list_group(parent=outer)
    doc.add_list_item(text="First", marker=marker, parent=group)
    item = doc.add_list_item(text="Set the key", marker=marker, parent=group)
    paragraph = doc.add_text(label=DocItemLabel.TEXT, text="Use this value", parent=item)
    code = doc.add_code(text="export KEY=value\nrun command", parent=item)
    doc.add_list_item(text="Last", marker=marker, parent=group)
    doc.add_text(label=DocItemLabel.TEXT, text="Outside")

    indent = " " * (len(marker) + 1)
    expected = (
        f"{marker} First\n"
        f"{marker} Set the key\n\n"
        f"{indent}Use this value\n\n"
        f"{indent}```\n"
        f"{indent}export KEY=value\n"
        f"{indent}run command\n"
        f"{indent}```\n\n"
        f"{marker} Last"
    )
    if nested:
        expected = "- Outer\n" + "\n".join("    " + line if line else "" for line in expected.split("\n"))
    expected += "\n\nOutside"
    result = MarkdownDocSerializer(doc=doc).serialize()
    assert result.text == expected
    assert result.text.count("Use this value") == 1
    assert result.text.count("export KEY=value") == 1
    assert {paragraph.self_ref, code.self_ref} <= {span.item.self_ref for span in result.spans}


def test_markdown_loose_list_last_item_no_trailing_blank_lines():
    doc = DoclingDocument(name="last item")
    item = doc.add_list_item(text="Item", parent=doc.add_list_group())
    doc.add_text(label=DocItemLabel.TEXT, text="Continuation", parent=item)
    assert doc.export_to_markdown() == "- Item\n\n  Continuation"


def test_markdown_loose_list_respects_label_filter():
    doc = DoclingDocument(name="filtered")
    item = doc.add_list_item(text="Item", parent=doc.add_list_group())
    doc.add_code(text="hidden", parent=item)
    assert doc.export_to_markdown(labels={DocItemLabel.LIST_ITEM}) == "- Item"


def test_markdown_list_inline_representation_unchanged():
    doc = DoclingDocument(name="inline")
    item = doc.add_list_item(text="", parent=doc.add_list_group())
    inline = doc.add_group(label=GroupLabel.INLINE, parent=item)
    doc.add_text(label=DocItemLabel.TEXT, text="Bold", formatting=Formatting(bold=True), parent=inline)
    doc.add_text(label=DocItemLabel.TEXT, text="and plain", parent=inline)
    assert doc.export_to_markdown() == "- **Bold** and plain"
