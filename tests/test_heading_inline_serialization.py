"""Rich headings must keep their inline content before the section's body."""

import pytest

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
