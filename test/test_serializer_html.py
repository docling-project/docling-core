"""Examples of using the HTML Serializer for DoclingDocument."""

from test.test_docling_doc import _construct_doc

from docling_core.experimental.serializer.html import HTMLDocSerializer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument  # BoundingBox,


def test_html_export():

    doc = _construct_doc()

    # Create the serializer with default parameters
    serializer = HTMLDocSerializer(doc=doc)

    # Serialize the document
    html_output = serializer.serialize().text

    # Save to file
    with open("example_document.new.html", "w", encoding="utf-8") as f:
        f.write(html_output)

    doc.save_as_html(filename="example_document.old.html")
    doc.save_as_markdown(filename="example_document.old.md")

    print("Basic example saved to 'example_document.html'")

    assert True


def test_markdown_export_with_pageimages():

    doc = DoclingDocument.load_from_json(
        "/Users/taa/Documents/projects/docling/2501.12948v1.json"
    )

    doc.save_as_markdown(
        filename="2501.12948v1.markdown", image_mode=ImageRefMode.REFERENCED
    )


def test_html_export_with_pageimages():

    doc = DoclingDocument.load_from_json(
        "/Users/taa/Documents/projects/docling/2501.12948v1.json"
    )
    doc.save_as_html(filename="2501.12948v1.html", image_mode=ImageRefMode.EMBEDDED)
    doc.save_as_html(filename="2501.12948v1.split.html", image_mode=ImageRefMode.EMBEDDED, split_page_view=True)

    
    """


    """

    """
    # Create the serializer with default parameters
    serializer = HTMLDocSerializer(doc=doc)

    # Serialize the document
    html_output = serializer.serialize().text

    # Save to file
    with open("example_document.new.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    """

    assert True
