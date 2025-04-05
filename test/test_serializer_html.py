"""Examples of using the HTML Serializer for DoclingDocument."""

from test.test_docling_doc import _construct_doc

from docling_core.experimental.serializer.html import HTMLDocSerializer


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
