"""Examples of using the HTML Serializer for DoclingDocument."""

from test.test_docling_doc import _construct_doc

from docling_core.experimental.serializer.html import HTMLDocSerializer
from docling_core.types.doc.document import DoclingDocument  # BoundingBox,


def test_html_export():

    doc = _construct_doc()

    # Create the serializer with default parameters
    serializer = HTMLDocSerializer(doc=doc)

    # doc.save_as_html(filename="test/data/doc/constructed_doc.html")
    pred_html = doc.export_to_html()

    with open("test/data/doc/constructed_doc.html", "r") as fr:
        true_html = fr.read()

    assert pred_html == true_html, "pred_html==true_html"
