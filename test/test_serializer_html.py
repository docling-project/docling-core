"""Examples of using the HTML Serializer for DoclingDocument."""

from pathlib import Path
from docling_core.types.doc.base import ImageRefMode, Size
from docling_core.types.doc.document import DoclingDocument, Formatting
from docling_core.types.doc.labels import DocItemLabel, CodeLanguageLabel
from docling_core.experimental.serializer.html import HTMLDocSerializer, HTMLParams

from test.test_docling_doc import _construct_doc

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
        
    print("Basic example saved to 'example_document.html'")
    
    assert True
