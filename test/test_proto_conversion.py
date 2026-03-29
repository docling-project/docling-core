import pytest
from docling_core.types.doc import DoclingDocument, DocItemLabel
from docling_core.proto import docling_document_to_proto
from docling_core.proto.gen.ai.docling.core.v1 import docling_document_pb2 as pb2

def test_minimal_doc_conversion():
    doc = DoclingDocument(name="test_doc")
    proto = docling_document_to_proto(doc)
    
    assert proto.name == "test_doc"
    assert proto.body.name == "_root_"
    assert proto.furniture.name == "_root_"

def test_doc_with_text_conversion():
    doc = DoclingDocument(name="test_doc")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
    
    proto = docling_document_to_proto(doc)
    
    assert len(proto.texts) == 1
    assert proto.texts[0].text.base.text == "Hello world"
    assert proto.texts[0].text.base.label == pb2.DOC_ITEM_LABEL_PARAGRAPH

def test_doc_with_title_conversion():
    doc = DoclingDocument(name="test_doc")
    doc.add_title(text="Main Title")
    
    proto = docling_document_to_proto(doc)
    
    assert len(proto.texts) == 1
    assert proto.texts[0].title.base.text == "Main Title"
    assert proto.texts[0].title.base.label == pb2.DOC_ITEM_LABEL_TITLE
