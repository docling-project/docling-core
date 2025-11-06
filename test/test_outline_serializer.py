from pathlib import Path

from docling_core.experimental.serializer.outline import OutlineDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.types.doc import DoclingDocument


def test_outline_serializer_basic():
    src = Path("test/data/doc/2408.09869_p1.json")
    doc = DoclingDocument.load_from_json(filename=src)
    
    print("MARKDOWN: \n\n")
    print(doc.export_to_markdown())
    
    # Only serialize metadata to focus on outline-like content
    params = MarkdownParams(include_non_meta=False)
    ser = OutlineDocSerializer(doc=doc, params=params)

    res = ser.serialize()
    actual = res.text

    print("SUMMARY: \n\n")
    print(actual)
    
    assert isinstance(actual, str)
    # Expect summaries from title and section header to appear
    assert "This is a title." in actual
    assert "This is a section header." in actual

