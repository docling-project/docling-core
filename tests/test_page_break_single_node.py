"""Regression test for https://github.com/docling-project/docling-core/issues/714.

Single-node serialize() calls never reach serialize_doc(), where the
internal page-break sentinel was previously resolved. This leaked
`#_#_DOCLING_DOC_PAGE_BREAK_*` into single-node output and chunk text.
"""

from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
)

_BBOX = BoundingBox(l=50, t=700, r=500, b=680, coord_origin=CoordOrigin.BOTTOMLEFT)


def _prov(pn: int, text: str) -> ProvenanceItem:
    return ProvenanceItem(page_no=pn, bbox=_BBOX, charspan=(0, len(text)))


def _make_doc() -> tuple[DoclingDocument, object]:
    doc = DoclingDocument(name="d")
    for pn in (1, 2):
        doc.pages[pn] = PageItem(page_no=pn, size=Size(width=595, height=842))
    lst = doc.add_list_group(name="l")
    for text, pn in (("item one", 1), ("item two", 2)):
        doc.add_list_item(text=text, parent=lst, prov=_prov(pn, text))
    return doc, lst


def test_single_node_page_break_resolved():
    doc, lst = _make_doc()
    ser = MarkdownDocSerializer(doc=doc, params=MarkdownParams(page_break_placeholder="<!-- page break -->"))
    whole = ser.serialize().text
    single = ser.serialize(item=lst).text
    assert "#_#_DOCLING_DOC_PAGE_BREAK" not in whole
    assert "#_#_DOCLING_DOC_PAGE_BREAK" not in single
    assert "<!-- page break -->" in single
    assert single == whole
