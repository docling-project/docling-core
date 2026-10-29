"""Page filtering must keep provenance aligned with the pages in the result."""

from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.items.key_value import GraphCell, GraphData
from docling_core.types.doc.labels import DocItemLabel, GraphCellLabel


def _document_with_cross_page_text() -> DoclingDocument:
    doc = DoclingDocument(name="cross-page-text")
    for page_no in (1, 2):
        doc.add_page(page_no, Size(width=100, height=100))

    item = doc.add_text(
        label=DocItemLabel.TEXT,
        text="continued",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=10, b=10),
            charspan=(0, 4),
        ),
    )
    item.prov.append(
        ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(l=0, t=20, r=10, b=30),
            charspan=(4, 9),
        )
    )
    return doc


def _assert_location_exports_work(doc: DoclingDocument) -> None:
    item = doc.texts[0]
    assert item.get_location_tokens(doc)
    assert item.text in doc.export_to_doctags()
    assert "<text" in doc.export_to_doclang()


def test_filter_drops_provenance_for_excluded_pages() -> None:
    source = _document_with_cross_page_text()

    filtered = source.filter({2})

    assert set(filtered.pages) == {2}
    assert [prov.page_no for prov in filtered.texts[0].prov] == [2]
    assert [prov.page_no for prov in source.texts[0].prov] == [1, 2]
    _assert_location_exports_work(filtered)


def test_repeated_filter_keeps_provenance_page_numbers_in_sync() -> None:
    filtered = _document_with_cross_page_text().filter({2}).filter({2})

    assert set(filtered.pages) == {1}
    assert [prov.page_no for prov in filtered.texts[0].prov] == [1]
    _assert_location_exports_work(filtered)


def test_filter_drops_excluded_graph_cell_provenance() -> None:
    doc = _document_with_cross_page_text()
    graph = GraphData(
        cells=[
            GraphCell(
                label=GraphCellLabel.KEY,
                cell_id=0,
                text="key on page 1",
                orig="key on page 1",
                prov=ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(l=0, t=0, r=10, b=10),
                    charspan=(0, 0),
                ),
            ),
            GraphCell(
                label=GraphCellLabel.VALUE,
                cell_id=1,
                text="value on page 2",
                orig="value on page 2",
                prov=ProvenanceItem(
                    page_no=2,
                    bbox=BoundingBox(l=0, t=20, r=10, b=30),
                    charspan=(0, 0),
                ),
            ),
        ]
    )
    doc.add_key_values(
        graph=graph,
        prov=ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(l=0, t=20, r=10, b=30),
            charspan=(0, 0),
        ),
    )

    filtered = doc.filter({2})

    cells = filtered.key_value_items[0].graph.cells
    assert cells[0].prov is None
    assert cells[1].prov is not None and cells[1].prov.page_no == 2
