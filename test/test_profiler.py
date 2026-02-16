"""Tests for document profiler."""

import json
from pathlib import Path

import pytest

from docling_core.transforms.profiler import DocumentProfiler
from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem
from docling_core.types.doc.document import DocumentOrigin, PageItem, Size, TableData
from docling_core.types.doc.labels import DocItemLabel


def test_profile_empty_document():
    """Test profiling an empty document."""
    doc = DoclingDocument(name="Empty Document")

    stats = DocumentProfiler.profile_document(doc)

    assert stats.name == "Empty Document"
    assert stats.num_pages == 0
    assert stats.num_tables == 0
    assert stats.num_pictures == 0
    assert stats.num_texts == 0
    assert stats.num_key_value_items == 0
    assert stats.num_form_items == 0
    assert stats.total_items == 0
    assert stats.avg_items_per_page == 0.0
    assert stats.origin_mimetype is None


def test_profile_simple_document():
    """Test profiling a simple document with basic content."""
    doc = DoclingDocument(
        name="Simple Document",
        origin=DocumentOrigin(
            mimetype="application/pdf",
            binary_hash=12345,
            filename="test.pdf",
        ),
    )

    # Add some pages
    doc.pages[1] = PageItem(page_no=1, size=Size(width=612, height=792))
    doc.pages[2] = PageItem(page_no=2, size=Size(width=612, height=792))

    # Add some text items
    doc.add_text(label=DocItemLabel.TEXT, text="Text 1", orig="Text 1")
    doc.add_text(label=DocItemLabel.TEXT, text="Text 2", orig="Text 2")
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Section", orig="Section")

    # Add a table
    doc.add_table(data=TableData(num_rows=2, num_cols=2))

    # Add a picture
    doc.add_picture()

    stats = DocumentProfiler.profile_document(doc)

    assert stats.name == "Simple Document"
    assert stats.num_pages == 2
    assert stats.num_tables == 1
    assert stats.num_pictures == 1
    assert stats.num_texts == 3
    assert stats.num_section_headers == 1
    assert stats.total_items == 5
    assert stats.avg_items_per_page == 2.5
    assert stats.origin_mimetype == "application/pdf"


def test_profile_document_with_pictures_for_ocr():
    """Test profiling pictures that would trigger OCR based on area coverage."""
    doc = DoclingDocument(name="Document with Pictures for OCR")

    # Add a page
    doc.pages[1] = PageItem(page_no=1, size=Size(width=1000, height=1000))

    # Add a large picture (10% of page area, above default 5% threshold)
    doc.add_picture(
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=316.2, b=316.2),  # ~10% of page area
            charspan=(0, 0),
        )
    )

    # Add a small picture (2% of page area, below default 5% threshold)
    doc.add_picture(
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=141.4, b=141.4),  # ~2% of page area
            charspan=(0, 0),
        )
    )

    # Add a medium picture (exactly 5% of page area, at threshold)
    doc.add_picture(
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=223.607, b=223.607),  # exactly 5% of page area
            charspan=(0, 0),
        )
    )

    stats = DocumentProfiler.profile_document(doc)

    assert stats.num_pictures == 3
    # 2 out of 3 pictures meet the threshold (large and medium)
    assert stats.num_pictures_for_ocr == 2

    # Test with custom threshold of 10%
    stats_custom = DocumentProfiler.profile_document(doc, bitmap_coverage_threshold=0.10)
    # Only large picture (9.99%) is below 10%, so 0 pictures
    assert stats_custom.num_pictures_for_ocr == 0

    # Test with custom threshold of 2%
    stats_custom2 = DocumentProfiler.profile_document(doc, bitmap_coverage_threshold=0.02)
    # 2 pictures are above 2% threshold (large and medium, small is 1.99%)
    assert stats_custom2.num_pictures_for_ocr == 2


def test_profile_collection_empty():
    """Test profiling an empty collection."""
    stats = DocumentProfiler.profile_collection([])

    assert stats.num_documents == 0
    assert stats.total_pages == 0
    assert stats.total_tables == 0
    assert stats.total_pictures == 0
    assert stats.avg_items_per_document == 0.0
    assert stats.avg_items_per_page == 0.0
    assert stats.deciles_pages == [0.0] * 9
    assert stats.deciles_tables == [0.0] * 9
    assert stats.histogram_pages.bins == []
    assert stats.histogram_pages.frequencies == []
    assert stats.histogram_pages.bin_width == 0.0


def test_profile_collection_single_document():
    """Test profiling a collection with a single document."""
    doc = DoclingDocument(name="Single Doc")
    doc.pages[1] = PageItem(page_no=1, size=Size(width=612, height=792))
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Text", orig="Text")
    doc.add_table(data=TableData(num_rows=1, num_cols=1))
    doc.add_picture()

    stats = DocumentProfiler.profile_collection(doc)

    assert stats.num_documents == 1
    assert stats.total_pages == 1
    assert stats.total_tables == 1
    assert stats.total_pictures == 1
    assert stats.total_texts == 1
    assert stats.min_pages == 1
    assert stats.max_pages == 1
    assert stats.deciles_pages[4] == 1.0  # median is d5 (5th decile, index 4)
    assert stats.mean_pages == 1.0
    assert stats.std_pages == 0.0
    # Check histogram exists
    assert len(stats.histogram_pages.bins) > 0
    assert len(stats.histogram_pages.frequencies) > 0


def test_profile_collection_multiple_documents():
    """Test profiling a collection with multiple documents."""
    docs = []

    # Document 1: 2 pages, 1 table, 2 pictures, 2 texts
    doc1 = DoclingDocument(
        name="Doc1",
        origin=DocumentOrigin(mimetype="application/pdf", binary_hash=1, filename="doc1.pdf"),
    )
    doc1.pages[1] = PageItem(page_no=1, size=Size(width=612, height=792))
    doc1.pages[2] = PageItem(page_no=2, size=Size(width=612, height=792))
    doc1.add_table(data=TableData(num_rows=1, num_cols=1))
    doc1.add_picture()
    doc1.add_picture()
    doc1.add_text(label=DocItemLabel.TEXT, text="Text 1", orig="Text 1")
    doc1.add_text(label=DocItemLabel.TEXT, text="Text 2", orig="Text 2")
    docs.append(doc1)

    # Document 2: 5 pages, 3 tables, 1 picture, 10 texts
    doc2 = DoclingDocument(
        name="Doc2",
        origin=DocumentOrigin(mimetype="application/pdf", binary_hash=2, filename="doc2.pdf"),
    )
    for i in range(1, 6):
        doc2.pages[i] = PageItem(page_no=i, size=Size(width=612, height=792))
    for _ in range(3):
        doc2.add_table(data=TableData(num_rows=1, num_cols=1))
    doc2.add_picture()
    for i in range(10):
        doc2.add_text(label=DocItemLabel.TEXT, text=f"Text {i}", orig=f"Text {i}")
    docs.append(doc2)

    # Document 3: 1 page, 0 tables, 5 pictures, 2 texts
    doc3 = DoclingDocument(
        name="Doc3",
        origin=DocumentOrigin(mimetype="text/html", binary_hash=3, filename="doc3.html"),
    )
    doc3.pages[1] = PageItem(page_no=1, size=Size(width=612, height=792))
    for _ in range(5):
        doc3.add_picture()
    doc3.add_text(label=DocItemLabel.TEXT, text="T1", orig="T1")
    doc3.add_text(label=DocItemLabel.TEXT, text="T2", orig="T2")
    docs.append(doc3)

    stats = DocumentProfiler.profile_collection(docs, include_individual_stats=True)

    # Basic counts
    assert stats.num_documents == 3
    assert stats.total_pages == 8  # 2 + 5 + 1
    assert stats.total_tables == 4  # 1 + 3 + 0
    assert stats.total_pictures == 8  # 2 + 1 + 5
    assert stats.total_texts == 14  # 2 + 10 + 2

    # Page statistics
    assert stats.min_pages == 1
    assert stats.max_pages == 5
    assert stats.deciles_pages[4] == 2.0  # median is d5 (5th decile, index 4)
    assert stats.mean_pages == pytest.approx(8 / 3)
    assert stats.std_pages > 0
    # Check deciles are in order: [d1, d2, d3, d4, d5, d6, d7, d8, d9]
    assert stats.deciles_pages[0] <= stats.deciles_pages[4] <= stats.deciles_pages[8]
    # Check histogram exists
    assert len(stats.histogram_pages.bins) > 0
    assert len(stats.histogram_pages.frequencies) > 0

    # Table statistics
    assert stats.min_tables == 0
    assert stats.max_tables == 3
    assert stats.deciles_tables[4] == 1.0  # median is d5 (5th decile, index 4)
    assert stats.mean_tables == pytest.approx(4 / 3)
    # Check histogram exists
    assert len(stats.histogram_tables.bins) > 0

    # Picture statistics
    assert stats.min_pictures == 1
    assert stats.max_pictures == 5
    assert stats.deciles_pictures[4] == 2.0  # median is d5 (5th decile, index 4)
    assert stats.mean_pictures == pytest.approx(8 / 3)
    # Check histogram exists
    assert len(stats.histogram_pictures.bins) > 0

    # Text statistics
    assert stats.min_texts == 2
    assert stats.max_texts == 10
    assert stats.deciles_texts[4] == 2.0  # median is d5 (5th decile, index 4)
    assert stats.mean_texts == pytest.approx(14 / 3)
    # Check histogram exists
    assert len(stats.histogram_texts.bins) > 0

    # Document characteristics
    assert len(stats.document_stats) == 3

    # MIME type distribution
    assert stats.mimetype_distribution["application/pdf"] == 2
    assert stats.mimetype_distribution["text/html"] == 1

    # Computed fields
    assert stats.total_items == 26  # 14 texts + 4 tables + 8 pictures
    assert stats.avg_items_per_document == pytest.approx(26 / 3)
    assert stats.avg_items_per_page == pytest.approx(26 / 8)


def test_profile_collection_with_iterator():
    """Test profiling a collection using an iterator (generator)."""

    def doc_generator():
        for i in range(3):
            doc = DoclingDocument(name=f"Doc{i}")
            doc.pages[1] = PageItem(page_no=1, size=Size(width=612, height=792))
            doc.add_text(label=DocItemLabel.TEXT, text=f"Text {i}", orig=f"Text {i}")
            yield doc

    stats = DocumentProfiler.profile_collection(doc_generator())

    assert stats.num_documents == 3
    assert stats.total_pages == 3
    assert stats.total_texts == 3


def test_profile_collection_without_individual_stats():
    """Test that individual stats are not included by default."""
    docs = [DoclingDocument(name=f"Doc{i}") for i in range(3)]

    stats = DocumentProfiler.profile_collection(docs, include_individual_stats=False)

    assert len(stats.document_stats) == 0


def test_statistics_serialization():
    """Test that statistics can be serialized to JSON."""
    doc = DoclingDocument(name="Test Doc")
    doc.pages[1] = PageItem(page_no=1, size=Size(width=612, height=792))
    doc.add_text(label=DocItemLabel.TEXT, text="Text", orig="Text")

    doc_stats = DocumentProfiler.profile_document(doc)

    # Test DocumentStatistics serialization
    json_str = doc_stats.model_dump_json()
    data = json.loads(json_str)
    assert data["name"] == "Test Doc"
    assert data["num_pages"] == 1
    assert data["total_items"] == 1

    # Test CollectionStatistics serialization
    coll_stats = DocumentProfiler.profile_collection([doc])
    json_str = coll_stats.model_dump_json()
    data = json.loads(json_str)
    assert data["num_documents"] == 1
    assert data["total_pages"] == 1


def test_profile_real_document():
    """Test profiling a real document from test data."""
    test_file = Path("./test/data/doc/2408.09869v3_enriched.json")
    if not test_file.exists():
        pytest.skip("Test file not found")

    doc = DoclingDocument.load_from_json(test_file)
    stats = DocumentProfiler.profile_document(doc)

    # Basic sanity checks
    assert stats.name == doc.name
    assert stats.num_pages == len(doc.pages)
    assert stats.num_tables == len(doc.tables)
    assert stats.num_pictures == len(doc.pictures)
    assert stats.num_texts == len(doc.texts)
    assert stats.total_items > 0


def test_label_specific_counts():
    """Test that label-specific counts are accurate."""
    doc = DoclingDocument(name="Label Test")

    # Add various types of text items
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Section", orig="Section")
    doc.add_text(label=DocItemLabel.LIST_ITEM, text="Item 1", orig="Item 1")
    doc.add_text(label=DocItemLabel.LIST_ITEM, text="Item 2", orig="Item 2")
    doc.add_text(label=DocItemLabel.LIST_ITEM, text="Item 3", orig="Item 3")
    doc.add_text(label=DocItemLabel.CODE, text="code", orig="code")
    doc.add_text(label=DocItemLabel.FORMULA, text="x=y", orig="x=y")
    doc.add_text(label=DocItemLabel.TEXT, text="Text", orig="Text")

    stats = DocumentProfiler.profile_document(doc)

    assert stats.num_section_headers == 1
    assert stats.num_list_items == 3
    assert stats.num_code_items == 1
    assert stats.num_formulas == 1
    assert stats.num_texts == 7


def test_profile_sample_document(sample_doc):
    """Test profiling the sample document from conftest.py fixture."""
    stats = DocumentProfiler.profile_document(sample_doc)

    # Verify basic document properties
    assert stats.name == "Untitled 1"
    assert stats.num_pages == 0  # sample_doc doesn't add pages explicitly

    # Verify item counts based on the sample_doc construction
    assert stats.num_tables == len(sample_doc.tables)
    assert stats.num_pictures == len(sample_doc.pictures)
    assert stats.num_texts == len(sample_doc.texts)
    assert stats.num_key_value_items == len(sample_doc.key_value_items)
    assert stats.num_form_items == len(sample_doc.form_items)

    # Verify label-specific counts
    assert stats.num_section_headers > 0  # sample_doc has section headers
    assert stats.num_list_items > 0  # sample_doc has many list items
    assert stats.num_code_items > 0  # sample_doc has code items
    assert stats.num_formulas > 0  # sample_doc has formulas

    # Verify computed fields
    assert stats.total_items > 0
    assert stats.total_items == (
        stats.num_texts
        + stats.num_tables
        + stats.num_pictures
        + stats.num_key_value_items
        + stats.num_form_items
    )

    # sample_doc has no pages, so avg_items_per_page should be 0
    assert stats.avg_items_per_page == 0.0
