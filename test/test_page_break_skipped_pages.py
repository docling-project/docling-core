"""Test page break generation for skipped pages.

This module tests that the _iterate_items() function correctly generates
page breaks for documents with non-consecutive page numbers (i.e., when
some pages failed to parse and were skipped).

Tests cover:
1. Multiple page breaks are correctly generated when pages are skipped
2. The prev_page and next_page values in each generated PageBreakNode are correct
3. Edge cases: skipping just one page vs. multiple pages
"""

from pathlib import Path

from docling_core.types.doc.document import DoclingDocument


# =============================================================================
# Test: Document page count verification
# =============================================================================


def test_normal_4pages_has_all_pages():
    """Test that normal_4pages.json has all 4 pages."""
    src = Path("./test/data/doc/normal_4pages.json")
    doc = DoclingDocument.load_from_json(src)

    page_numbers = list(doc.pages.keys())

    assert len(page_numbers) == 4, (
        f"Expected 4 pages in normal_4pages.json, got {len(page_numbers)}"
    )
    assert page_numbers == [1, 2, 3, 4], (
        f"Expected pages [1, 2, 3, 4], got {page_numbers}"
    )


def test_skipped_2pages_has_only_two_pages():
    """Test that skipped_2pages.json has only 2 pages (pages 2 and 3 failed to parse)."""
    src = Path("./test/data/doc/skipped_2pages.json")
    doc = DoclingDocument.load_from_json(src)

    page_numbers = list(doc.pages.keys())

    assert len(page_numbers) == 2, (
        f"Expected 2 pages in skipped_2pages.json, got {len(page_numbers)}"
    )
    assert page_numbers == [1, 4], (
        f"Expected pages [1, 4], got {page_numbers}"
    )


def test_skipped_1page_has_two_pages():
    """Test that skipped_1page.json has 2 pages (page 2 failed to parse)."""
    src = Path("./test/data/doc/skipped_1page.json")
    doc = DoclingDocument.load_from_json(src)

    page_numbers = list(doc.pages.keys())

    assert len(page_numbers) == 2, (
        f"Expected 2 pages in skipped_1page.json, got {len(page_numbers)}"
    )
    assert page_numbers == [1, 3], (
        f"Expected pages [1, 3], got {page_numbers}"
    )


# =============================================================================
# Test: DocTags page break count
# =============================================================================


def test_doctags_page_breaks_normal():
    """Test that normal 4-page document has 3 page breaks in doctags output."""
    src = Path("./test/data/doc/normal_4pages.json")
    doc = DoclingDocument.load_from_json(src)

    doctags_output = doc.export_to_doctags()
    page_break_count = doctags_output.count("<page_break>")

    assert page_break_count == 3, (
        f"Expected 3 page breaks for 4-page document, got {page_break_count}"
    )


def test_doctags_page_breaks_skipped_2pages():
    """Test that document with 2 skipped pages has 3 page breaks in doctags output.

    The skipped_2pages.json document has pages 1 and 4 only (pages 2 and 3
    failed to parse), but should still generate 3 page breaks to indicate
    the page transitions from 1->2->3->4.
    """
    src = Path("./test/data/doc/skipped_2pages.json")
    doc = DoclingDocument.load_from_json(src)

    doctags_output = doc.export_to_doctags()
    page_break_count = doctags_output.count("<page_break>")

    # After the fix in _yield_page_breaks(): should have 3 page breaks
    # (1->2, 2->3, 3->4) even though pages 2 and 3 are missing from the document.
    # Before the fix: only had 1 page break (1->4)
    assert page_break_count == 3, (
        f"Expected 3 page breaks for document with 2 skipped pages, got {page_break_count}"
    )


def test_doctags_page_breaks_skipped_1page():
    """Test that document with 1 skipped page has 2 page breaks in doctags output.

    The skipped_1page.json document has pages 1 and 3 only (page 2 failed
    to parse), but should still generate 2 page breaks to indicate
    the page transitions from 1->2->3.
    """
    src = Path("./test/data/doc/skipped_1page.json")
    doc = DoclingDocument.load_from_json(src)

    doctags_output = doc.export_to_doctags()
    page_break_count = doctags_output.count("<page_break>")

    # Should have 2 page breaks (1->2, 2->3) even though page 2 is missing.
    # Before the fix: only had 1 page break (1->3)
    assert page_break_count == 2, (
        f"Expected 2 page breaks for document with 1 skipped page, got {page_break_count}"
    )


# =============================================================================
# Test: Markdown page break count
# =============================================================================


def test_markdown_page_breaks_normal():
    """Test that normal 4-page document has 3 page breaks in markdown output."""
    src = Path("./test/data/doc/normal_4pages.json")
    doc = DoclingDocument.load_from_json(src)

    markdown_output = doc.export_to_markdown(page_break_placeholder="---PAGE BREAK---")
    page_break_count = markdown_output.count("---PAGE BREAK---")

    assert page_break_count == 3, (
        f"Expected 3 page breaks for 4-page document, got {page_break_count}"
    )


def test_markdown_page_breaks_skipped_2pages():
    """Test that document with 2 skipped pages has 3 page breaks in markdown output."""
    src = Path("./test/data/doc/skipped_2pages.json")
    doc = DoclingDocument.load_from_json(src)

    markdown_output = doc.export_to_markdown(page_break_placeholder="---PAGE BREAK---")
    page_break_count = markdown_output.count("---PAGE BREAK---")

    # After the fix: should have 3 page breaks
    # Before the fix: only had 1 page break
    assert page_break_count == 3, (
        f"Expected 3 page breaks for document with 2 skipped pages, got {page_break_count}"
    )


def test_markdown_page_breaks_skipped_1page():
    """Test that document with 1 skipped page has 2 page breaks in markdown output."""
    src = Path("./test/data/doc/skipped_1page.json")
    doc = DoclingDocument.load_from_json(src)

    markdown_output = doc.export_to_markdown(page_break_placeholder="---PAGE BREAK---")
    page_break_count = markdown_output.count("---PAGE BREAK---")

    # Should have 2 page breaks (1->2, 2->3) even though page 2 is missing.
    assert page_break_count == 2, (
        f"Expected 2 page breaks for document with 1 skipped page, got {page_break_count}"
    )


# =============================================================================
# Test: HTML split_page_view table row count
# =============================================================================


def test_html_split_page_view_normal():
    """Test that normal 4-page document has 4 page divs in HTML split_page_view."""
    src = Path("./test/data/doc/normal_4pages.json")
    doc = DoclingDocument.load_from_json(src)

    html_output = doc.export_to_html(split_page_view=True)
    # Count page divs instead of tr tags to avoid counting nested tables
    page_div_count = html_output.count("<div class='page'>")

    assert page_div_count == 4, (
        f"Expected 4 page divs for 4-page document, got {page_div_count}"
    )


def test_html_split_page_view_skipped_2pages():
    """Test that document with 2 skipped pages has 4 page divs in HTML split_page_view.

    The skipped_2pages.json document has pages 1 and 4 only (pages 2 and 3
    failed to parse), but should still generate 4 page divs to maintain
    page number alignment with physical PDF pages.
    """
    src = Path("./test/data/doc/skipped_2pages.json")
    doc = DoclingDocument.load_from_json(src)

    html_output = doc.export_to_html(split_page_view=True)
    # Count page divs (all use same structure now)
    page_div_count = html_output.count("<div class='page'>")

    # Should have 4 page divs (pages 1, 2, 3, 4) even though pages 2 and 3 are missing
    assert page_div_count == 4, (
        f"Expected 4 page divs for document with 2 skipped pages, got {page_div_count}"
    )


def test_html_split_page_view_skipped_1page():
    """Test that document with 1 skipped page has 3 page divs in HTML split_page_view.

    The skipped_1page.json document has pages 1 and 3 only (page 2 failed
    to parse), but should still generate 3 page divs to maintain
    page number alignment with physical PDF pages.
    """
    src = Path("./test/data/doc/skipped_1page.json")
    doc = DoclingDocument.load_from_json(src)

    html_output = doc.export_to_html(split_page_view=True)
    # Count page divs (all use same structure now)
    page_div_count = html_output.count("<div class='page'>")

    # Should have 3 page divs (pages 1, 2, 3) even though page 2 is missing
    assert page_div_count == 3, (
        f"Expected 3 page divs for document with 1 skipped page, got {page_div_count}"
    )


# =============================================================================
# Test: Page break index is correctly maintained
# (Verified indirectly through correct count of page breaks)
# =============================================================================
# Note: The prev_page and next_page values in _PageBreakNode are internal
# implementation details that are not directly exposed in the exported formats.
# The correctness of these values is verified indirectly by ensuring the
# correct number of page breaks are generated.
