"""Test the Markdown serializer functionality."""

import tempfile
from pathlib import Path

from docling_core.types.doc.document import BoundingBox, DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel


def create_test_document_with_pages():
    """Create a test document with page information."""
    doc = DoclingDocument(name="test_doc")

    # Add content to page 1
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="This is text on page 1.",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=10, t=10, r=110, b=30),
            charspan=(0, 23),
        ),
    )

    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="This is another text on page 1.",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=10, t=40, r=110, b=60),
            charspan=(0, 31),
        ),
    )

    # Add content to page 2
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="This is text on page 2.",
        prov=ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(l=10, t=10, r=110, b=30),
            charspan=(0, 23),
        ),
    )

    # Add content to page 3
    doc.add_heading(
        text="Heading on page 3",
        level=1,
        prov=ProvenanceItem(
            page_no=3,
            bbox=BoundingBox(l=10, t=10, r=110, b=30),
            charspan=(0, 17),
        ),
    )

    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="This is text on page 3.",
        prov=ProvenanceItem(
            page_no=3,
            bbox=BoundingBox(l=10, t=40, r=110, b=60),
            charspan=(0, 23),
        ),
    )

    return doc


def test_export_to_markdown_with_page_markers():
    """Test exporting to markdown with page markers."""
    doc = create_test_document_with_pages()

    # Export without page markers
    md_without_markers = doc.export_to_markdown(add_page_markers=False)

    # Export with page markers
    md_with_markers = doc.export_to_markdown(add_page_markers=True)

    # Check that the page markers are included
    assert "##PAGE 1##" in md_with_markers
    assert "##PAGE 2##" in md_with_markers
    assert "##PAGE 3##" in md_with_markers

    # Check that the content is included
    assert "This is text on page 1." in md_with_markers
    assert "This is another text on page 1." in md_with_markers
    assert "This is text on page 2." in md_with_markers
    assert "Heading on page 3" in md_with_markers
    assert "This is text on page 3." in md_with_markers

    # Check that the page markers are not included in the version without markers
    assert "##PAGE 1##" not in md_without_markers
    assert "##PAGE 2##" not in md_without_markers
    assert "##PAGE 3##" not in md_without_markers


def test_save_as_markdown_with_page_markers():
    """Test saving to markdown file with page markers."""
    doc = create_test_document_with_pages()

    # Create temporary files
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False
    ) as temp_file_with_markers:
        file_path_with_markers = Path(temp_file_with_markers.name)

    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False
    ) as temp_file_without_markers:
        file_path_without_markers = Path(temp_file_without_markers.name)

    try:
        # Test that export_to_markdown works
        md_with_markers = doc.export_to_markdown(add_page_markers=True)
        assert "##PAGE 1##" in md_with_markers

        # Save with page markers
        doc.save_as_markdown(
            filename=file_path_with_markers,
            add_page_markers=True,
        )

        # Save without page markers
        doc.save_as_markdown(
            filename=file_path_without_markers,
            add_page_markers=False,
        )

        # Read the files
        with open(file_path_with_markers, "r", encoding="utf-8") as f:
            content_with_markers = f.read()
            print(f"Content with markers: {content_with_markers}")

        with open(file_path_without_markers, "r", encoding="utf-8") as f:
            content_without_markers = f.read()
            print(f"Content without markers: {content_without_markers}")

        # Check file sizes
        print(f"File with markers size: {file_path_with_markers.stat().st_size}")
        print(f"File without markers size: {file_path_without_markers.stat().st_size}")

        # Check that the page markers are included
        assert "##PAGE 1##" in content_with_markers
        assert "##PAGE 2##" in content_with_markers
        assert "##PAGE 3##" in content_with_markers

        # Check that the content is included
        assert "This is text on page 1." in content_with_markers
        assert "This is another text on page 1." in content_with_markers
        assert "This is text on page 2." in content_with_markers
        assert "Heading on page 3" in content_with_markers
        assert "This is text on page 3." in content_with_markers

        # Check that the page markers are not included in the version without markers
        assert "##PAGE 1##" not in content_without_markers
        assert "##PAGE 2##" not in content_without_markers
        assert "##PAGE 3##" not in content_without_markers
    finally:
        # Clean up
        if file_path_with_markers.exists():
            file_path_with_markers.unlink()
        if file_path_without_markers.exists():
            file_path_without_markers.unlink()


def test_export_to_markdown_with_page_markers_and_page_filter():
    """Test exporting to markdown with page markers and page filter."""
    doc = create_test_document_with_pages()

    # Export with page markers and filter for page 2
    md_page_2 = doc.export_to_markdown(add_page_markers=True, page_no=2)

    # Check that only page 2 marker is included
    assert "##PAGE 1##" not in md_page_2
    assert "##PAGE 2##" in md_page_2
    assert "##PAGE 3##" not in md_page_2

    # Check that only page 2 content is included
    assert "This is text on page 1." not in md_page_2
    assert "This is another text on page 1." not in md_page_2
    assert "This is text on page 2." in md_page_2
    assert "Heading on page 3" not in md_page_2
    assert "This is text on page 3." not in md_page_2
