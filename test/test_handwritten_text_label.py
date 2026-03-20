"""Test HANDWRITTEN_TEXT label integration.

This module tests that the HANDWRITTEN_TEXT label is properly integrated
across all required components:
1. DocTags serialization (tokens.py mapping)
2. Export labels (document.py DEFAULT_EXPORT_LABELS)
3. Markdown export
4. Token mapping completeness
"""

import pytest

from docling_core.types.doc.document import (
    DEFAULT_EXPORT_LABELS,
    DOCUMENT_TOKENS_EXPORT_LABELS,
    DoclingDocument,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.tokens import DocumentToken


class TestHandwrittenTextLabelIntegration:
    """Test suite for HANDWRITTEN_TEXT label integration."""

    def test_label_exists_in_enum(self):
        """Test that HANDWRITTEN_TEXT exists in DocItemLabel enum."""
        assert hasattr(DocItemLabel, "HANDWRITTEN_TEXT")
        assert DocItemLabel.HANDWRITTEN_TEXT.value == "handwritten_text"

    def test_label_has_color(self):
        """Test that HANDWRITTEN_TEXT has a color mapping."""
        color = DocItemLabel.get_color(DocItemLabel.HANDWRITTEN_TEXT)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_document_token_exists(self):
        """Test that HANDWRITTEN_TEXT exists in DocumentToken enum."""
        assert hasattr(DocumentToken, "HANDWRITTEN_TEXT")
        assert DocumentToken.HANDWRITTEN_TEXT.value == "handwritten_text"

    def test_token_mapping_exists(self):
        """Test that HANDWRITTEN_TEXT has a token mapping in create_token_name_from_doc_item_label.

        Without this mapping, RuntimeError would be raised during DocTags serialization.
        """
        # This should not raise RuntimeError
        token_name = DocumentToken.create_token_name_from_doc_item_label(
            DocItemLabel.HANDWRITTEN_TEXT.value
        )
        assert token_name == "handwritten_text"

    def test_label_in_default_export_labels(self):
        """Test that HANDWRITTEN_TEXT is in DEFAULT_EXPORT_LABELS.

        Without this, HANDWRITTEN_TEXT content would be silently omitted from exports.
        """
        assert DocItemLabel.HANDWRITTEN_TEXT in DEFAULT_EXPORT_LABELS

    def test_label_in_document_tokens_export_labels(self):
        """Test that HANDWRITTEN_TEXT is in DOCUMENT_TOKENS_EXPORT_LABELS.

        This set is used by markdown and other serializers.
        """
        assert DocItemLabel.HANDWRITTEN_TEXT in DOCUMENT_TOKENS_EXPORT_LABELS


class TestHandwrittenTextDocTagsSerialization:
    """Test DocTags serialization with HANDWRITTEN_TEXT label."""

    def test_doctags_serialization_succeeds(self):
        """Test that DocTags serialization works with HANDWRITTEN_TEXT.

        This would raise RuntimeError if token mapping is missing.
        """
        doc = DoclingDocument(name="test_doctags")
        doc.add_text(
            label=DocItemLabel.HANDWRITTEN_TEXT,
            text="This is handwritten text",
        )

        # This should not raise RuntimeError
        doctags_output = doc.export_to_doctags()

        assert "<handwritten_text>" in doctags_output
        assert "</handwritten_text>" in doctags_output
        assert "This is handwritten text" in doctags_output

    def test_doctags_with_multiple_handwritten_items(self):
        """Test DocTags serialization with multiple HANDWRITTEN_TEXT items."""
        doc = DoclingDocument(name="test_multiple")
        doc.add_text(label=DocItemLabel.HANDWRITTEN_TEXT, text="First handwritten")
        doc.add_text(label=DocItemLabel.HANDWRITTEN_TEXT, text="Second handwritten")
        doc.add_text(label=DocItemLabel.TEXT, text="Regular text")

        doctags_output = doc.export_to_doctags()

        assert doctags_output.count("<handwritten_text>") == 2
        assert doctags_output.count("</handwritten_text>") == 2
        assert "First handwritten" in doctags_output
        assert "Second handwritten" in doctags_output


class TestHandwrittenTextMarkdownExport:
    """Test Markdown export with HANDWRITTEN_TEXT label."""

    def test_markdown_export_includes_handwritten_text(self):
        """Test that HANDWRITTEN_TEXT content appears in markdown export.

        Without DEFAULT_EXPORT_LABELS inclusion, this content would be silently omitted.
        """
        doc = DoclingDocument(name="test_markdown")
        doc.add_text(label=DocItemLabel.TEXT, text="Regular text.")
        doc.add_text(
            label=DocItemLabel.HANDWRITTEN_TEXT,
            text="Handwritten content here.",
        )

        markdown = doc.export_to_markdown()

        assert "Regular text." in markdown
        assert "Handwritten content here." in markdown

    def test_markdown_export_preserves_order(self):
        """Test that HANDWRITTEN_TEXT items maintain their order in markdown export."""
        doc = DoclingDocument(name="test_order")
        doc.add_text(label=DocItemLabel.TEXT, text="First")
        doc.add_text(label=DocItemLabel.HANDWRITTEN_TEXT, text="Second (handwritten)")
        doc.add_text(label=DocItemLabel.TEXT, text="Third")

        markdown = doc.export_to_markdown()

        first_pos = markdown.find("First")
        second_pos = markdown.find("Second (handwritten)")
        third_pos = markdown.find("Third")

        assert first_pos < second_pos < third_pos


class TestHandwrittenTextPlainTextExport:
    """Test plain text export with HANDWRITTEN_TEXT label."""

    def test_plain_text_export_includes_handwritten_text(self):
        """Test that HANDWRITTEN_TEXT content appears in plain text export."""
        doc = DoclingDocument(name="test_plain")
        doc.add_text(label=DocItemLabel.TEXT, text="Normal text.")
        doc.add_text(
            label=DocItemLabel.HANDWRITTEN_TEXT,
            text="Handwritten note.",
        )

        plain_text = doc.export_to_text()

        assert "Normal text." in plain_text
        assert "Handwritten note." in plain_text


class TestHandwrittenTextHtmlExport:
    """Test HTML export with HANDWRITTEN_TEXT label."""

    def test_html_export_includes_handwritten_text(self):
        """Test that HANDWRITTEN_TEXT content appears in HTML export."""
        doc = DoclingDocument(name="test_html")
        doc.add_text(label=DocItemLabel.TEXT, text="Regular paragraph.")
        doc.add_text(
            label=DocItemLabel.HANDWRITTEN_TEXT,
            text="Handwritten section.",
        )

        html = doc.export_to_html()

        assert "Regular paragraph." in html
        assert "Handwritten section." in html


class TestHandwrittenTextDocumentOperations:
    """Test document operations with HANDWRITTEN_TEXT label."""

    def test_add_text_with_handwritten_label(self):
        """Test adding text with HANDWRITTEN_TEXT label."""
        doc = DoclingDocument(name="test_add")
        item = doc.add_text(
            label=DocItemLabel.HANDWRITTEN_TEXT,
            text="My handwritten note",
        )

        assert item.label == DocItemLabel.HANDWRITTEN_TEXT
        assert item.text == "My handwritten note"

    def test_iterate_items_includes_handwritten_text(self):
        """Test that iterate_items includes HANDWRITTEN_TEXT items."""
        doc = DoclingDocument(name="test_iterate")
        doc.add_text(label=DocItemLabel.TEXT, text="Normal")
        doc.add_text(label=DocItemLabel.HANDWRITTEN_TEXT, text="Handwritten")

        labels = [
            item.label
            for item, _ in doc.iterate_items()
            if hasattr(item, "label")
        ]

        assert DocItemLabel.TEXT in labels
        assert DocItemLabel.HANDWRITTEN_TEXT in labels

    def test_json_roundtrip_preserves_handwritten_label(self, tmp_path):
        """Test that JSON save/load preserves HANDWRITTEN_TEXT label."""
        doc = DoclingDocument(name="test_roundtrip")
        doc.add_text(
            label=DocItemLabel.HANDWRITTEN_TEXT,
            text="Preserved handwritten text",
        )

        json_path = tmp_path / "test.json"
        doc.save_as_json(json_path)

        loaded_doc = DoclingDocument.load_from_json(json_path)

        # Find the handwritten text item
        found = False
        for item, _ in loaded_doc.iterate_items():
            if hasattr(item, "label") and item.label == DocItemLabel.HANDWRITTEN_TEXT:
                assert item.text == "Preserved handwritten text"
                found = True
                break

        assert found, "HANDWRITTEN_TEXT item not found after JSON roundtrip"


class TestTokenMappingCompleteness:
    """Test that all expected labels have token mappings."""

    # Labels that are expected to NOT have direct token mappings
    # (they're handled specially or are container types)
    SPECIAL_CASE_LABELS = {
        DocItemLabel.SECTION_HEADER,  # Handled with level suffix
        DocItemLabel.GRADING_SCALE,
        DocItemLabel.EMPTY_VALUE,
        DocItemLabel.FIELD_REGION,
        DocItemLabel.FIELD_HEADING,
        DocItemLabel.FIELD_ITEM,
        DocItemLabel.FIELD_KEY,
        DocItemLabel.FIELD_VALUE,
        DocItemLabel.FIELD_HINT,
        DocItemLabel.MARKER,
    }

    def test_all_text_labels_have_token_mappings(self):
        """Test that all text-type labels have token mappings.

        This proactively identifies any labels that would cause RuntimeError.
        """
        missing_mappings = []

        for label in DocItemLabel:
            if label in self.SPECIAL_CASE_LABELS:
                continue
            try:
                DocumentToken.create_token_name_from_doc_item_label(label.value)
            except RuntimeError:
                missing_mappings.append(label)

        assert not missing_mappings, (
            f"Labels missing token mappings: {[l.name for l in missing_mappings]}"
        )
