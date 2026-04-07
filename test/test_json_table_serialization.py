"""Test JSON table serialization."""

import json
from pathlib import Path

import pandas as pd
import pytest

from docling_core.transforms.serializer.json_table import (
    JsonTableParams,
    JsonTableSerializer,
    _clean_text,
    _detect_header_type,
    _detect_table_metadata,
    _is_valid_table,
)
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc import TableCell, TableData
from docling_core.types.doc.document import DoclingDocument


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        assert _clean_text("  hello  ") == "hello"
        assert _clean_text("hello\nworld") == "hello world"
        assert _clean_text("hello  \t  world") == "hello world"
        assert _clean_text("") == ""
        # Test with empty/None-like values
        assert _clean_text("") == ""

    def test_is_valid_table_pass(self):
        """Test valid table detection."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })
        assert _is_valid_table(df, min_rows=2, min_cols=2)

    def test_is_valid_table_fail_rows(self):
        """Test invalid table - too few rows."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert not _is_valid_table(df, min_rows=2)

    def test_is_valid_table_fail_cols(self):
        """Test invalid table - too few columns."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        assert not _is_valid_table(df, min_cols=2)

    def test_is_valid_table_fail_empty_ratio(self):
        """Test invalid table - too many empty cells."""
        df = pd.DataFrame({
            "A": [1, None, None],
            "B": [None, None, 6]
        })
        # 4 out of 6 cells are empty = 66% > 50%
        assert not _is_valid_table(df, max_empty_ratio=0.5)

    def test_detect_header_type_top_only(self):
        """Test top-only header detection."""
        df = pd.DataFrame({
            "Product": ["Apple", "Orange"],
            "Q1": [100, 80]
        })
        assert _detect_header_type(df) == "top_only"

    def test_detect_header_type_both(self):
        """Test both headers detection."""
        df = pd.DataFrame({
            "": ["Apple", "Orange"],
            "Q1": [100, 80],
            "Q2": [150, 120]
        })
        assert _detect_header_type(df) == "both"

    def test_detect_header_type_both_with_none(self):
        """Test both headers detection with None."""
        df = pd.DataFrame({
            None: ["Apple", "Orange"],
            "Q1": [100, 80]
        })
        # Top-left is None, should detect as "both"
        assert _detect_header_type(df) == "both"

    def test_detect_table_metadata_with_title(self):
        """Test metadata detection with title."""
        df = pd.DataFrame({
            "A": ["Sales Report", "Apple", "Orange"],
            "B": ["Sales Report", 100, 80]
        })
        metadata = _detect_table_metadata(df)
        assert metadata["title"] == "Sales Report"
        assert metadata["data_start_row"] == 1

    def test_detect_table_metadata_with_description(self):
        """Test metadata detection with description."""
        df = pd.DataFrame({
            "A": ["Apple", "Orange", "Source: Finance"],
            "B": [100, 80, "Source: Finance"]
        })
        metadata = _detect_table_metadata(df)
        assert metadata["description"] == "Source: Finance"
        assert metadata["data_end_row"] == 2

    def test_detect_table_metadata_no_metadata(self):
        """Test metadata detection with no metadata."""
        df = pd.DataFrame({
            "A": ["Apple", "Orange"],
            "B": [100, 80]
        })
        metadata = _detect_table_metadata(df)
        assert metadata["title"] is None
        assert metadata["description"] is None
        assert metadata["data_start_row"] == 0
        assert metadata["data_end_row"] == 2


# ============================================================================
# JsonTableSerializer Tests
# ============================================================================


class TestJsonTableSerializer:
    """Test JsonTableSerializer class."""

    @pytest.fixture
    def simple_table_doc(self):
        """Create a document with a simple table."""
        doc = DoclingDocument(name="Test Doc")
        
        # Create a simple 2x2 table
        table = doc.add_table(data=TableData(num_rows=2, num_cols=2))
        
        # Add cells using add_table_cell method
        cells_data = [
            (0, 0, "Product"),
            (0, 1, "Price"),
            (1, 0, "Apple"),
            (1, 1, "$1.00"),
        ]
        
        for row, col, text in cells_data:
            doc.add_table_cell(
                table_item=table,
                cell=TableCell(
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    text=text,
                ),
            )
        
        return doc, table

    @pytest.fixture
    def table_with_metadata_doc(self):
        """Create a document with a table that has title and description."""
        doc = DoclingDocument(name="Test Doc")
        
        # Create table with title and description
        table = doc.add_table(data=TableData(num_rows=4, num_cols=2))
        
        # Add cells
        cells_data = [
            # Title row (merged cells pattern - same text)
            (0, 0, "Sales Report"),
            (0, 1, "Sales Report"),
            # Header row
            (1, 0, "Product"),
            (1, 1, "Price"),
            # Data row
            (2, 0, "Apple"),
            (2, 1, "$1.00"),
            # Description row (merged cells pattern - same text)
            (3, 0, "Source: Finance"),
            (3, 1, "Source: Finance"),
        ]
        
        for row, col, text in cells_data:
            doc.add_table_cell(
                table_item=table,
                cell=TableCell(
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    text=text,
                ),
            )
        
        return doc, table

    def test_serialize_basic_table(self, simple_table_doc):
        """Test basic table serialization."""
        doc, table = simple_table_doc
        serializer = JsonTableSerializer()
        
        # Create a mock doc serializer
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc
        )
        
        assert result.text
        table_json = json.loads(result.text)
        assert "table_index" in table_json
        assert "data" in table_json
        assert table_json["header_type"] in ["top_only", "both"]

    def test_serialize_with_validation(self, simple_table_doc):
        """Test table serialization with validation."""
        doc, table = simple_table_doc
        serializer = JsonTableSerializer()
        
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        # Should pass validation (2x2 table)
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc,
            validate_tables=True,
            min_rows=2,
            min_cols=2
        )
        
        table_json = json.loads(result.text)
        assert "error" not in table_json

    def test_serialize_natural_language(self, simple_table_doc):
        """Test natural language output."""
        doc, table = simple_table_doc
        serializer = JsonTableSerializer()
        
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc,
            convert_to_natural_language=True
        )
        
        # Should be text, not JSON
        assert not result.text.startswith("{")
        assert ":" in result.text  # key: value format

    def test_serialize_simple_json_format(self, simple_table_doc):
        """Test simple JSON format."""
        doc, table = simple_table_doc
        serializer = JsonTableSerializer()
        
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc,
            output_format="simple_json"
        )
        
        table_json = json.loads(result.text)
        assert "columns" in table_json
        assert "rows" in table_json

    def test_get_header_and_body_lines_json(self, simple_table_doc):
        """Test splitting JSON table for chunking."""
        doc, table = simple_table_doc
        serializer = JsonTableSerializer()
        
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc
        )
        
        header_lines, body_lines = serializer.get_header_and_body_lines(
            table_text=result.text
        )
        
        # Should have some header lines (metadata)
        assert len(header_lines) > 0
        # Should have some body lines (data)
        assert len(body_lines) > 0

    def test_get_header_and_body_lines_natural_language(self, simple_table_doc):
        """Test splitting natural language table for chunking."""
        doc, table = simple_table_doc
        serializer = JsonTableSerializer()
        
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc,
            convert_to_natural_language=True
        )
        
        header_lines, body_lines = serializer.get_header_and_body_lines(
            table_text=result.text
        )
        
        # Natural language format: no header, all body
        assert len(header_lines) == 0
        assert len(body_lines) > 0


# ============================================================================
# JsonTableDocSerializer Tests
# ============================================================================


class TestMarkdownDocWithJsonTables:
    """Test using JsonTableSerializer with MarkdownDocSerializer."""

    @pytest.fixture
    def doc_with_text_and_table(self):
        """Create a document with both text and table."""
        from docling_core.types.doc import DocItemLabel
        
        doc = DoclingDocument(name="Mixed Doc")
        
        # Add some text
        doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text="This is a paragraph before the table."
        )
        
        # Add a table
        table = doc.add_table(data=TableData(num_rows=2, num_cols=2))
        
        # Add cells
        cells_data = [
            (0, 0, "Product"),
            (0, 1, "Price"),
            (1, 0, "Apple"),
            (1, 1, "$1.00"),
        ]
        
        for row, col, text in cells_data:
            doc.add_table_cell(
                table_item=table,
                cell=TableCell(
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + 1,
                    text=text,
                ),
            )
        
        # Add more text
        doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text="This is a paragraph after the table."
        )
        
        return doc

    def test_mixed_format_document(self, doc_with_text_and_table):
        """Test document with text and tables using custom table serializer."""
        from docling_core.transforms.serializer.markdown import MarkdownParams
        
        doc = doc_with_text_and_table
        
        # Use MarkdownDocSerializer with JsonTableSerializer
        serializer = MarkdownDocSerializer(
            doc=doc,
            params=MarkdownParams()
        )
        # Replace the table serializer
        serializer.table_serializer = JsonTableSerializer()
        
        result = serializer.serialize()
        
        # Should contain both text and JSON
        assert "paragraph" in result.text.lower()
        assert "{" in result.text  # JSON object
        assert "table_index" in result.text or "Product" in result.text

    def test_json_table_with_markdown_text(self, doc_with_text_and_table):
        """Test that text remains in markdown while tables are JSON."""
        from docling_core.transforms.serializer.markdown import MarkdownParams
        
        doc = doc_with_text_and_table
        
        # Use MarkdownDocSerializer with JsonTableSerializer
        serializer = MarkdownDocSerializer(
            doc=doc,
            params=MarkdownParams()
        )
        serializer.table_serializer = JsonTableSerializer()
        
        result = serializer.serialize()
        
        # Text should be plain (no special formatting in this case)
        # Tables should be JSON
        assert "paragraph" in result.text.lower()
        
        # Try to find and parse the JSON table
        lines = result.text.split("\n")
        json_found = False
        for line in lines:
            if line.strip().startswith("{"):
                try:
                    json.loads(line)
                    json_found = True
                    break
                except json.JSONDecodeError:
                    # Might be multi-line JSON, try to find complete object
                    pass
        
        # At minimum, should have JSON-like content
        assert "{" in result.text and "}" in result.text


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration with other components."""

    def test_json_params_inheritance(self):
        """Test that JsonTableParams inherits from CommonParams."""
        params = JsonTableParams(
            output_format="smart_json",
            min_rows=3,
            include_metadata=True
        )
        
        assert params.output_format == "smart_json"
        assert params.min_rows == 3
        assert params.include_metadata is True
        # Should also have CommonParams attributes
        assert hasattr(params, "labels")
        assert hasattr(params, "layers")

    def test_serializer_with_empty_table(self):
        """Test handling of empty table."""
        doc = DoclingDocument(name="Empty Table Doc")
        
        # Create empty table
        table_data = TableData(num_rows=0, num_cols=0, grid=[])
        table = doc.add_table(data=table_data)
        
        serializer = JsonTableSerializer()
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
        doc_ser = MarkdownDocSerializer(doc=doc)
        
        result = serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc,
            validate_tables=True
        )
        
        # Should handle gracefully
        assert result.text
        # Likely will fail validation or return error
        table_json = json.loads(result.text)
        assert "error" in table_json or "data" in table_json


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
