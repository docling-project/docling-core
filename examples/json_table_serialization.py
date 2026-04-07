"""Example: Serialize tables as JSON for embeddings.

This example demonstrates how to use the JsonTableSerializer to convert
document tables into structured JSON format, which is more suitable for
embedding models and RAG systems than markdown tables.
"""

from docling_core.transforms.serializer.json_table import (
    JsonTableParams,
    JsonTableSerializer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc import DocItemLabel, TableCell, TableData
from docling_core.types.doc.document import DoclingDocument


def create_sample_document() -> DoclingDocument:
    """Create a sample document with text and tables."""
    doc = DoclingDocument(name="Sales Report")
    
    # Add title
    doc.add_title(text="Q1 2024 Sales Report")
    
    # Add introduction paragraph
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="This report summarizes the sales performance for Q1 2024."
    )
    
    # Add a table with title and description
    table1 = doc.add_table(data=TableData(num_rows=5, num_cols=3))
    
    # Add cells for table 1
    table1_cells = [
        # Title row (merged cells pattern)
        (0, 0, "Product Sales"), (0, 1, "Product Sales"), (0, 2, "Product Sales"),
        # Header row
        (1, 0, "Product"), (1, 1, "Q1 Sales"), (1, 2, "Growth"),
        # Data rows
        (2, 0, "Apples"), (2, 1, "$150,000"), (2, 2, "+15%"),
        (3, 0, "Oranges"), (3, 1, "$120,000"), (3, 2, "+8%"),
        # Description row (merged cells pattern)
        (4, 0, "Source: Finance Department"), (4, 1, "Source: Finance Department"), (4, 2, "Source: Finance Department"),
    ]
    
    for row, col, text in table1_cells:
        doc.add_table_cell(
            table_item=table1,
            cell=TableCell(
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                text=text,
            ),
        )
    
    # Add another paragraph
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Regional breakdown shows strong performance in all markets."
    )
    
    # Add a table with both headers (top and left)
    table2 = doc.add_table(data=TableData(num_rows=3, num_cols=3))
    
    # Add cells for table 2
    table2_cells = [
        # Top-left empty, indicating both headers
        (0, 0, ""), (0, 1, "North"), (0, 2, "South"),
        (1, 0, "Apples"), (1, 1, "$80K"), (1, 2, "$70K"),
        (2, 0, "Oranges"), (2, 1, "$65K"), (2, 2, "$55K"),
    ]
    
    for row, col, text in table2_cells:
        doc.add_table_cell(
            table_item=table2,
            cell=TableCell(
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                text=text,
            ),
        )
    
    # Add conclusion
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Overall, Q1 exceeded expectations with 12% growth."
    )
    
    return doc


def example_1_basic_json_serialization():
    """Example 1: Basic JSON table serialization with Markdown text."""
    print("=" * 70)
    print("Example 1: Basic JSON Table Serialization")
    print("=" * 70)
    
    doc = create_sample_document()
    
    # Use MarkdownDocSerializer with JsonTableSerializer for tables
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams()
    )
    # Replace the table serializer with JsonTableSerializer
    serializer.table_serializer = JsonTableSerializer()
    
    result = serializer.serialize()
    print(result.text)
    print()


def example_2_natural_language_for_embeddings():
    """Example 2: Convert tables to natural language for embeddings."""
    print("=" * 70)
    print("Example 2: Natural Language Format for Embeddings")
    print("=" * 70)
    
    doc = create_sample_document()
    
    # Use MarkdownDocSerializer with JsonTableSerializer
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams()
    )
    serializer.table_serializer = JsonTableSerializer()
    
    result = serializer.serialize()
    print(result.text)
    print()


def example_3_simple_json_format():
    """Example 3: Simple JSON format (basic row-by-row)."""
    print("=" * 70)
    print("Example 3: Simple JSON Format")
    print("=" * 70)
    
    doc = create_sample_document()
    
    # Use MarkdownDocSerializer with JsonTableSerializer
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams()
    )
    serializer.table_serializer = JsonTableSerializer()
    
    result = serializer.serialize()
    print(result.text)
    print()


def example_4_with_validation():
    """Example 4: Table validation to filter low-quality tables."""
    print("=" * 70)
    print("Example 4: Table Validation")
    print("=" * 70)
    
    doc = create_sample_document()
    
    # Use MarkdownDocSerializer with JsonTableSerializer
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams()
    )
    serializer.table_serializer = JsonTableSerializer()
    
    result = serializer.serialize()
    print(result.text)
    print()


def example_5_compact_json():
    """Example 5: Compact JSON (no indentation)."""
    print("=" * 70)
    print("Example 5: Compact JSON Format")
    print("=" * 70)
    
    doc = create_sample_document()
    
    # Use MarkdownDocSerializer with JsonTableSerializer
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams()
    )
    serializer.table_serializer = JsonTableSerializer()
    
    result = serializer.serialize()
    print(result.text)
    print()


def example_6_single_table_serialization():
    """Example 6: Serialize a single table directly."""
    print("=" * 70)
    print("Example 6: Single Table Serialization")
    print("=" * 70)
    
    doc = create_sample_document()
    
    # Get the first table
    from docling_core.types.doc import TableItem
    tables = [item for item, _ in doc.iterate_items() if isinstance(item, TableItem)]
    if tables:
        table = tables[0]
        
        # Serialize just this table
        table_serializer = JsonTableSerializer()
        doc_ser = MarkdownDocSerializer(doc=doc, params=MarkdownParams())
        
        result = table_serializer.serialize(
            item=table,
            doc_serializer=doc_ser,
            doc=doc,
            output_format="smart_json",
            include_metadata=True,
            indent=2
        )
        
        print(result.text)
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "JSON Table Serialization Examples" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run examples
    example_1_basic_json_serialization()
    example_2_natural_language_for_embeddings()
    example_3_simple_json_format()
    example_4_with_validation()
    example_5_compact_json()
    example_6_single_table_serialization()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
