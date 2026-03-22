"""Tests for DocChunk expansion methods."""

import pytest

from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument, Size
from docling_core.types.doc.document import TableData

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
INPUT_FILE = "test/data/chunker/2_inp_dl_doc.json"


@pytest.fixture
def sample_doc():
    """Create a sample document for testing."""
    doc = DoclingDocument(name="test_doc")
    
    # Add some content with hierarchy
    doc.add_heading(text="Section 1", level=1)
    doc.add_text(text="This is the first paragraph.", label=DocItemLabel.PARAGRAPH)
    doc.add_text(text="This is the second paragraph.", label=DocItemLabel.PARAGRAPH)
    
    doc.add_heading(text="Section 2", level=1)
    doc.add_text(text="Content in section 2.", label=DocItemLabel.PARAGRAPH)
    
    # Add a table
    table_data = TableData(num_cols=2)
    table_data.add_row(["Header 1", "Header 2"])
    table_data.add_row(["Value 1", "Value 2"])
    doc.add_table(data=table_data)
    
    return doc


@pytest.fixture
def sample_doc_with_pages():
    """Create a sample document with page information."""
    doc = DoclingDocument(name="test_doc_pages")
    
    # Add page
    page = doc.add_page(size=Size(width=612, height=792), page_no=1)
    
    # Add content to page 1
    doc.add_heading(text="Page 1 Heading", level=1)
    doc.add_text(text="Content on page 1.", label=DocItemLabel.PARAGRAPH)
    
    # Add another page
    page2 = doc.add_page(size=Size(width=612, height=792), page_no=2)
    
    # Add content to page 2
    doc.add_heading(text="Page 2 Heading", level=1)
    doc.add_text(text="Content on page 2.", label=DocItemLabel.PARAGRAPH)
    
    return doc


@pytest.fixture
def hybrid_chunker():
    """Create a reusable HybridChunker instance."""
    return HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(
            model_name=EMBED_MODEL_ID,
            max_tokens=MAX_TOKENS,
        ),
    )


@pytest.fixture
def chunking_serializer(sample_doc):
    """Create a chunking serializer for testing."""
    return ChunkingDocSerializer(doc=sample_doc)


class TestGetTopContainingItems:
    """Tests for _get_top_containing_items method."""
    
    def test_get_top_items_basic(self, sample_doc, hybrid_chunker):
        """Test getting top-level items from a chunk."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc))
        assert len(chunks) > 0, "Should have at least one chunk"
        
        # Test the first chunk - convert to DocChunk
        chunk = DocChunk.model_validate(chunks[0])
        top_items = chunk._get_top_containing_items(sample_doc)
        
        assert top_items is not None, "Should return top items"
        assert len(top_items) > 0, "Should have at least one top item"
        
        # Verify all returned items are direct children of body
        for item in top_items:
            assert item.parent == sample_doc.body.get_ref(), (
                f"Item {item.self_ref} should be direct child of body"
            )
        
        # Verify that at least one doc_item from the chunk is a descendant of a top item
        chunk_item_refs = {item.self_ref for item in chunk.meta.doc_items}
                
        # Additional check: recursively traverse top item children to find chunk items
        def find_chunk_item_in_descendants(item, doc, target_refs):
            """Recursively check if any target_refs are in item's descendants."""
            # Check if this item itself is a target
            if item.self_ref in target_refs:
                return True
            
            # Check children if item has them
            if hasattr(item, 'children') and item.children:
                for child_ref in item.children:
                    child = child_ref.resolve(doc)
                    if find_chunk_item_in_descendants(child, doc, target_refs):
                        return True
            
            return False
        
        
        for top_item in top_items:
            assert find_chunk_item_in_descendants(top_item, sample_doc, chunk_item_refs), (
                f"Could not find any chunk items in descendants of top item {top_item.self_ref}"
            )
       
    
    def test_get_top_items_maintains_order(self, sample_doc, hybrid_chunker):
        """Test that top items maintain document reading order."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc))
        
        for chunk in chunks:
            top_items = chunk._get_top_containing_items(sample_doc)
            if top_items and len(top_items) > 1:
                # Get the order in the document body
                body_refs = [ref.cref for ref in sample_doc.body.children]
                top_refs = [item.self_ref for item in top_items]
                
                # Verify order is maintained
                prev_idx = -1
                for ref in top_refs:
                    curr_idx = body_refs.index(ref)
                    assert curr_idx > prev_idx, "Items should maintain reading order"
                    prev_idx = curr_idx
    
    def test_get_top_items_empty_chunk(self):
        """Test _get_top_containing_items with chunk containing non-body items."""
        doc = DoclingDocument(name="empty_doc")
        text_item = doc.add_text(text="Some text", label=DocItemLabel.PARAGRAPH)
        
        # Create a chunk with a doc item that doesn't have proper parent
        # This simulates an edge case where get_top_containing_items might return None
        meta = DocMeta(doc_items=[text_item])
        chunk = DocChunk(text="test", meta=meta)
        
        # Should return the text item as top item since it's a direct child of body
        result = chunk._get_top_containing_items(doc)
        assert result is not None, "Should return top items for valid doc_items"
        assert len(result) > 0, "Should have at least one top item"


class TestExpandToItem:
    """Tests for expand_to_item method."""
    
    def test_expand_to_item_basic(self, sample_doc, chunking_serializer, hybrid_chunker):
        """Test basic expansion to full items."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc))
        assert len(chunks) > 0, "Should have chunks"
        
        # Expand the first chunk - convert to DocChunk
        original_chunk = DocChunk.model_validate(chunks[0])
        expanded_chunk = original_chunk.expand_to_item(
            dl_doc=sample_doc,
            serializer=chunking_serializer
        )
        
        assert expanded_chunk is not None, "Should return expanded chunk"
        assert isinstance(expanded_chunk, DocChunk), "Should return DocChunk instance"
        
        # Expanded chunk should have content
        assert len(expanded_chunk.text.strip()) > 0, "Expanded chunk should have text"
        
        # Expanded chunk text should contain original chunk text (or be a superset)
        assert original_chunk.text in expanded_chunk.text, (
            f"Expanded chunk should contain of original chunk text. "
            f"original {original_chunk.text}"
            f"expanded: {expanded_chunk.text}"
        )
    
  
    
    def test_expand_to_item_with_table(self, hybrid_chunker):
        """Test expansion with table content."""
        doc = DoclingDocument(name="table_doc")
        doc.add_heading(text="Table Section", level=1)
        
        # Add a table
        table_data = TableData(num_cols=3)
        table_data.add_row(["Col1", "Col2", "Col3"])
        table_data.add_row(["A", "B", "C"])
        table_data.add_row(["D", "E", "F"])
        table_item = doc.add_table(data=table_data)
        
        chunks = list(hybrid_chunker.chunk(dl_doc=doc))
        serializer = hybrid_chunker.serializer_provider.get_serializer(doc)
        
        # Serialize the table to get expected text
        table_serialized = serializer.serialize(item=table_item)
        table_text = table_serialized.text
        
        # Find chunk with table
        table_chunk = None
        for c in chunks:
            chunk = DocChunk.model_validate(c)
            if any(hasattr(item, 'data') for item in chunk.meta.doc_items):
                table_chunk = chunk
                break
        
        if table_chunk:
            expanded = table_chunk.expand_to_item(
                dl_doc=doc,
                serializer=serializer
            )
            
          
            
            # Verify that the serialized table text is in expanded text
            assert table_text in expanded.text, (
                f"Expanded chunk should contain the full serialized table text. "
                f"table text: {table_text}\n"
                f"expanded: {expanded.text}"
            )
    
    def test_expand_to_item_error_handling(self, sample_doc, hybrid_chunker):
        """Test error handling in expand_to_item when serialization fails."""
        # Create a mock serializer that raises an exception
        class FailingSerializer:
            def serialize(self, item):
                raise RuntimeError("Serialization failed")
        
        # Create a chunk with valid doc items
        text_item = sample_doc.texts[0]
        meta = DocMeta(doc_items=[text_item])
        original_chunk = DocChunk(text="original text", meta=meta)
        
        # Call expand_to_object with failing serializer
        # Should catch the exception and return original chunk
        result = original_chunk.expand_to_item(
            dl_doc=sample_doc,
            serializer=FailingSerializer()
        )
        
        # Should return original chunk when serialization fails
        assert result == original_chunk, "Should return original chunk when serialization fails"
        assert result.text == "original text", "Original text should be preserved"
    
    def test_expand_to_item_preserves_metadata(self, sample_doc, chunking_serializer, hybrid_chunker):
        """Test that expansion preserves chunk metadata."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc))
        if len(chunks) > 0:
            original = chunks[0]
            expanded = original.expand_to_item(
                dl_doc=sample_doc,
                serializer=chunking_serializer
            )
            
          
            assert expanded.meta.origin == original.meta.origin, (
                "Origin should be preserved"
            )


class TestExpandToPage:
    """Tests for expand_to_page method."""
    
    def test_expand_to_page_basic(self, sample_doc_with_pages, hybrid_chunker):
        """Test basic page expansion."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc_with_pages))
        serializer = hybrid_chunker.serializer_provider.get_serializer(sample_doc_with_pages)
        
        if len(chunks) > 0:
            chunk = chunks[0]
            expanded = chunk.expand_to_page(
                doc=sample_doc_with_pages,
                serializer=serializer
            )
            
            assert expanded is not None, "Should return expanded chunk"
            assert isinstance(expanded, DocChunk), "Should return DocChunk"
    
    def test_expand_to_page_includes_page_content(self, sample_doc_with_pages, hybrid_chunker):
        """Test that page expansion includes all page content."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc_with_pages))
        serializer = hybrid_chunker.serializer_provider.get_serializer(sample_doc_with_pages)
        
        for c in chunks:
            chunk = DocChunk.model_validate(c)
            # Get page numbers from chunk
            page_ids = [
                i.page_no for item in chunk.meta.doc_items for i in item.prov
            ]
            
            if page_ids:
                expanded = chunk.expand_to_page(
                    doc=sample_doc_with_pages,
                    serializer=serializer
                )
                
                # Expanded text should contain page content
                assert len(expanded.text) > 0, "Expanded chunk should have text"
                
                # Verify it contains original 
                assert chunk.text in expanded.text, (
                        "Expanded text should contain original"
                    )
    
    def test_expand_to_page_no_pages(self, sample_doc):
        """Test expand_to_page when document has no pages."""
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc))
        serializer = chunker.serializer_provider.get_serializer(sample_doc)
        
        if len(chunks) > 0:
            chunk = DocChunk.model_validate(chunks[0])
            result = chunk.expand_to_page(
                doc=sample_doc,
                serializer=serializer
            )
            
            # Should return original chunk when no pages
            assert result == chunk, "Should return original chunk when no pages"
    
        
    def test_expand_to_page_preserves_metadata(self, sample_doc_with_pages, hybrid_chunker):
        """Test that page expansion preserves metadata."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc_with_pages))
        serializer = hybrid_chunker.serializer_provider.get_serializer(sample_doc_with_pages)
        
        if len(chunks) > 0:
            original = DocChunk.model_validate(chunks[0])
            expanded = original.expand_to_page(
                doc=sample_doc_with_pages,
                serializer=serializer
            )
            
            # Metadata should be preserved
            assert expanded.meta == original.meta, "Metadata should be preserved"

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
       
    def test_expand_with_none_serializer(self, sample_doc, hybrid_chunker):
        """Test expansion with None serializer."""
        chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc))
        if len(chunks) > 0:
            # Convert to DocChunk to access expansion methods
            chunk = DocChunk.model_validate(chunks[0])
            # Should handle None serializer gracefully by returning original chunk
            # (errors are caught and logged, not raised)
            result = chunk.expand_to_item(
                dl_doc=sample_doc,
                serializer=None
            )
            # Should return original chunk when serializer fails
            assert result == chunk, "Should return original chunk when serializer is None"
    
    

