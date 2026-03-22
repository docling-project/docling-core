"""Tests for DocChunk expansion methods."""

import pytest

from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument, Size
from docling_core.types.doc.document import TableData

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 50

def check_lines_equal_in_order(text_a: str, text_b: str) -> bool:
    """
    Check if lines of string A are equal to lines of string B in the same order.
    
    This function splits both strings into lines and verifies that:
    1. All lines from A appear in B
    2. They appear in the same order
    3. Lines can be non-consecutive in B (other lines can appear between them)
    
    Args:
        text_a (str): First string (subset) to check
        text_b (str): Second string (superset) to check against
    
    Returns:
        bool: True if all lines of A appear in B in the same order, False otherwise
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise TypeError("Both inputs must be strings.")
    
    lines_a = [line for line in text_a.splitlines() if line.strip()]
    lines_b = [line for line in text_b.splitlines() if line.strip()]
    
    # If A is empty, it's always contained in B
    if not lines_a:
        return True
    
    # If B is empty but A is not, A cannot be contained in B
    if not lines_b:
        return False
    
    # Track position in B
    b_index = 0
    
    # Try to find each line of A in B in order
    for line_a in lines_a:
        found = False
        # Search for line_a starting from current position in B
        while b_index < len(lines_b):
            if lines_b[b_index] == line_a:
                found = True
                b_index += 1  # Move to next position in B
                break
            b_index += 1
        
        # If we couldn't find this line of A in B, return False
        if not found:
            return False
    
    return True


@pytest.fixture
def sample_doc():
    """Create a comprehensive sample document for testing with pages and various content types.
    
    Content is associated with pages through ProvenanceItem which includes page_no.
    When add_text/add_heading is called with prov parameter containing page_no,
    that content is associated with that specific page.
    """
    from docling_core.types.doc.document import ProvenanceItem, BoundingBox
    
    doc = DoclingDocument(name="test_doc")
    
    # Add pages
    page1 = doc.add_page(size=Size(width=612, height=792), page_no=1)
    page2 = doc.add_page(size=Size(width=612, height=792), page_no=2)
    
    # Section 1 on page 1 (explicitly set page_no in prov)
    doc.add_heading(
        text="Section 1",
        level=1,
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=50, t=50, r=550, b=80), charspan=(0, 9))
    )
    doc.add_text(
        text="This is the first paragraph.",
        label=DocItemLabel.PARAGRAPH,
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=50, t=90, r=550, b=120), charspan=(10, 38))
    )
    doc.add_text(
        text="This is the second paragraph.",
        label=DocItemLabel.PARAGRAPH,
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=50, t=130, r=550, b=160), charspan=(39, 68))
    )
    
    # Section 2 on page 2 with list (explicitly set page_no=2 in prov)
    doc.add_heading(
        text="Section 2",
        level=1,
        prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=50, t=50, r=550, b=80), charspan=(69, 78))
    )
    doc.add_text(
        text="Content in section 2.",
        label=DocItemLabel.PARAGRAPH,
        prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=50, t=90, r=550, b=120), charspan=(79, 100))
    )
    
    # Add a list in section 2 on page 2
    list_group = doc.add_list_group()
    doc.add_list_item(
        text="First list item",
        enumerated=False,
        parent=list_group,
        prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=70, t=130, r=550, b=150), charspan=(101, 116))
    )
    doc.add_list_item(
        text="Second list item",
        enumerated=False,
        parent=list_group,
        prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=70, t=160, r=550, b=180), charspan=(117, 133))
    )
    doc.add_list_item(
        text="Third list item",
        enumerated=False,
        parent=list_group,
        prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=70, t=190, r=550, b=210), charspan=(134, 149))
    )
    
    # Add a table on page 2
    table_data = TableData(num_cols=2)
    table_data.add_row(["Header 1", "Header 2"])
    table_data.add_row(["Value 1", "Value 2"])
    table_data.add_row(["Value 3", "Value 4"])
    table_data.add_row(["Value 5", "Value 6"])
    doc.add_table(
        data=table_data,
        prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=50, t=220, r=550, b=300), charspan=(150, 200))
    )
    
    return doc

class MarkdownSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
              
            )

@pytest.fixture
def hybrid_chunker():
    """Create a reusable HybridChunker instance."""
    return HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(
            model_name=EMBED_MODEL_ID,
            max_tokens=MAX_TOKENS,
        ),
        serializer_provider=MarkdownSerializerProvider(),
        repeat_table_header=True
    )    

@pytest.fixture
def sample_chunks(sample_doc, hybrid_chunker):
    """Create chunks from sample_doc once and cache them."""
    chunks = list(hybrid_chunker.chunk(dl_doc=sample_doc))
    assert len(chunks) > 0, "Expected at least one chunk to be created"
    return chunks 
    

@pytest.fixture
def sample_serializer(sample_doc, hybrid_chunker):
    """Create serializer for sample_doc once and cache it."""
    return hybrid_chunker.serializer_provider.get_serializer(sample_doc)
   

class TestGetTopContainingItems:
    """Tests for _get_top_containing_items method."""
    
    # helper method: recursively traverse top item children to find chunk items
    def _find_chunk_item_in_descendants(self, item, doc, target_refs):
            """Recursively check if any target_refs are in item's descendants."""
            # Check if this item itself is a target
            if item.self_ref in target_refs:
                return True
            
            # Check children if item has them
            if hasattr(item, 'children') and item.children:
                for child_ref in item.children:
                    child = child_ref.resolve(doc)
                    if self._find_chunk_item_in_descendants(child, doc, target_refs):
                        return True
            
            return False


    def test_get_top_items_basic(self, sample_doc, sample_chunks):
        """Test getting top-level items from a chunk."""
        assert len(sample_chunks) > 0, "Should have at least one chunk"
        
        for chunk in sample_chunks:
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
          
            for top_item in top_items:
                assert self._find_chunk_item_in_descendants(top_item, sample_doc, chunk_item_refs), (
                    f"Could not find any chunk items in descendants of top item {top_item.self_ref}"
            )
       
    
    def test_get_top_items_maintains_order(self, sample_doc, sample_chunks):
        """Test that top items maintain document reading order."""
        for chunk in sample_chunks:
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
    
    def test_expand_to_item_basic(self, sample_doc, sample_serializer, sample_chunks):
        """Test basic expansion to full items."""
        
        for chunk in sample_chunks:
            expanded = chunk.expand_to_item(
                dl_doc=sample_doc,
                serializer=sample_serializer
            )
        
            assert expanded is not None, "Should return expanded chunk"
            assert isinstance(expanded, DocChunk), "Should return DocChunk instance"
        
            # Expanded chunk should have content
            assert len(expanded.text.strip()) > 0, "Expanded chunk should have text"
        
            # Expanded chunk text should contain original chunk text (or be a superset)
            assert check_lines_equal_in_order(chunk.text,expanded.text), (
                f"Expanded chunk should contain of original chunk text. "
                f"original {chunk.text}"
                f"expanded: {expanded.text}"
            )
            assert expanded.meta.origin == chunk.meta.origin, (
                "Origin should be preserved"
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
        chunk = DocChunk(text="original text", meta=meta)
        
        # Call expand_to_object with failing serializer
        # Should catch the exception and return original chunk
        expanded = chunk.expand_to_item(
            dl_doc=sample_doc,
            serializer=FailingSerializer()
        )
        
        # Should return original chunk when serialization fails
        assert expanded == chunk, "Should return original chunk when serialization fails"
       

class TestExpandToPage:
    """Tests for expand_to_page method."""
                   
    
    def test_expand_to_page_basic(self, sample_doc, sample_chunks, sample_serializer):
        """Test that page expansion includes all page content."""
        
        for chunk in sample_chunks:
                    
            # Get page numbers from chunk
            page_ids = [
                i.page_no for item in chunk.meta.doc_items for i in item.prov
            ]
            
            if page_ids:
                expanded = chunk.expand_to_page(
                    doc=sample_doc,
                    serializer=sample_serializer
                )
                assert expanded is not None, "Should return expanded chunk"
                assert isinstance(expanded, DocChunk), "Should return DocChunk"
                # Expanded text should contain page content
                assert len(expanded.text) > 0, "Expanded chunk should have text"
                
                # Verify it contains original 
                assert check_lines_equal_in_order(chunk.text,expanded.text), (
                        "Expanded text should contain original"
                    )

                 # Metadata fields should be updated with expanded content
                assert expanded.meta.origin == chunk.meta.origin, "Expanded chunk should have metadata"
                def get_ref_items(chunk:DocChunk):
                    return [item.self_ref for item in chunk.meta.doc_items]
                assert set(get_ref_items(chunk)).issubset(get_ref_items(expanded)) , (
                "Expanded chunk should have at least as many doc_items as original"
            )    
    
    def test_expand_to_page_no_pages(self, hybrid_chunker):
        """Test expand_to_page when document has no pages for all chunks."""
        # Create a document without pages
        doc_no_pages = DoclingDocument(name="no_pages_doc")
        doc_no_pages.add_heading(text="Section 1", level=1)
        doc_no_pages.add_text(text="Some content.", label=DocItemLabel.PARAGRAPH)
        
        chunks = list(hybrid_chunker.chunk(dl_doc=doc_no_pages))
        serializer = hybrid_chunker.serializer_provider.get_serializer(doc_no_pages)
        
        assert len(chunks) > 0, "Should have at least one chunk"
        
        for chunk in chunks:
            result = chunk.expand_to_page(
                doc=doc_no_pages,
                serializer=serializer
            )
            
            # Should return original chunk when no pages
            assert result == chunk, "Should return original chunk when document has no pages"
    
 