"""Tests for DocChunk expansion methods."""

from ast import Or
import re

import pytest

from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument, Size
from docling_core.types.doc.document import TableData

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
INPUT_FILE = "test/data/chunker/2_inp_dl_doc.json"


def text_contains_ignoring_whitespace(haystack: str, needle: str) -> bool:
    """
    Check if needle text is contained in haystack, ignoring whitespace differences.
    
    Normalizes both strings by removing all whitespace characters.
    
    Args:
        haystack: The text to search in
        needle: The text to search for
        
    Returns:
        True if needle is found in haystack (ignoring whitespace), False otherwise
    """
    # Remove all whitespace
    haystack_normalized = re.sub(r'\s+', '', haystack)
    needle_normalized = re.sub(r'\s+', '', needle)
    return needle_normalized in haystack_normalized


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
def chunking_serializer(sample_doc):
    """Create a chunking serializer for testing."""
    return ChunkingDocSerializer(doc=sample_doc)


class TestGetTopContainingObjects:
    """Tests for get_top_containing_objects method."""
    
    def test_get_top_objects_basic(self, sample_doc):
        """Test getting top-level objects from a chunk."""
        # Create a chunker and get chunks
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc))
        assert len(chunks) > 0, "Should have at least one chunk"
        
        # Test the first chunk - convert to DocChunk
        chunk = DocChunk.model_validate(chunks[0])
        top_objects = chunk.get_top_containing_objects(sample_doc)
        
        assert top_objects is not None, "Should return top objects"
        assert len(top_objects) > 0, "Should have at least one top object"
        
        # Verify all returned objects are direct children of body
        for obj in top_objects:
            assert obj.parent == sample_doc.body.get_ref(), (
                f"Object {obj.self_ref} should be direct child of body"
            )
        
        # Verify that at least one doc_item from the chunk is a descendant of a top object
        chunk_item_refs = {item.self_ref for item in chunk.meta.doc_items}
                
        # Additional check: recursively traverse top object children to find chunk items
        def find_chunk_item_in_descendants(obj, doc, target_refs):
            """Recursively check if any target_refs are in obj's descendants."""
            # Check if this object itself is a target
            if obj.self_ref in target_refs:
                return True
            
            # Check children if object has them
            if hasattr(obj, 'children') and obj.children:
                for child_ref in obj.children:
                    child = child_ref.resolve(doc)
                    if find_chunk_item_in_descendants(child, doc, target_refs):
                        return True
            
            return False
        
        
        for top_obj in top_objects:
            assert find_chunk_item_in_descendants(top_obj, sample_doc, chunk_item_refs), (
                f"Could not find any chunk items in descendants of top object {top_obj.self_ref}"
            )
       
    
    def test_get_top_objects_maintains_order(self, sample_doc):
        """Test that top objects maintain document reading order."""
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc))
        
        for chunk in chunks:
            top_objects = chunk.get_top_containing_objects(sample_doc)
            if top_objects and len(top_objects) > 1:
                # Get the order in the document body
                body_refs = [ref.cref for ref in sample_doc.body.children]
                top_refs = [obj.self_ref for obj in top_objects]
                
                # Verify order is maintained
                prev_idx = -1
                for ref in top_refs:
                    curr_idx = body_refs.index(ref)
                    assert curr_idx > prev_idx, "Objects should maintain reading order"
                    prev_idx = curr_idx
    
    def test_get_top_objects_empty_chunk(self):
        """Test get_top_containing_objects with chunk containing non-body items."""
        doc = DoclingDocument(name="empty_doc")
        text_item = doc.add_text(text="Some text", label=DocItemLabel.PARAGRAPH)
        
        # Create a chunk with a doc item that doesn't have proper parent
        # This simulates an edge case where get_top_containing_objects might return None
        meta = DocMeta(doc_items=[text_item])
        chunk = DocChunk(text="test", meta=meta)
        
        # Should return the text item as top object since it's a direct child of body
        result = chunk.get_top_containing_objects(doc)
        assert result is not None, "Should return top objects for valid doc_items"
        assert len(result) > 0, "Should have at least one top object"


class TestExpandToObject:
    """Tests for expand_to_object method."""
    
    def test_expand_to_object_basic(self, sample_doc, chunking_serializer):
        """Test basic expansion to full objects."""
        # Create chunks
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc))
        assert len(chunks) > 0, "Should have chunks"
        
        # Expand the first chunk - convert to DocChunk
        original_chunk = DocChunk.model_validate(chunks[0])
        expanded_chunk = original_chunk.expand_to_object(
            dl_doc=sample_doc,
            serializer=chunking_serializer
        )
        
        assert expanded_chunk is not None, "Should return expanded chunk"
        assert isinstance(expanded_chunk, DocChunk), "Should return DocChunk instance"
        
        # Expanded chunk should have content
        assert len(expanded_chunk.text.strip()) > 0, "Expanded chunk should have text"
        
        # Expanded chunk text should contain original chunk text (or be a superset)
        assert text_contains_ignoring_whitespace(expanded_chunk.text, needle=original_chunk.text), (
            f"Expanded chunk should contain of original chunk text. "
            f"original {original_chunk.text}"
            f"expanded: {expanded_chunk.text}"
        )
    
  
    
    def test_expand_to_object_with_table(self):
        """Test expansion with table content."""
        doc = DoclingDocument(name="table_doc")
        doc.add_heading(text="Table Section", level=1)
        
        # Add a table
        table_data = TableData(num_cols=3)
        table_data.add_row(["Col1", "Col2", "Col3"])
        table_data.add_row(["A", "B", "C"])
        table_data.add_row(["D", "E", "F"])
        table_item = doc.add_table(data=table_data)
        
        # Create chunks
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=doc))
        serializer = chunker.serializer_provider.get_serializer(doc)
        
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
            expanded = table_chunk.expand_to_object(
                dl_doc=doc,
                serializer=serializer
            )
            
          
            
            # Verify that the serialized table text is in expanded text
            assert table_text in expanded.text, (
                f"Expanded chunk should contain the full serialized table text. "
                f"table text: {table_text}\n"
                f"expanded: {expanded.text}"
            )
    
    def test_expand_to_object_error_handling(self, sample_doc):
        """Test error handling in expand_to_object when serialization fails."""
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
        result = original_chunk.expand_to_object(
            dl_doc=sample_doc,
            serializer=FailingSerializer()
        )
        
        # Should return original chunk when serialization fails
        assert result == original_chunk, "Should return original chunk when serialization fails"
        assert result.text == "original text", "Original text should be preserved"
    
    def test_expand_to_object_preserves_metadata(self, sample_doc, chunking_serializer):
        """Test that expansion preserves chunk metadata."""
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc))
        if len(chunks) > 0:
            original = chunks[0]
            expanded = original.expand_to_object(
                dl_doc=sample_doc,
                serializer=chunking_serializer
            )
            
          
            assert expanded.meta.origin == original.meta.origin, (
                "Origin should be preserved"
            )


class TestExpandToPage:
    """Tests for expand_to_page method."""
    
    def test_expand_to_page_basic(self, sample_doc_with_pages):
        """Test basic page expansion."""
        # Create chunks
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc_with_pages))
        serializer = chunker.serializer_provider.get_serializer(sample_doc_with_pages)
        
        if len(chunks) > 0:
            chunk = chunks[0]
            expanded = chunk.expand_to_page(
                doc=sample_doc_with_pages,
                serializer=serializer
            )
            
            assert expanded is not None, "Should return expanded chunk"
            assert isinstance(expanded, DocChunk), "Should return DocChunk"
    
    def test_expand_to_page_includes_page_content(self, sample_doc_with_pages):
        """Test that page expansion includes all page content."""
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc_with_pages))
        serializer = chunker.serializer_provider.get_serializer(sample_doc_with_pages)
        
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
    
        
    def test_expand_to_page_preserves_metadata(self, sample_doc_with_pages):
        """Test that page expansion preserves metadata."""
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc_with_pages))
        serializer = chunker.serializer_provider.get_serializer(sample_doc_with_pages)
        
        if len(chunks) > 0:
            original = DocChunk.model_validate(chunks[0])
            expanded = original.expand_to_page(
                doc=sample_doc_with_pages,
                serializer=serializer
            )
            
            # Metadata should be preserved
            assert expanded.meta == original.meta, "Metadata should be preserved"


class TestExpandToObjectWithRealDocument:
    """Tests using real document from test data."""
    
    def test_expand_with_real_document(self):
        """Test expansion methods with real document data."""
        with open(INPUT_FILE, encoding="utf-8") as f:
            data_json = f.read()
        dl_doc = DoclingDocument.model_validate_json(data_json)
        
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=dl_doc))
        serializer = chunker.serializer_provider.get_serializer(dl_doc)
        
        assert len(chunks) > 0, "Should have chunks from real document"
        
        # Test expand_to_object on first chunk
        chunk = DocChunk.model_validate(chunks[0])
        expanded_obj = chunk.expand_to_object(
            dl_doc=dl_doc,
            serializer=serializer
        )
        
        assert expanded_obj is not None, "Should expand successfully"
        assert len(expanded_obj.text) > 0, "Expanded chunk should have text"
        
        # Test expand_to_page if document has pages
        if len(dl_doc.pages) > 0:
            expanded_page = chunk.expand_to_page(
                doc=dl_doc,
                serializer=serializer
            )
            
            assert expanded_page is not None, "Should expand to page successfully"
    
    def test_expand_all_chunks(self):
        """Test expanding all chunks from a document."""
        with open(INPUT_FILE, encoding="utf-8") as f:
            data_json = f.read()
        dl_doc = DoclingDocument.model_validate_json(data_json)
        
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=dl_doc))
        serializer = chunker.serializer_provider.get_serializer(dl_doc)
        
        # Expand all chunks to objects
        expanded_chunks = []
        for c in chunks:
            chunk = DocChunk.model_validate(c)
            expanded = chunk.expand_to_object(
                dl_doc=dl_doc,
                serializer=serializer
            )
            expanded_chunks.append(expanded)
        
        assert len(expanded_chunks) == len(chunks), (
            "Should have same number of expanded chunks"
        )
        
        # All expanded chunks should have content
        for expanded in expanded_chunks:
            assert len(expanded.text.strip()) > 0, (
                "Each expanded chunk should have text"
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
       
    def test_expand_with_none_serializer(self, sample_doc):
        """Test expansion with None serializer."""
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=EMBED_MODEL_ID,
                max_tokens=MAX_TOKENS,
            ),
        )
        
        chunks = list(chunker.chunk(dl_doc=sample_doc))
        if len(chunks) > 0:
            # Convert to DocChunk to access expansion methods
            chunk = DocChunk.model_validate(chunks[0])
            # Should handle None serializer gracefully by returning original chunk
            # (errors are caught and logged, not raised)
            result = chunk.expand_to_object(
                dl_doc=sample_doc,
                serializer=None
            )
            # Should return original chunk when serializer fails
            assert result == chunk, "Should return original chunk when serializer is None"
    
    

