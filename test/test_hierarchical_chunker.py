from pathlib import Path

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
    TripletTableSerializer,
)
from docling_core.transforms.serializer.html import HTMLDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TableData, TextItem

from .test_utils import assert_or_generate_json_ground_truth, build_single_cell_rich_table_doc


def test_chunk():
    with open("test/data/chunker/0_inp_dl_doc.json", encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(
        merge_list_items=True,
    )
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(act_data, "test/data/chunker/0_out_chunks.json")


def test_chunk_custom_serializer():
    with open("test/data/chunker/0_inp_dl_doc.json", encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DoclingDocument.model_validate_json(data_json)

    class MySerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    chunker = HierarchicalChunker(
        merge_list_items=True,
        serializer_provider=MySerializerProvider(),
    )

    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(act_data, "test/data/chunker/0b_out_chunks.json")


def test_traverse_pictures():
    """Test that traverse_pictures parameter works correctly with HierarchicalChunker."""
    # Load a document that has TextItem children of PictureItem
    INPUT_FILE = "test/data/doc/concatenated.json"
    dl_doc = DoclingDocument.load_from_json(Path(INPUT_FILE))

    # Verify the document has non-caption TextItem children of PictureItem
    has_non_caption_text_in_picture = False
    for item, _ in dl_doc.iterate_items(with_groups=True, traverse_pictures=True):
        if isinstance(item, TextItem) and item.parent:
            parent = item.parent.resolve(dl_doc)
            if isinstance(parent, PictureItem) and item.label != DocItemLabel.CAPTION:
                has_non_caption_text_in_picture = True
                break

    assert has_non_caption_text_in_picture, "Test document should have non-caption TextItem children of PictureItem"

    # Test 1: Default behavior (traverse_pictures=False)
    # Only caption TextItems should be included
    chunker_default = HierarchicalChunker()
    chunks_default = list(chunker_default.chunk(dl_doc=dl_doc))

    caption_count_default = 0
    non_caption_count_default = 0
    for chunk in chunks_default:
        for doc_item in chunk.meta.doc_items:
            if isinstance(doc_item, TextItem) and doc_item.parent:
                parent = doc_item.parent.resolve(dl_doc)
                if isinstance(parent, PictureItem):
                    if doc_item.label == DocItemLabel.CAPTION:
                        caption_count_default += 1
                    else:
                        non_caption_count_default += 1

    # Test 2: With traverse_pictures=True
    # Both caption and non-caption TextItems should be included
    class TraversePicturesSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            params = MarkdownParams(traverse_pictures=True)
            return ChunkingDocSerializer(doc=doc, params=params)

    chunker_traverse = HierarchicalChunker(
        serializer_provider=TraversePicturesSerializerProvider(),
    )
    chunks_traverse = list(chunker_traverse.chunk(dl_doc=dl_doc))

    caption_count_traverse = 0
    non_caption_count_traverse = 0
    for chunk in chunks_traverse:
        for doc_item in chunk.meta.doc_items:
            if isinstance(doc_item, TextItem) and doc_item.parent:
                parent = doc_item.parent.resolve(dl_doc)
                if isinstance(parent, PictureItem):
                    if doc_item.label == DocItemLabel.CAPTION:
                        caption_count_traverse += 1
                    else:
                        non_caption_count_traverse += 1

    assert non_caption_count_default == 0, (
        f"With traverse_pictures=False (default), non-caption TextItems that are "
        f"children of PictureItems should NOT be included. Found {non_caption_count_default}"
    )

    assert caption_count_default > 0, "Caption TextItems should be included even with traverse_pictures=False"

    assert non_caption_count_traverse > 0, (
        f"With traverse_pictures=True, non-caption TextItems that are children of "
        f"PictureItems should be included. Found {non_caption_count_traverse}"
    )

    assert caption_count_traverse >= caption_count_default, (
        f"Caption count should not decrease with traverse_pictures=True. "
        f"Got {caption_count_traverse} vs {caption_count_default}"
    )

    total_items_default = sum(len(chunk.meta.doc_items) for chunk in chunks_default)
    total_items_traverse = sum(len(chunk.meta.doc_items) for chunk in chunks_traverse)
    assert total_items_traverse > total_items_default, (
        f"With traverse_pictures=True, more doc_items should be included in chunks. "
        f"Got {total_items_traverse} vs {total_items_default}"
    )


def test_triplet_table_serializer_single_column():
    """Test TripletTableSerializer with a single-column table."""
    # Create a document with a single-column table
    doc = DoclingDocument(name="test_single_column")
    table_data = TableData(num_cols=1)
    table_data.add_row(["Country"])  # Header row
    table_data.add_row(["Italy"])
    table_data.add_row(["Canada"])
    table_data.add_row(["Switzerland"])
    doc.add_table(data=table_data)

    serializer = ChunkingDocSerializer(doc=doc)
    table_serializer = TripletTableSerializer()
    table_item = next(iter(doc.iterate_items()))[0]

    result = table_serializer.serialize(
        item=table_item,
        doc_serializer=serializer,
        doc=doc,
    )

    expected = "Country = Italy. Country = Canada. Country = Switzerland"
    assert result.text == expected, f"Expected '{expected}', got '{result.text}'"


def test_triplet_table_serializer_handles_empty_table_output():
    """TripletTableSerializer must produce text and isolate `visited` for tables
    whose triplet form is empty.

    Covers:
    - header-only multi-column table emits the header row as text
    - single-row single-column header-only table emits its cell text
    - HierarchicalChunker emits chunks for a doc with a header-only table
    - single-cell RichTableCell table falls back to the referenced text
    - empty-text table does not consume its child refs via the shared visited set
    - HierarchicalChunker returns the referenced text as a chunk for rich-table layouts
    """
    header_only_doc = DoclingDocument(name="header_only")
    header_only_data = TableData(num_cols=3)
    header_only_data.add_row(["Name", "Age", "City"])
    for cell in header_only_data.table_cells:
        cell.column_header = True
    header_only_doc.add_table(data=header_only_data)

    serializer = ChunkingDocSerializer(doc=header_only_doc)
    table_serializer = TripletTableSerializer()
    header_only_table = next(iter(header_only_doc.iterate_items()))[0]
    header_only_result = table_serializer.serialize(
        item=header_only_table,
        doc_serializer=serializer,
        doc=header_only_doc,
    )
    assert header_only_result.text
    assert "Name" in header_only_result.text
    assert "Age" in header_only_result.text
    assert "City" in header_only_result.text
    assert "None" not in header_only_result.text

    single_col_doc = DoclingDocument(name="single_row_single_col")
    single_col_data = TableData(num_cols=1)
    single_col_data.add_row(["Total"])
    for cell in single_col_data.table_cells:
        cell.column_header = True
    single_col_doc.add_table(data=single_col_data)
    single_col_serializer = ChunkingDocSerializer(doc=single_col_doc)
    single_col_table = next(iter(single_col_doc.iterate_items()))[0]
    single_col_result = table_serializer.serialize(
        item=single_col_table,
        doc_serializer=single_col_serializer,
        doc=single_col_doc,
    )
    assert single_col_result.text == "Total"

    chunker_doc = DoclingDocument(name="chunker_header_only")
    chunker_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Introduction paragraph.")
    chunker_table_data = TableData(num_cols=2)
    chunker_table_data.add_row(["Field", "Value"])
    for cell in chunker_table_data.table_cells:
        cell.column_header = True
    chunker_doc.add_table(data=chunker_table_data)
    chunker_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Conclusion paragraph.")
    chunks = list(HierarchicalChunker().chunk(dl_doc=chunker_doc))
    assert len(chunks) > 0
    all_text = " ".join(c.text for c in chunks)
    assert "Introduction" in all_text
    assert "Conclusion" in all_text

    rich_doc = build_single_cell_rich_table_doc("Important body text inside layout table")
    rich_serializer = ChunkingSerializerProvider().get_serializer(rich_doc)
    visited: set[str] = set()
    rich_result = rich_serializer.serialize(item=rich_doc.tables[0], visited=visited)
    assert rich_result.text == "Important body text inside layout table"
    assert visited == {
        rich_doc.tables[0].self_ref,
        rich_doc.groups[0].self_ref,
        rich_doc.texts[0].self_ref,
    }

    empty_rich_doc = build_single_cell_rich_table_doc("")
    empty_rich_serializer = ChunkingSerializerProvider().get_serializer(empty_rich_doc)
    empty_visited: set[str] = set()
    empty_rich_result = empty_rich_serializer.serialize(item=empty_rich_doc.tables[0], visited=empty_visited)
    assert empty_rich_result.text == ""
    assert empty_visited == {empty_rich_doc.tables[0].self_ref}

    rich_chunks = list(HierarchicalChunker().chunk(dl_doc=rich_doc))
    assert len(rich_chunks) == 1
    assert rich_chunks[0].text == "Important body text inside layout table"
    assert [item.self_ref for item in rich_chunks[0].meta.doc_items] == [rich_doc.tables[0].self_ref]


def test_chunk_rich_table_custom_serializer(rich_table_doc: DoclingDocument):
    doc = rich_table_doc

    class MySerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return HTMLDocSerializer(
                doc=doc,
                table_serializer=TripletTableSerializer(),
            )

    chunker = HierarchicalChunker(
        merge_list_items=True,
        serializer_provider=MySerializerProvider(),
    )

    chunks = chunker.chunk(dl_doc=doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])

    assert_or_generate_json_ground_truth(act_data, "test/data/chunker/0c_out_chunks.json")
