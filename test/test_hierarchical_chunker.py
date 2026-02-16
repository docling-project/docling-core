import json
from pathlib import Path

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
)
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import DoclingDocument as DLDocument
from docling_core.types.doc.document import DoclingDocument, PictureItem, TextItem
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA


def _process(act_data, exp_path_str):
    if GEN_TEST_DATA:
        with open(exp_path_str, mode="w", encoding="utf-8") as f:
            json.dump(act_data, fp=f, indent=4)
            f.write("\n")
    else:
        with open(exp_path_str, encoding="utf-8") as f:
            exp_data = json.load(fp=f)
        assert exp_data == act_data


def test_chunk():
    with open("test/data/chunker/0_inp_dl_doc.json", encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(
        merge_list_items=True,
    )
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str="test/data/chunker/0_out_chunks.json",
    )


def test_chunk_custom_serializer():
    with open("test/data/chunker/0_inp_dl_doc.json", encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

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
    _process(
        act_data=act_data,
        exp_path_str="test/data/chunker/0b_out_chunks.json",
    )



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

    assert has_non_caption_text_in_picture, (
        "Test document should have non-caption TextItem children of PictureItem"
    )

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

    assert caption_count_default > 0, (
        "Caption TextItems should be included even with traverse_pictures=False"
    )

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
