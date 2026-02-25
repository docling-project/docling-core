import json
from dataclasses import dataclass

import pytest
import tiktoken
from transformers import AutoTokenizer

from docling_core.transforms.chunker.base import BaseChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, MarkdownParams
from docling_core.types.doc import DoclingDocument as DLDocument
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
INPUT_FILE = "test/data/chunker/2_inp_dl_doc.json"

INNER_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)


def _process(act_data, exp_path_str):
    if GEN_TEST_DATA:
        with open(exp_path_str, mode="w", encoding="utf-8") as f:
            json.dump(act_data, fp=f, indent=4)
            f.write("\n")
    else:
        with open(exp_path_str, encoding="utf-8") as f:
            exp_data = json.load(fp=f)
        assert exp_data == act_data


def test_chunk_merge_peers():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_with_model_name():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(
            model_name=EMBED_MODEL_ID,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_deprecated_max_tokens():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    with pytest.warns(DeprecationWarning, match="Deprecated initialization"):
        chunker = HybridChunker(
            max_tokens=MAX_TOKENS,
        )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_contextualize():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_ser_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
    )

    chunks = chunker.chunk(dl_doc=dl_doc)

    act_data = dict(
        root=[
            dict(
                text=chunk.text,
                ser_text=(ser_text := chunker.contextualize(chunk)),
                num_tokens=chunker.tokenizer.count_tokens(ser_text),
            )
            for chunk in chunks
        ]
    )
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_no_merge_peers():
    EXPECTED_OUT_FILE = "test/data/chunker/2b_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=False,
    )

    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_deprecated_explicit_hf_obj():
    EXPECTED_OUT_FILE = "test/data/chunker/2c_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    with pytest.warns(DeprecationWarning, match="Deprecated initialization"):
        chunker = HybridChunker(
            tokenizer=INNER_TOKENIZER,
        )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_ignore_deprecated_param_if_new_tokenizer_passed():
    EXPECTED_OUT_FILE = "test/data/chunker/2c_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
        ),
        max_tokens=MAX_TOKENS,
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_deprecated_no_max_tokens():
    EXPECTED_OUT_FILE = "test/data/chunker/2c_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
        ),
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_contextualize_altered_delim():
    EXPECTED_OUT_FILE = "test/data/chunker/2d_out_ser_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
        delim="####",
    )

    chunks = chunker.chunk(dl_doc=dl_doc)

    act_data = dict(
        root=[
            dict(
                text=chunk.text,
                ser_text=(ser_text := chunker.contextualize(chunk)),
                num_tokens=chunker.tokenizer.count_tokens(ser_text),
            )
            for chunk in chunks
        ]
    )
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_custom_serializer():
    EXPECTED_OUT_FILE = "test/data/chunker/2e_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    class MySerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=MAX_TOKENS,
        ),
        merge_peers=True,
        serializer_provider=MySerializerProvider(),
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_openai():
    EXPECTED_OUT_FILE = "test/data/chunker/2f_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o"),
            max_tokens=128 * 1024,
        )
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_default():
    EXPECTED_OUT_FILE = "test/data/chunker/2g_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker()

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_chunk_explicit():
    EXPECTED_OUT_FILE = "test/data/chunker/2g_out_chunks.json"

    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer.from_pretrained(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=256,
        ),
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )


def test_shadowed_headings_wout_content():

    @dataclass
    class Setup:
        exp: str  # expected output file path
        chunker: BaseChunker

    setups = [
        Setup(
            exp="test/data/chunker/2h_out_chunks_hier_emit_false.json",
            chunker=HierarchicalChunker(always_emit_headings=False),
        ),
        Setup(
            exp="test/data/chunker/2h_out_chunks_hier_emit_true.json",
            chunker=HierarchicalChunker(always_emit_headings=True),
        ),
        Setup(
            exp="test/data/chunker/2h_out_chunks_hybr_emit_false.json",
            chunker=HybridChunker(always_emit_headings=False),
        ),
        Setup(
            exp="test/data/chunker/2h_out_chunks_hybr_emit_true.json",
            chunker=HybridChunker(always_emit_headings=True),
        ),
    ]

    # prepare document with different types of empty "sections" and headings shadowing each other
    doc = DoclingDocument(name="")
    doc.add_heading(text="Section 1", level=1)
    doc.add_heading(text="Section 1.1", level=2)
    doc.add_heading(text="Section 1.2", level=2)
    doc.add_heading(text="Section 2", level=1)
    doc.add_heading(text="Section 2.1", level=2)
    doc.add_heading(text="Section 2.1.1", level=3)
    doc.add_heading(text="Section 3", level=1)
    doc.add_heading(text="Section 3.1", level=2)
    doc.add_text(text="Foo", label=DocItemLabel.TEXT)
    doc.add_heading(text="Section 4", level=1)
    doc.add_heading(text="Section 4.1", level=2)

    for setup in setups:
        chunker = setup.chunker
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)
        act_data = dict(
            root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
        )
        _process(
            act_data=act_data,
            exp_path_str=setup.exp,
        )

def test_chunk_with_duplicate_table_header():
    """Test that table headers are repeated when a table is split across chunks."""
    INPUT_FILE = "test/data/chunker/0_inp_dl_doc.json"
    EXPECTED_OUT_FILE = "test/data/chunker/0c_out_chunks.json"
    
    with open(INPUT_FILE, encoding="utf-8") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    
    # Verify the document has tables
    assert len(dl_doc.tables) > 0, "Input file should contain at least one table"
    
    class MarkdownSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params = MarkdownParams(compact_tables=True),  # Use compact table format to reduce token count
            )
            
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=INNER_TOKENIZER,
            max_tokens=250,
        ),
        merge_peers=True,
        duplicate_table_header=True,
        serializer_provider=MarkdownSerializerProvider(),
    )
    # Create table serializer to serialize individual tables
    serializer = chunker.serializer_provider.get_serializer(dl_doc)
    
    # Serialize each table item individually to get expected content
    table_contents = {}
    for table_item in dl_doc.tables:
        # Serialize the table
        ser_result = serializer.serialize(
            item=table_item,
            )
        table_contents[table_item.self_ref] = ser_result.text
    

    chunks = list(chunker.chunk(dl_doc=dl_doc))
 #   for chunk in chunks:
 #       print(chunk, file=open("output.txt", "a", encoding="utf-8"))
 #       print("+"*50, file=open("output.txt", "a", encoding="utf-8"))
    
    # Verify we got chunks
    assert len(chunks) > 0, "Expected at least one chunk from the input document"
    
    # For each table, verify its content appears in chunks
    for table_ref, table_text in table_contents.items():
        # Get header and body lines from the serialized table
        if table_text:
            header_lines, body_lines = serializer.table_serializer.get_header_and_body_lines(
                table_text=table_text
            )
        
            # Find all chunks that contain content from this table
            chunks_with_table = [chunk for chunk in chunks if table_ref in [i.self_ref for i in chunk.meta.doc_items]]
                       
            # Verify table content appears in at least one chunk
            assert len(chunks_with_table) > 0, f"Table {table_ref} content should appear in at least one chunk"
        
            # If table is split across multiple chunks, verify header is repeated
            if len(chunks_with_table) > 1:
                # Each chunk with table content should have the header
                for chunk in chunks_with_table:
                    # Check if header lines are present
                    has_header = all(
                        header_line.strip() in chunk.text
                        for header_line in header_lines
                    )
                    assert has_header, (
                        f"Table {table_ref} split across chunks should have header repeated in each chunk. "
                        f"Missing header in chunk: {chunk.text[:200]}..."
                    )
                            
            # Verify all body lines appear somewhere in the chunks
            all_chunk_text = "\n".join(chunk.text for chunk in chunks_with_table)
            for body_line in body_lines:
                assert body_line.strip() in all_chunk_text, (
                    f"Table {table_ref} body line '{body_line.strip()}' should appear in chunks"
                )
    
    # Save chunks to output file for inspection
    chunks_data = [
        {
            "text": chunk.text,
            "meta": {
                "doc_items": [item.self_ref for item in chunk.meta.doc_items] if chunk.meta.doc_items else [],
                "headings": chunk.meta.headings,
            }
        }
        for chunk in chunks
    ]
    
    _process(chunks_data, EXPECTED_OUT_FILE)