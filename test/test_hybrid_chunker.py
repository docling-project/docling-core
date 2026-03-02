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
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
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


def test_section_header_with_content():
    """Test that SECTION_HEADER items with substantial content are chunked.

    This reproduces a bug where HybridChunker skips SECTION_HEADER items that
    contain content, which is common in legal documents where the section number
    and content are combined (e.g., "## 1 This Agreement...").

    Expected: All content sections should be chunked, including those labeled
    as SECTION_HEADER.
    Actual (bug): SECTION_HEADER items are skipped even when they contain content.
    """
    doc = DoclingDocument(name="test_legal_doc")

    # Title
    doc.add_text(
        label=DocItemLabel.TITLE,
        text="MUTUAL NON-DISCLOSURE AGREEMENT",
    )

    # Section 1: SECTION_HEADER with content (should chunk - currently doesn't)
    doc.add_text(
        label=DocItemLabel.SECTION_HEADER,
        text="1 This Agreement is entered into by and between Company A and Company B. Each party agrees to the following terms and conditions. This section contains substantial content that should definitely be chunked.",
    )

    # Section 2: Header only (ok to skip)
    doc.add_text(
        label=DocItemLabel.SECTION_HEADER,
        text="2 Purpose",
    )

    # Section 3: SECTION_HEADER with content (should chunk - currently doesn't)
    doc.add_text(
        label=DocItemLabel.SECTION_HEADER,
        text="3 The Parties wish to explore a business opportunity. Discloser may disclose confidential information to Recipient. This is another substantial content section that should be chunked.",
    )

    # Section 4: Header only (ok to skip)
    doc.add_text(
        label=DocItemLabel.SECTION_HEADER,
        text="4 Confidential Information",
    )

    # Section 6: PARAGRAPH with content (this DOES get chunked)
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Recipient shall not use any Confidential Information for any purpose except to evaluate and engage in discussions concerning the Opportunity. Recipient shall not disclose information to third parties without consent.",
    )

    # Chunk the document
    chunker = HybridChunker()
    chunks = list(chunker.chunk(dl_doc=doc))

    # Define expected content sections (ones with substantial text)
    content_sections = {
        "Section 1": "This Agreement is entered into",
        "Section 3": "The Parties wish to explore",
        "Section 6": "Recipient shall not use",
    }

    # Check which sections are present in chunks
    missing_sections = []
    for section_name, text_snippet in content_sections.items():
        found = any(text_snippet in chunk.text for chunk in chunks)
        if not found:
            missing_sections.append(section_name)

    # Assert all content sections are chunked
    assert len(missing_sections) == 0, (
        f"HybridChunker failed to chunk sections with SECTION_HEADER label: {missing_sections}. "
        f"Created only {len(chunks)} chunks from {len(content_sections)} content sections. "
        "SECTION_HEADER items with substantial content should be chunked, not skipped."
    )
