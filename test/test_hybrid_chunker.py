#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json

import pytest
import tiktoken
from transformers import AutoTokenizer

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DoclingDocument as DLDocument
from docling_core.types.doc.document import DoclingDocument

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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    _process(
        act_data=act_data,
        exp_path_str=EXPECTED_OUT_FILE,
    )
