#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json

from transformers import AutoTokenizer

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument as DLDocument

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
INPUT_FILE = "test/data/chunker/2_inp_dl_doc.json"

TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)


def test_chunk_merge_peers():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE) as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=TOKENIZER,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    with open(EXPECTED_OUT_FILE) as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_no_merge_peers():
    EXPECTED_OUT_FILE = "test/data/chunker/2b_out_chunks.json"

    with open(INPUT_FILE) as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=TOKENIZER,
        max_tokens=MAX_TOKENS,
        merge_peers=False,
    )

    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    with open(EXPECTED_OUT_FILE) as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_serialize():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_ser_chunks.json"

    with open(INPUT_FILE) as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=TOKENIZER,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    chunks = chunker.chunk(dl_doc=dl_doc)

    act_data = dict(
        root=[
            dict(
                text=chunk.text,
                ser_text=(ser_text := chunker.serialize(chunk)),
                num_tokens=len(TOKENIZER.tokenize(ser_text, max_length=None)),
            )
            for chunk in chunks
        ]
    )
    with open(EXPECTED_OUT_FILE) as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_with_model_name():
    EXPECTED_OUT_FILE = "test/data/chunker/2a_out_chunks.json"

    with open(INPUT_FILE) as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)

    chunker = HybridChunker(
        tokenizer=EMBED_MODEL_ID,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    with open(EXPECTED_OUT_FILE) as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data
