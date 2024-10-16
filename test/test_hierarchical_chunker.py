#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc import DoclingDocument as DLDocument


def test_chunk_merge_list_items():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(
        merge_list_items=True,
    )
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    with open("test/data/chunker/0_out_chunks.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_no_merge_list_items():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(
        merge_list_items=False,
    )
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    with open("test/data/chunker/1_out_chunks.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data
