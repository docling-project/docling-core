#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types import Document as DLDocument


def test_chunk_without_metadata():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(include_metadata=False)
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[n.model_dump() for n in chunks])
    with open("test/data/chunker/0_out_chunks_wout_meta.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_with_metadata():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(include_metadata=True)
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[n.model_dump() for n in chunks])
    with open("test/data/chunker/0_out_chunks_with_meta.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data
