#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types import Document as DLDocument


def test_chunk_heading_in_text_wout_extras():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(heading_as_metadata=False, include_metadata=False)
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[n.model_dump(exclude_none=True) for n in chunks])
    with open("test/data/chunker/0_out_chunks_heading_in_text_wout_extras.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_heading_in_text_with_extras():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(heading_as_metadata=False, include_metadata=True)
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[n.model_dump(exclude_none=True) for n in chunks])
    with open("test/data/chunker/0_out_chunks_heading_in_text_with_extras.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_heading_in_meta_wout_extras():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(heading_as_metadata=True, include_metadata=False)
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[n.model_dump(exclude_none=True) for n in chunks])
    with open("test/data/chunker/0_out_chunks_heading_in_meta_wout_extras.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_chunk_heading_in_meta_with_extras():
    with open("test/data/chunker/0_inp_dl_doc.json") as f:
        data_json = f.read()
    dl_doc = DLDocument.model_validate_json(data_json)
    chunker = HierarchicalChunker(heading_as_metadata=True, include_metadata=True)
    chunks = chunker.chunk(dl_doc=dl_doc)
    act_data = dict(root=[n.model_dump(exclude_none=True) for n in chunks])
    with open("test/data/chunker/0_out_chunks_heading_in_meta_with_extras.json") as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data
