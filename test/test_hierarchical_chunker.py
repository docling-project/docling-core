#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DoclingDocument as DLDocument
from docling_core.types.doc.document import DoclingDocument

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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
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
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    _process(
        act_data=act_data,
        exp_path_str="test/data/chunker/0b_out_chunks.json",
    )
