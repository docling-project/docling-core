import json
from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.page_chunker import PageChunker
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


def test_page_chunks():
    src = Path("./test/data/doc/cross_page_lists.json")
    doc = DoclingDocument.load_from_json(src)

    chunker = PageChunker()

    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)
    act_data = dict(
        root=[DocChunk.model_validate(n).export_json_dict() for n in chunks]
    )
    _process(
        act_data=act_data,
        exp_path_str=src.parent / f"{src.stem}_chunks.json",
    )
