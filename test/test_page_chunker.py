from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.types.doc.document import DoclingDocument

from .test_utils import assert_or_generate_json_ground_truth


def test_page_chunks():
    src = Path("./test/data/doc/cross_page_lists.json")
    doc = DoclingDocument.load_from_json(src)

    chunker = PageChunker()

    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)
    act_data = dict(root=[DocChunk.model_validate(n).export_json_dict() for n in chunks])
    assert_or_generate_json_ground_truth(
        act_data,
        src.parent / f"{src.stem}_chunks.json",
    )
