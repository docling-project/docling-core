from pathlib import Path

import PIL.Image

from docling_core.types.doc.document import DoclingDocument

from .test_data_gen_flag import GEN_TEST_DATA

VIZ_TEST_DATA_PATH = Path("./test/data/viz")


def verify(exp_file: Path, actual: PIL.Image.Image):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            actual.save(exp_file)
    else:
        with PIL.Image.open(exp_file) as expected:
            assert actual == expected


def test_doc_visualization():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)
    viz_pages = doc.get_visualization()
    for k in viz_pages:
        if k <= 3:
            verify(
                exp_file=VIZ_TEST_DATA_PATH / f"{src.stem}_viz_p{k}.png",
                actual=viz_pages[k],
            )


def test_doc_visualization_inline_circumscribed_bbox():
    src = Path("./test/data/doc/2408.09869v3_enriched.dt.json")
    doc = DoclingDocument.load_from_json(src)
    viz_pages = doc.get_visualization()
    for k in viz_pages:
        if k == 2:
            verify(
                exp_file=VIZ_TEST_DATA_PATH / f"{src.stem}_viz_p{k}.png",
                actual=viz_pages[k],
            )


def test_doc_visualization_no_label():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)
    viz_pages = doc.get_visualization(show_label=False)
    for k in viz_pages:
        if k <= 3:
            verify(
                exp_file=VIZ_TEST_DATA_PATH / f"{src.stem}_viz_wout_lbl_p{k}.png",
                actual=viz_pages[k],
            )
