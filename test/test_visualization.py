from pathlib import Path

import PIL.Image

from docling_core.transforms.visualizer.table_visualizer import TableVisualizer
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


def test_table_visualization_for_cells():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    visualizer = TableVisualizer()
    viz_pages = visualizer.get_visualization(doc=doc)

    verify(
        exp_file=VIZ_TEST_DATA_PATH / f"{src.stem}_table_viz_wout_lbl_p5.png",
        actual=viz_pages[5],
    )


def test_table_visualization_for_rows_and_cols():
    src = Path("./test/data/doc/2408.09869v3_enriched.json")
    doc = DoclingDocument.load_from_json(src)

    visualizer = TableVisualizer(
        params=TableVisualizer.Params(show_cells=False, show_rows=True, show_cols=True)
    )
    viz_pages = visualizer.get_visualization(doc=doc)

    verify(
        exp_file=VIZ_TEST_DATA_PATH
        / f"{src.stem}_table_viz_wout_lbl_p5_rows_and_cols.png",
        actual=viz_pages[5],
    )


def test_cross_page_lists_with_branch_nums():
    src = Path("./test/data/doc/cross_page_lists.json")
    doc = DoclingDocument.load_from_json(src)

    viz_pages = doc.get_visualization(show_branch_numbering=True)

    for i in range(2):
        verify(
            exp_file=VIZ_TEST_DATA_PATH / f"{src.stem}_p{i+1}.png",
            actual=viz_pages[i + 1],
        )
