import json
from pathlib import Path

from PIL import Image as PILImage

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from docling_core.types.doc.utils import parse_otsl_table_content

from .test_data_gen_flag import GEN_TEST_DATA


def verify(exp_file: Path, actual: dict):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(actual, f, indent=2)
            f.write("\n")
    else:
        with open(exp_file, encoding="utf-8") as f:
            expected = json.load(f)

        # we removed image URIs in both dicts for lossy comparison
        # as the test was flaky due to URIs
        def strip_image_uris(d):
            if isinstance(d, dict):
                return {k: strip_image_uris(v) for k, v in d.items() if k not in {"uri", "image_uri"}}
            elif isinstance(d, list):
                return [strip_image_uris(x) for x in d]
            else:
                return d

        expected_stripped = strip_image_uris(expected)
        actual_stripped = strip_image_uris(actual)
        assert expected_stripped == actual_stripped, "Dicts differ (ignoring image URIs)"

        if "data:image/png;base64" in str(expected):
            # check if the image URIs are the same
            assert "data:image/png;base64" in str(actual), "Image URIs does not exist"


def test_doctags_load_from_files():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("tests/data/doc/page_with_pic.dt")],
        [Path("tests/data/doc/page_with_pic.png")],
    )

    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "tests/data/doc/page_with_pic_from_files.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_load_from_memory():
    with Path("tests/data/doc/page_with_pic.dt").open() as file:
        doctags = file.read()
    image = PILImage.open(Path("tests/data/doc/page_with_pic.png"))

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])

    doc = DoclingDocument.load_from_doctags(doctags_doc)

    exp = "tests/data/doc/page_with_pic.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_load_without_image():
    with Path("tests/data/doc/page_with_pic.dt").open() as file:
        doctags = file.read()
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], None)
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "tests/data/doc/page_without_pic.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_load_for_kv_region():
    with Path("tests/data/doc/doc_with_kv.dt").open() as file:
        doctags = file.read()
    image = PILImage.open(Path("tests/data/doc/doc_with_kv.png"))
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "tests/data/doc/doc_with_kv.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_multipage_doctags_load():
    with Path("tests/data/doc/2206.01062.yaml.dt").open() as file:
        doctags = file.read()
    doctags_doc = DocTagsDocument.from_multipage_doctags_and_images(doctags, None)
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "tests/data/doc/2206.01062.yaml.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_chart():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("tests/data/doc/barchart.dt")],
        [Path("tests/data/doc/barchart.png")],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "tests/data/doc/barchart.dt.out.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_table_provenances_and_captions():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("tests/data/doc/01030000000083.dt")],
        [Path("tests/data/doc/01030000000083.png")],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    for table in doc.tables:
        assert len(table.prov) > 0
        assert len(table.captions) > 0


def test_doctags_picture_provenances_and_captions():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("tests/data/doc/01030000000111.dt")],
        [Path("tests/data/doc/01030000000111.png")],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    for picture in doc.pictures:
        assert len(picture.prov) > 0
        assert len(picture.captions) > 0


def test_doctags_load_preserves_angle_brackets_in_text():
    # Regression for #618: text nodes containing a "<" followed by a later ">"
    # (e.g. statistical notation like "P < 0.05 ... P > 0.05") had the whole
    # span between the two characters silently deleted by extract_inner_text.
    original = "We found (r = -0.36, P < 0.05), but not (r = -0.08, P > 0.05), lending support."
    doctags = f"<doctag><text><loc_10><loc_10><loc_400><loc_20>{original}</text></doctag>"

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], None)
    doc = DoclingDocument.load_from_doctags(doctags_doc)

    assert [t.text for t in doc.texts] == [original]


def test_doctags_inline():
    src_path = Path("tests/data/doc/2408.09869v3_enriched.dt")
    with open(src_path) as f:
        doctags = f.read()
    doc = DoclingDocument.load_from_json("tests/data/doc/2408.09869v3_enriched.json")

    doctags_doc = DocTagsDocument.from_multipage_doctags_and_images(
        doctags=doctags,
        images=[pil_img for p in doc.pages if (img_ref := doc.pages[p].image) and (pil_img := img_ref.pil_image)],
    )
    deser_doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = f"{src_path.parent / src_path.stem}.out.dt.json"
    verify(
        exp_file=exp,
        actual=deser_doc.export_to_dict(),
    )


def _cells(otsl: str) -> list[tuple[str, int, int, int, int]]:
    td = parse_otsl_table_content(otsl)
    return [(c.text, c.start_row_offset_idx, c.start_col_offset_idx, c.row_span, c.col_span) for c in td.table_cells]


def test_parse_otsl_ucel_span_into_shorter_row():
    # Regression for docling-project/docling#2467: VLM output can have rows of
    # different length, and a vertical span that reached a shorter row raised
    # IndexError.
    cells = _cells("<otsl><fcel>a<fcel>b<nl><fcel>c<ucel><nl><fcel>d<nl></otsl>")

    assert cells == [
        ("a", 0, 0, 1, 1),
        ("b", 0, 1, 2, 1),
        ("c", 1, 0, 1, 1),
        ("d", 2, 0, 1, 1),
    ]


def test_parse_otsl_lcel_after_row_end():
    # An empty cell at the end of a row makes the look-ahead see the lcel of the
    # next row, and the horizontal span count read past the row end.
    cells = _cells("<otsl><ched><nl><lcel><fcel>x<nl></otsl>")

    assert ("x", 1, 1, 1, 1) in cells
