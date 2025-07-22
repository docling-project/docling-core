import json
from pathlib import Path

from PIL import Image as PILImage

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument, PictureTabularChartData

from .test_data_gen_flag import GEN_TEST_DATA


def verify(exp_file: Path, actual: dict):
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(actual, f, indent=2)
            f.write("\n")
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = json.load(f)

        # we removed image URIs in both dicts for lossy comparison
        # as the test was flaky due to URIs
        def strip_image_uris(d):
            if isinstance(d, dict):
                return {
                    k: strip_image_uris(v)
                    for k, v in d.items()
                    if k not in {"uri", "image_uri"}
                }
            elif isinstance(d, list):
                return [strip_image_uris(x) for x in d]
            else:
                return d

        expected_stripped = strip_image_uris(expected)
        actual_stripped = strip_image_uris(actual)
        assert (
            expected_stripped == actual_stripped
        ), "Dicts differ (ignoring image URIs)"

        if "data:image/png;base64" in str(expected):
            # check if the image URIs are the same
            assert "data:image/png;base64" in str(actual), "Image URIs does not exist"


def test_doctags_load_from_files():

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("test/data/doc/page_with_pic.dt")],
        [Path("test/data/doc/page_with_pic.png")],
    )

    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "test/data/doc/page_with_pic_from_files.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_load_from_memory():

    with Path("test/data/doc/page_with_pic.dt").open() as file:
        doctags = file.read()
    image = PILImage.open(Path("test/data/doc/page_with_pic.png"))

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])

    doc = DoclingDocument.load_from_doctags(doctags_doc)

    exp = "test/data/doc/page_with_pic.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_load_without_image():
    with Path("test/data/doc/page_with_pic.dt").open() as file:
        doctags = file.read()
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], None)
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "test/data/doc/page_without_pic.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_load_for_kv_region():
    with Path("test/data/doc/doc_with_kv.dt").open() as file:
        doctags = file.read()
    image = PILImage.open(Path("test/data/doc/doc_with_kv.png"))
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "test/data/doc/doc_with_kv.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_multipage_doctags_load():
    with Path("test/data/doc/2206.01062.yaml.dt").open() as file:
        doctags = file.read()
    doctags_doc = DocTagsDocument.from_multipage_doctags_and_images(doctags, None)
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = "test/data/doc/2206.01062.yaml.dt.json"
    verify(
        exp_file=exp,
        actual=doc.export_to_dict(),
    )


def test_doctags_chart():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("test/data/doc/barchart.dt")],
        [Path("test/data/doc/barchart.png")],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    for pic in doc.pictures:
        tabular_chart_annotations = [
            ann for ann in pic.annotations if isinstance(ann, PictureTabularChartData)
        ]
        assert len(tabular_chart_annotations) > 0


def test_doctags_table_provenances_and_captions():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("test/data/doc/01030000000083.dt")],
        [Path("test/data/doc/01030000000083.png")],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    for table in doc.tables:
        assert len(table.prov) > 0
        assert len(table.captions) > 0


def test_doctags_picture_provenances_and_captions():
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("test/data/doc/01030000000111.dt")],
        [Path("test/data/doc/01030000000111.png")],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc)
    for picture in doc.pictures:
        assert len(picture.prov) > 0
        assert len(picture.captions) > 0


def test_doctags_inline():
    src_path = Path("test/data/doc/2408.09869v3_enriched.dt")
    with open(src_path) as f:
        doctags = f.read()
    doc = DoclingDocument.load_from_json("test/data/doc/2408.09869v3_enriched.json")

    doctags_doc = DocTagsDocument.from_multipage_doctags_and_images(
        doctags=doctags,
        images=[
            pil_img
            for p in doc.pages
            if (img_ref := doc.pages[p].image) and (pil_img := img_ref.pil_image)
        ],
    )

    deser_doc = DoclingDocument.load_from_doctags(doctags_doc)
    exp = f"{src_path.parent / src_path.stem}.out.dt.json"
    verify(
        exp_file=exp,
        actual=deser_doc.export_to_dict(),
    )
