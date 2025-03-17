from pathlib import Path

from PIL import Image as PILImage

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument


def test_doctags_load_from_files():
    doc = DoclingDocument(name="Document")

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [Path("test/data/doc/page_with_pic.dt")],
        [Path("test/data/doc/page_with_pic.png")],
    )

    doc.load_from_doctags(doctags_doc)
    # print(doc.export_to_html())


def test_doctags_load_from_memory():
    doc = DoclingDocument(name="Document")

    doctags = Path("test/data/doc/page_with_pic.dt").open("r").read()
    image = PILImage.open(Path("test/data/doc/page_with_pic.png"))

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])

    doc.load_from_doctags(doctags_doc)
    # print(doc.export_to_html())


def test_doctags_load_for_kv_region():
    doc = DoclingDocument(name="Document")
    doctags = Path("test/data/doc/doc_with_kv.dt").open("r").read()
    image = PILImage.open(Path("test/data/doc/doc_with_kv.png"))
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc.load_from_doctags(doctags_doc)
    # print(doc.export_to_html())
