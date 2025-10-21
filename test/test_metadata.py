from pathlib import Path

from docling_core.types.doc.document import DoclingDocument, RefItem

from .test_data_gen_flag import GEN_TEST_DATA


def test_metadata():
    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)
    example_item = RefItem(cref="#/texts/2").resolve(doc=doc)
    example_item.meta.example_custom_field_added_programmaticaly = True

    exp_file = src.parent / f"{src.stem}_modified.yaml"
    if GEN_TEST_DATA:
        doc.save_as_yaml(filename=exp_file)
    else:
        expected = DoclingDocument.load_from_yaml(filename=exp_file)
        assert doc == expected
