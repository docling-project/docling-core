from pathlib import Path
from docling_core.types.doc.document import DoclingDocument
from test.test_serialization_doclang import _verify_doc
from test.test_serialization import verify


def test_flatten(mixed_hierarchy_doc: DoclingDocument):
    doc: DoclingDocument = mixed_hierarchy_doc
    doc._flatten()

    doc._normalize_references()

    exp_json = Path("./test/data/doc/flattened.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    exp_dclg = Path("./test/data/doc/flattened.dclg.xml")
    actual=doc.export_to_doclang()
    verify(actual=actual, exp_file=exp_dclg)

def test_hierarchize(mixed_hierarchy_doc):
    doc: DoclingDocument = mixed_hierarchy_doc
    doc._hierarchize()

    doc._normalize_references()

    exp_json = Path("./test/data/doc/hierarchized.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    exp_dclg = Path("./test/data/doc/hierarchized.dclg.xml")
    actual=doc.export_to_doclang()
    verify(actual=actual, exp_file=exp_dclg)
