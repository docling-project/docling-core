from pathlib import Path

import yaml

from docling_core.types.doc import DoclingDocument
from docling_core.types.legacy_doc.document import ExportedCCSDocument as DsDocument
from docling_core.utils.legacy import (
    docling_document_to_legacy,
    legacy_to_docling_document,
)

from .test_data_gen_flag import GEN_TEST_DATA


def test_new_to_old():
    filename = "test/data/doc/2206.01062.yaml"

    with open(filename, "r", encoding="utf-8") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    docling_document_to_legacy(doc)


def test_old_to_new():
    filepath = Path("test/data/legacy_doc/doc-export.json")
    leg_doc = DsDocument.model_validate_json(filepath.read_text())

    doc = legacy_to_docling_document(leg_doc)

    gt_filepath = Path(filepath.with_suffix(".docling.yaml.gt"))
    if GEN_TEST_DATA:
        doc.save_as_yaml(gt_filepath)
    else:
        with gt_filepath.open() as gt_fp:
            gt_dict = yaml.safe_load(gt_fp)
            gt_doc = DoclingDocument.model_validate(gt_dict)

        assert doc == gt_doc
