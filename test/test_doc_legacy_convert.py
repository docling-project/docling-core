import yaml

from docling_core.types.doc import DoclingDocument
from docling_core.utils.legacy import docling_document_to_legacy


def test_new_to_old():
    filename = "test/data/doc/2206.01062.yaml"

    with open(filename, "r") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    docling_document_to_legacy(doc)
