from pathlib import Path

import pytest
from pydantic import BaseModel

from docling_core.types.doc.document import (
    DoclingDocument,
    NodeItem,
    RefItem,
    create_meta_field_name,
)

from .test_data_gen_flag import GEN_TEST_DATA


def test_metadata_usage():
    class CustomCoordinates(BaseModel):
        longitude: float
        latitude: float

    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)
    example_item: NodeItem = RefItem(cref="#/texts/2").resolve(doc=doc)
    assert example_item.meta is not None

    # add a custom metadata object to the item
    target_name = create_meta_field_name(namespace="my_corp", name="coords")
    value = CustomCoordinates(longitude=47.3769, latitude=8.5417)
    setattr(example_item.meta, target_name, value)

    # save the document
    exp_file = src.parent / f"{src.stem}_modified.yaml"
    if GEN_TEST_DATA:
        doc.save_as_yaml(filename=exp_file)
    else:
        expected = DoclingDocument.load_from_yaml(filename=exp_file)
        assert doc.model_dump(mode="json") == expected.model_dump(mode="json")

    # load back the document and read the custom metadata object
    loaded_doc = DoclingDocument.load_from_yaml(filename=exp_file)
    loaded_item: NodeItem = RefItem(cref="#/texts/2").resolve(doc=loaded_doc)
    assert loaded_item.meta is not None

    loaded_dict = loaded_item.meta.get_custom_part()[target_name]
    loaded_value = CustomCoordinates.model_validate(loaded_dict)

    # ensure the value is the same
    assert loaded_value == value


def test_namespace_absence_raises():
    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)
    example_item = RefItem(cref="#/texts/2").resolve(doc=doc)

    with pytest.raises(ValueError):
        example_item.meta.my_corp_programmaticaly_added_field = True
