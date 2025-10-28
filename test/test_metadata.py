from pathlib import Path

import pytest
from pydantic import BaseModel

from docling_core.types.doc.document import (
    BaseMeta,
    DoclingDocument,
    NodeItem,
    RefItem,
    SummaryMetaField,
)
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA


class CustomCoordinates(BaseModel):
    longitude: float
    latitude: float


def test_metadata_usage():
    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)
    example_item: NodeItem = RefItem(cref="#/texts/2").resolve(doc=doc)
    assert example_item.meta is not None

    # add a custom metadata object to the item
    value = CustomCoordinates(longitude=47.3769, latitude=8.5417)
    target_name = example_item.meta.set_custom_field(
        namespace="my_corp", name="coords", value=value
    )
    assert target_name == "my_corp__coords"

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


def _create_doc_with_group_with_metadata() -> DoclingDocument:
    doc = DoclingDocument(name="")
    grp = doc.add_group()
    grp.meta = BaseMeta(
        summary=SummaryMetaField(text="This part talks about foo and bar.")
    )
    doc.add_text(text="Foo", label=DocItemLabel.TEXT)
    doc.add_text(text="Bar", label=DocItemLabel.TEXT)
    return doc


def test_group_with_metadata():
    doc = _create_doc_with_group_with_metadata()

    # test dumping to and loading from YAML
    exp_file = Path("test/data/doc/group_with_metadata.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(filename=exp_file)
    else:
        expected = DoclingDocument.load_from_yaml(filename=exp_file)
        assert doc == expected

    # test exporting to Markdown
    exp_file = exp_file.with_suffix(".md")
    if GEN_TEST_DATA:
        doc.save_as_markdown(filename=exp_file)
    else:
        actual = doc.export_to_markdown()
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_legacy_annotations():
    inp = Path("test/data/doc/dummy_doc.yaml")
    doc = DoclingDocument.load_from_yaml(filename=inp)
    exp_file = inp.parent / f"{inp.stem}_legacy_annotations.md"
    if GEN_TEST_DATA:
        doc.save_as_markdown(filename=exp_file, use_legacy_annotations=True)
    else:
        actual = doc.export_to_markdown(use_legacy_annotations=True)
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_mark_meta():
    inp = Path("test/data/doc/dummy_doc.yaml")
    doc = DoclingDocument.load_from_yaml(filename=inp)
    exp_file = inp.parent / f"{inp.stem}_mark_meta.md"
    if GEN_TEST_DATA:
        doc.save_as_markdown(filename=exp_file, mark_meta=True)
    else:
        actual = doc.export_to_markdown(mark_meta=True)
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected
