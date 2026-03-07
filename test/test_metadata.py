from pathlib import Path
from typing import Any, Optional

import pytest
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownMetaSerializer,
    MarkdownParams,
)
from docling_core.types.doc import (
    BaseMeta,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    MetaFieldName,
    MetaUtils,
    NodeItem,
    RefItem,
    SummaryMetaField,
)

from .test_data_gen_flag import GEN_TEST_DATA


class CustomCoordinates(BaseModel):
    longitude: float
    latitude: float


def test_metadata_usage() -> None:
    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)

    first_pic = doc.pictures[0]
    assert first_pic.meta
    assert first_pic.meta.classification
    assert first_pic.meta.classification.predictions
    assert first_pic.meta.classification.predictions[0].confidence == 0.78

    example_item: NodeItem = RefItem(cref="#/texts/2").resolve(doc=doc)
    assert example_item.meta is not None

    # add a custom metadata object to the item
    value = CustomCoordinates(longitude=47.3769, latitude=8.5417)
    target_name = example_item.meta.set_custom_field(namespace="my_corp", name="coords", value=value)
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


def test_metadata_relaxed_migration() -> None:
    src = Path("test/data/doc/dummy_doc_with_meta_2.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)

    first_pic = doc.pictures[0]
    assert first_pic.meta
    assert first_pic.meta.classification
    assert first_pic.meta.classification.predictions
    # check migration was skipped since respetive meta already present:
    assert first_pic.meta.classification.predictions[0].confidence == 0.42


def test_namespace_absence_raises():
    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    doc = DoclingDocument.load_from_yaml(filename=src)
    example_item = RefItem(cref="#/texts/2").resolve(doc=doc)

    with pytest.raises(ValueError):
        example_item.meta.my_corp_programmaticaly_added_field = True


def _create_doc_with_group_with_metadata() -> DoclingDocument:
    doc = DoclingDocument(name="")
    doc.body.meta = BaseMeta(summary=SummaryMetaField(text="This document talks about various topics."))
    grp1 = doc.add_group(name="1", label=GroupLabel.CHAPTER)
    grp1.meta = BaseMeta(summary=SummaryMetaField(text="This chapter discusses foo and bar."))
    doc.add_text(text="This is some introductory text.", label=DocItemLabel.TEXT, parent=grp1)

    grp1a = doc.add_group(parent=grp1, name="1a", label=GroupLabel.SECTION)
    grp1a.meta = BaseMeta(summary=SummaryMetaField(text="This section talks about foo."))
    grp1a.meta.set_custom_field(namespace="my_corp", name="test_1", value="custom field value 1")
    txt1 = doc.add_text(text="Regarding foo...", label=DocItemLabel.TEXT, parent=grp1a)
    txt1.meta = BaseMeta(summary=SummaryMetaField(text="This paragraph provides more details about foo."))
    lst1a = doc.add_list_group(parent=grp1a)
    lst1a.meta = BaseMeta(summary=SummaryMetaField(text="Here some foo specifics are listed."))
    doc.add_list_item(text="lorem", parent=lst1a, enumerated=True)
    doc.add_list_item(text="ipsum", parent=lst1a, enumerated=True)

    grp1b = doc.add_group(parent=grp1, name="1b", label=GroupLabel.SECTION)
    grp1b.meta = BaseMeta(summary=SummaryMetaField(text="This section talks about bar."))
    grp1b.meta.set_custom_field(namespace="my_corp", name="test_2", value="custom field value 2")
    doc.add_text(text="Regarding bar...", label=DocItemLabel.TEXT, parent=grp1b)

    return doc


def test_ser_deser():
    doc = _create_doc_with_group_with_metadata()

    # test dumping to and loading from YAML
    exp_file = Path("test/data/doc/group_with_metadata.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(filename=exp_file)
    else:
        expected = DoclingDocument.load_from_yaml(filename=exp_file)
        assert doc == expected


def test_md_ser_default():
    doc = _create_doc_with_group_with_metadata()

    # test exporting to Markdown
    params = MarkdownParams()
    ser = MarkdownDocSerializer(doc=doc, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_default.md")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(actual)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_md_ser_marked():
    doc = _create_doc_with_group_with_metadata()

    # test exporting to Markdown
    params = MarkdownParams(
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_marked.md")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(actual)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_md_ser_allowed_meta_names():
    doc = _create_doc_with_group_with_metadata()
    params = MarkdownParams(
        allowed_meta_names={
            MetaUtils.create_meta_field_name(namespace="my_corp", name="test_1"),
        },
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_allowed_meta_names.md")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(actual)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_md_ser_blocked_meta_names():
    doc = _create_doc_with_group_with_metadata()
    params = MarkdownParams(
        blocked_meta_names={
            MetaUtils.create_meta_field_name(namespace="my_corp", name="test_1"),
            MetaFieldName.SUMMARY.value,
        },
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_blocked_meta_names.md")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(actual)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_md_ser_without_non_meta():
    doc = _create_doc_with_group_with_metadata()
    params = MarkdownParams(
        include_non_meta=False,
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_without_non_meta.md")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(actual)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected


def test_ser_custom_meta_serializer():
    class SummaryMarkdownMetaSerializer(MarkdownMetaSerializer):
        @override
        def serialize(
            self,
            *,
            item: NodeItem,
            doc: DoclingDocument,
            level: Optional[int] = None,
            **kwargs: Any,
        ) -> SerializationResult:
            """Serialize the item's meta."""
            params = MarkdownParams(**kwargs)
            return create_ser_result(
                text="\n\n".join(
                    [
                        f"{'  ' * (level or 0)}[{item.self_ref}] [{item.__class__.__name__}:{item.label.value}] {tmp}"  # type:ignore[attr-defined]
                        for key in (list(item.meta.__class__.model_fields) + list(item.meta.get_custom_part()))
                        if (tmp := self._serialize_meta_field(item.meta, key, params.mark_meta))
                    ]
                    if item.meta
                    else []
                ),
                span_source=item if isinstance(item, DocItem) else [],
            )

        def _serialize_meta_field(self, meta: BaseMeta, name: str, mark_meta: bool) -> Optional[str]:
            if (field_val := getattr(meta, name)) is not None and isinstance(field_val, SummaryMetaField):
                txt = field_val.text
                return f"[{self._humanize_text(name, title=True)}] {txt}" if mark_meta else txt
            else:
                return None

    doc = _create_doc_with_group_with_metadata()

    # test exporting to Markdown
    params = MarkdownParams(
        include_non_meta=False,
    )
    ser = MarkdownDocSerializer(doc=doc, params=params, meta_serializer=SummaryMarkdownMetaSerializer())
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_summaries.md")
    if GEN_TEST_DATA:
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(actual)
    else:
        with open(exp_file, "r", encoding="utf-8") as f:
            expected = f.read()
        assert actual == expected
