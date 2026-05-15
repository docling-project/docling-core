from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.serializer.html import HTMLDocSerializer, HTMLParams
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
    DocumentMeta,
    EntitiesMetaField,
    EntityMention,
    GroupLabel,
    HumanLanguageLabel,
    KeywordPrediction,
    KeywordsMetaField,
    LanguageMetaField,
    MetaFieldName,
    MetaUtils,
    NodeItem,
    PropertyValue,
    PropertyValueKind,
    RefItem,
    Statement,
    StatementProperty,
    StatementsMetaField,
    SummaryMetaField,
    TopicMetaField,
    TopicPrediction,
)

from .test_data_gen_flag import GEN_TEST_DATA
from .test_utils import assert_or_generate_ground_truth


class CustomCoordinates(BaseModel):
    longitude: float
    latitude: float


@pytest.fixture(scope="module")
def dummy_doc_with_meta() -> DoclingDocument:
    """Fixture that loads dummy_doc_with_meta.yaml once per module."""
    src = Path("test/data/doc/dummy_doc_with_meta.yaml")
    return DoclingDocument.load_from_yaml(filename=src)


@pytest.fixture(scope="module")
def doc_with_group_with_metadata() -> DoclingDocument:
    """Fixture that creates a document with groups and metadata once per module."""
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


def test_metadata_usage(dummy_doc_with_meta: DoclingDocument) -> None:
    doc = dummy_doc_with_meta.model_copy(deep=True)

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
    exp_file = Path("test/data/doc/dummy_doc_with_meta_modified.yaml")
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


def test_namespace_absence_raises(dummy_doc_with_meta: DoclingDocument):
    example_item = RefItem(cref="#/texts/2").resolve(doc=dummy_doc_with_meta)

    with pytest.raises(ValueError):
        example_item.meta.my_corp_programmaticaly_added_field = True


def test_ser_deser(doc_with_group_with_metadata: DoclingDocument):
    doc = doc_with_group_with_metadata

    # test dumping to and loading from YAML
    exp_file = Path("test/data/doc/group_with_metadata.yaml")
    if GEN_TEST_DATA:
        doc.save_as_yaml(filename=exp_file)
    else:
        expected = DoclingDocument.load_from_yaml(filename=exp_file)
        assert doc == expected


def test_md_ser_default(doc_with_group_with_metadata: DoclingDocument):
    # test exporting to Markdown
    doc = doc_with_group_with_metadata
    params = MarkdownParams()
    ser = MarkdownDocSerializer(doc=doc, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_default.md")
    assert_or_generate_ground_truth(actual, exp_file)


def test_md_ser_marked(doc_with_group_with_metadata: DoclingDocument):
    # test exporting to Markdown
    doc = doc_with_group_with_metadata
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


def test_md_ser_allowed_meta_names(doc_with_group_with_metadata: DoclingDocument):
    params = MarkdownParams(
        allowed_meta_names={
            MetaUtils.create_meta_field_name(namespace="my_corp", name="test_1"),
        },
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc_with_group_with_metadata, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_allowed_meta_names.md")
    assert_or_generate_ground_truth(actual, exp_file)


def test_md_ser_blocked_meta_names(doc_with_group_with_metadata: DoclingDocument):
    params = MarkdownParams(
        blocked_meta_names={
            MetaUtils.create_meta_field_name(namespace="my_corp", name="test_1"),
            MetaFieldName.SUMMARY.value,
        },
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc_with_group_with_metadata, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_blocked_meta_names.md")
    assert_or_generate_ground_truth(actual, exp_file)


def test_md_ser_without_non_meta(doc_with_group_with_metadata: DoclingDocument):
    params = MarkdownParams(
        include_non_meta=False,
        mark_meta=True,
    )
    ser = MarkdownDocSerializer(doc=doc_with_group_with_metadata, params=params)
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_without_non_meta.md")
    assert_or_generate_ground_truth(actual, exp_file)


def test_ser_custom_meta_serializer(doc_with_group_with_metadata: DoclingDocument):
    class SummaryMarkdownMetaSerializer(MarkdownMetaSerializer):
        @override
        def serialize(
            self,
            *,
            item: NodeItem,
            doc: DoclingDocument,
            level: int | None = None,
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

        def _serialize_meta_field(self, meta: BaseMeta, name: str, mark_meta: bool) -> str | None:
            if (field_val := getattr(meta, name)) is not None and isinstance(field_val, SummaryMetaField):
                txt = field_val.text
                return f"[{self._humanize_text(name, title=True)}] {txt}" if mark_meta else txt
            else:
                return None

    # test exporting to Markdown
    params = MarkdownParams(
        include_non_meta=False,
    )
    ser = MarkdownDocSerializer(doc=doc_with_group_with_metadata, params=params, meta_serializer=SummaryMarkdownMetaSerializer())
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_summaries.md")
    assert_or_generate_ground_truth(actual, exp_file)


def test_document_level_metadata(dummy_doc_with_meta: DoclingDocument) -> None:
    """Test that document-level metadata can be loaded and accessed through 'body' field."""
    # Verify document-level metadata exists
    assert dummy_doc_with_meta.body.meta is not None
    assert dummy_doc_with_meta.body.meta.summary is not None
    assert dummy_doc_with_meta.body.meta.summary.text == "This is a document-level summary describing the entire document."
    assert dummy_doc_with_meta.body.meta.summary.confidence == 0.98

    # Verify custom metadata fields at document level
    custom_part = dummy_doc_with_meta.body.meta.get_custom_part()
    assert custom_part["my_corp__doc_category"] == "technical_report"
    assert custom_part["my_corp__doc_version"] == "1.0"

    # Verify that item-level metadata still works alongside document-level metadata
    first_text = dummy_doc_with_meta.texts[1]  # The title item
    assert first_text.meta is not None
    assert first_text.meta.summary is not None
    assert first_text.meta.summary.text == "This is a title."


def test_semantic_base_meta_fields_roundtrip_and_html_rendering() -> None:
    doc = DoclingDocument(name="semantic-meta")
    item = doc.add_text(label=DocItemLabel.TEXT, text="IBM is based in Zurich.")
    item.meta = BaseMeta(
        summary=SummaryMetaField(text="A short company/location statement."),
        language=LanguageMetaField(code=HumanLanguageLabel.EN),
        entities=EntitiesMetaField(
            mentions=[
                EntityMention(text="IBM", label="ORG"),
                EntityMention(text="Zurich", label="LOC"),
            ]
        ),
    )

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    meta = roundtrip.texts[0].meta
    assert meta is not None
    assert meta.language is not None
    assert meta.language.code == HumanLanguageLabel.EN
    assert meta.entities is not None
    assert [mention.text for mention in meta.entities.mentions] == ["IBM", "Zurich"]

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert "data-meta-language" in html
    assert "data-meta-entities" in html
    assert ">en<" in html
    assert "IBM (ORG), Zurich (LOC)" in html


def test_topics_roundtrip_and_rendering() -> None:
    doc = DoclingDocument(name="topics-test")
    item = doc.add_text(label=DocItemLabel.TEXT, text="Machine learning advances.")
    item.meta = BaseMeta(
        topics=TopicMetaField(
            predictions=[
                TopicPrediction(label="Artificial Intelligence", taxonomy="arXiv", taxonomy_id="cs.AI"),
                TopicPrediction(label="Machine Learning"),
            ]
        ),
    )

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    meta = roundtrip.texts[0].meta
    assert meta is not None
    assert meta.topics is not None
    assert len(meta.topics.predictions) == 2
    assert meta.topics.predictions[0].label == "Artificial Intelligence"
    assert meta.topics.predictions[0].taxonomy == "arXiv"
    assert meta.topics.get_main_prediction().label == "Artificial Intelligence"

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert "data-meta-topics" in html
    assert "Artificial Intelligence [arXiv]" in html
    assert "Machine Learning" in html


def test_keywords_roundtrip_and_rendering() -> None:
    doc = DoclingDocument(name="keywords-test")
    item = doc.add_text(label=DocItemLabel.TEXT, text="Polymers and material science.")
    item.meta = BaseMeta(
        keywords=KeywordsMetaField(
            predictions=[
                KeywordPrediction(text="polymer", weight=0.9),
                KeywordPrediction(text="material science", normalized="materials science"),
            ]
        ),
    )

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    meta = roundtrip.texts[0].meta
    assert meta is not None
    assert meta.keywords is not None
    assert len(meta.keywords.predictions) == 2
    assert meta.keywords.predictions[0].text == "polymer"
    assert meta.keywords.predictions[0].weight == 0.9

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert "data-meta-keywords" in html
    assert "polymer" in html


def test_property_value_scalar() -> None:
    pv = PropertyValue(kind=PropertyValueKind.SCALAR, scalar=13.2)
    assert pv.scalar == 13.2


def test_property_value_range() -> None:
    pv = PropertyValue(kind=PropertyValueKind.RANGE, range_min=10.0, range_max=20.0)
    assert pv.range_min == 10.0
    assert pv.range_max == 20.0


def test_property_value_text() -> None:
    pv = PropertyValue(kind=PropertyValueKind.TEXT, text="positive")
    assert pv.text == "positive"


def test_property_value_categorical() -> None:
    pv = PropertyValue(kind=PropertyValueKind.CATEGORICAL, text="high")
    assert pv.text == "high"


def test_property_value_scalar_missing_raises() -> None:
    with pytest.raises(Exception):
        PropertyValue(kind=PropertyValueKind.SCALAR)


def test_property_value_range_missing_raises() -> None:
    with pytest.raises(Exception):
        PropertyValue(kind=PropertyValueKind.RANGE, range_min=10.0)


def test_property_value_text_missing_raises() -> None:
    with pytest.raises(Exception):
        PropertyValue(kind=PropertyValueKind.TEXT)


def test_statement_subject_properties_shape() -> None:
    """Test Statement with subject + properties (no predicate/object)."""
    stmt = Statement(
        subject=EntityMention(text="Patient X"),
        properties=[
            StatementProperty(
                name="hemoglobin",
                value=PropertyValue(kind=PropertyValueKind.SCALAR, scalar=13.2),
                unit="g/dL",
            ),
        ],
    )
    assert stmt.subject.text == "Patient X"
    assert stmt.predicate is None
    assert stmt.object is None
    assert len(stmt.properties) == 1
    assert stmt.properties[0].value.scalar == 13.2
    assert stmt.properties[0].unit == "g/dL"


def test_statement_triple_shape() -> None:
    """Test Statement with subject + predicate + object (full triple)."""
    stmt = Statement(
        subject=EntityMention(text="Drug A", label="DRUG"),
        predicate=EntityMention(text="inhibits", label="RELATION"),
        object=EntityMention(text="Protein B", label="PROTEIN"),
    )
    assert stmt.subject.text == "Drug A"
    assert stmt.predicate.text == "inhibits"
    assert stmt.object.text == "Protein B"


def test_statements_roundtrip_and_rendering() -> None:
    doc = DoclingDocument(name="statements-test")
    item = doc.add_text(label=DocItemLabel.TEXT, text="Drug A inhibits Protein B.")
    item.meta = BaseMeta(
        statements=StatementsMetaField(
            statements=[
                Statement(
                    subject=EntityMention(text="Drug A", label="DRUG"),
                    predicate=EntityMention(text="inhibits"),
                    object=EntityMention(text="Protein B", label="PROTEIN"),
                ),
            ]
        ),
    )

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    meta = roundtrip.texts[0].meta
    assert meta is not None
    assert meta.statements is not None
    assert len(meta.statements.statements) == 1
    stmt = meta.statements.statements[0]
    assert stmt.subject.text == "Drug A"
    assert stmt.predicate.text == "inhibits"
    assert stmt.object.text == "Protein B"

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert "data-meta-statements" in html
    assert "Drug A inhibits Protein B" in html


def test_document_meta_roundtrip() -> None:
    """Test DocumentMeta on DoclingDocument."""
    doc = DoclingDocument(name="doc-meta-test")
    doc.meta = DocumentMeta(
        summary=SummaryMetaField(text="A paper about polymers."),
        language=LanguageMetaField(code=HumanLanguageLabel.EN),
        topics=TopicMetaField(
            predictions=[TopicPrediction(label="Materials Science")]
        ),
        keywords=KeywordsMetaField(
            predictions=[KeywordPrediction(text="polymer"), KeywordPrediction(text="synthesis")]
        ),
        title="Advances in Polymer Science",
        authors=["Alice", "Bob"],
        publication_date="2026-01-15",
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Some content.")

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    assert roundtrip.meta is not None
    assert roundtrip.meta.title == "Advances in Polymer Science"
    assert roundtrip.meta.authors == ["Alice", "Bob"]
    assert roundtrip.meta.publication_date == "2026-01-15"
    assert roundtrip.meta.language.code == HumanLanguageLabel.EN
    assert roundtrip.meta.topics.predictions[0].label == "Materials Science"
    assert len(roundtrip.meta.keywords.predictions) == 2
    assert roundtrip.meta.summary.text == "A paper about polymers."


def test_document_meta_none_by_default() -> None:
    """Verify DoclingDocument.meta defaults to None (backward compat)."""
    doc = DoclingDocument(name="empty")
    assert doc.meta is None


def test_basemeta_new_fields_default_none() -> None:
    """Verify all new BaseMeta fields default to None (backward compat)."""
    meta = BaseMeta()
    assert meta.topics is None
    assert meta.keywords is None
    assert meta.statements is None
    assert meta.language is None
    assert meta.entities is None
