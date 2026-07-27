from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import override

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.html import HTMLDocSerializer, HTMLParams
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownMetaSerializer,
    MarkdownParams,
)
from docling_core.types.doc import (
    BaseMeta,
    DataPointMention,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    EntitiesMetaField,
    EntityMention,
    GroupLabel,
    HumanLanguageLabel,
    KeywordsMetaField,
    LanguageMetaField,
    MetaFieldName,
    MetaUtils,
    NodeItem,
    RefItem,
    SummaryMetaField,
    TopicsMetaField,
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
        with open(exp_file, encoding="utf-8") as f:
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
    ser = MarkdownDocSerializer(
        doc=doc_with_group_with_metadata, params=params, meta_serializer=SummaryMarkdownMetaSerializer()
    )
    ser_res = ser.serialize()
    actual = ser_res.text
    exp_file = Path("test/data/doc/group_with_metadata_summaries.md")
    assert_or_generate_ground_truth(actual, exp_file)


def test_document_level_metadata(dummy_doc_with_meta: DoclingDocument) -> None:
    """Test that document-level metadata can be loaded and accessed through 'body' field."""
    # Verify document-level metadata exists
    assert dummy_doc_with_meta.body.meta is not None
    assert dummy_doc_with_meta.body.meta.summary is not None
    assert (
        dummy_doc_with_meta.body.meta.summary.text == "This is a document-level summary describing the entire document."
    )
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
                EntityMention(text="IBM", label="ORG", charspan=(0, 3)),
                EntityMention(text="Zurich", label="LOC", charspan=(16, 22)),
            ]
        ),
        keywords=KeywordsMetaField(values=["ibm", "zurich", "company"]),
        topics=TopicsMetaField(values=["business", "geography"]),
    )

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    meta = roundtrip.texts[0].meta
    assert meta is not None
    assert meta.language is not None
    assert meta.language.code == HumanLanguageLabel.EN
    assert meta.entities is not None
    assert [mention.text for mention in meta.entities.mentions] == ["IBM", "Zurich"]
    assert meta.keywords is not None and meta.keywords.values == ["ibm", "zurich", "company"]
    assert meta.topics is not None and meta.topics.values == ["business", "geography"]
    assert meta.has_content()

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert 'data-meta-name="language"' in html
    assert 'data-meta-name="entities"' in html
    assert 'data-meta-name="keywords"' in html
    assert 'data-meta-name="topics"' in html
    assert ">en<" in html
    assert "IBM (ORG, [0,3]), Zurich (LOC, [16,22])" in html
    assert "ibm, zurich, company" in html
    assert ">business, geography<" in html

    # duplicate values are removed without rejection
    assert KeywordsMetaField(values=["ai", "ml", "ai"]).values == ["ai", "ml"]
    assert TopicsMetaField(values=["nlp", "nlp"]).values == ["nlp"]


def test_html_escapes_entity_text() -> None:
    doc = DoclingDocument(name="escaped-entity-meta")
    item = doc.add_text(label=DocItemLabel.TEXT, text="A<B & C> appears here.")
    item.meta = BaseMeta(
        entities=EntitiesMetaField(
            mentions=[
                EntityMention(text="A<B & C>", label="TAG", charspan=(0, 7)),
            ]
        ),
    )

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert "A&lt;B &amp; C&gt; (TAG, [0,7])" in html


def test_html_skips_empty_base_meta() -> None:
    doc = DoclingDocument(name="empty-meta")
    item = doc.add_text(label=DocItemLabel.TEXT, text="IBM is based in Zurich.")
    item.meta = BaseMeta()

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert '<details class="docling-meta">' not in html
    assert "data-meta-entities" not in html


def test_html_escapes_keywords() -> None:
    doc = DoclingDocument(name="kw-escape")
    item = doc.add_text(label=DocItemLabel.TEXT, text="x")
    item.meta = BaseMeta(keywords=KeywordsMetaField(values=["A<B & C>"]))

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert "A&lt;B &amp; C&gt;" in html


def test_md_marked_renders_keywords_and_topics() -> None:
    doc = DoclingDocument(name="kw-md")
    item = doc.add_text(label=DocItemLabel.TEXT, text="IBM is based in Zurich.")
    item.meta = BaseMeta(
        keywords=KeywordsMetaField(values=["ibm", "zurich"]),
        topics=TopicsMetaField(values=["business"]),
    )
    md = MarkdownDocSerializer(doc=doc, params=MarkdownParams(mark_meta=True)).serialize().text
    assert "[Keywords] ibm, zurich" in md
    assert "[Topics] business" in md


def test_keywords_topics_required_values() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        KeywordsMetaField(values=[])
    with pytest.raises(ValidationError, match="at least 1 item"):
        TopicsMetaField(values=[])
    with pytest.raises(ValidationError, match="list of strings"):
        TopicsMetaField(values=34)
    with pytest.raises(ValidationError, match="valid string"):
        TopicsMetaField(values=[34])


# ---------------------------------------------------------------------------
# Entity mentions tests
# ---------------------------------------------------------------------------

_META_GT_DIR = Path("test/data/doc/meta")


@pytest.fixture(scope="module")
def doc_with_entity_mentions() -> DoclingDocument:
    """A document with three paragraphs annotated with mixed entity mentions.

    Paragraph 1 — financial statement:
        "IBM reported revenue of $16.3B in Q3 2025, up 9% year-over-year."
        Mentions: ORG, DATE, REVENUE (DataPointMention), GROWTH_RATE (DataPointMention)

    Paragraph 2 — population fact:
        "The population of Switzerland in 2026 is approximately 9 million people."
        Mentions: LOC, DATE, HEADCOUNT (DataPointMention)

    Paragraph 3 — range data point:
        "Operating margins improved to between 18% and 21% during the period."
        Mentions: OPERATING_MARGIN (DataPointMention, range)
    """
    doc = DoclingDocument(name="entity-mentions-example")

    # ── Paragraph 1 ──────────────────────────────────────────────────────────
    p1 = doc.add_text(
        label=DocItemLabel.TEXT,
        text="IBM reported revenue of $16.3B in Q3 2025, up 9% year-over-year.",
    )
    p1.meta = BaseMeta(
        entities=EntitiesMetaField(
            mentions=[
                EntityMention(text="IBM", label="ORG", charspan=(0, 3)),
                EntityMention(text="Q3 2025", label="DATE", charspan=(34, 41)),
                DataPointMention(
                    text="16.3 billion USD",
                    orig="$16.3B",
                    label="REVENUE",
                    charspan=(24, 30),
                    value=16.3,
                    unit="USD",
                    scale="billion",
                    normalized_value=16_300_000_000.0,
                    display_dp=1,
                    precision="exact",
                ),
                DataPointMention(
                    text="9 percent",
                    orig="9%",
                    label="GROWTH_RATE",
                    charspan=(46, 48),
                    value=9.0,
                    unit="%",
                    display_dp=0,
                    precision="exact",
                    direction="increase",
                ),
            ]
        )
    )

    # ── Paragraph 2 ──────────────────────────────────────────────────────────
    p2 = doc.add_text(
        label=DocItemLabel.TEXT,
        text="The population of Switzerland in 2026 is approximately 9 million people.",
    )
    p2.meta = BaseMeta(
        entities=EntitiesMetaField(
            mentions=[
                EntityMention(text="Switzerland", label="LOC", charspan=(18, 29)),
                EntityMention(text="2026", label="DATE", charspan=(33, 37)),
                DataPointMention(
                    text="9 million",
                    label="POPULATION",
                    charspan=(55, 64),
                    value=9.0,
                    scale="million",
                    normalized_value=9_000_000.0,
                    display_dp=0,
                    precision="approximate",
                ),
            ]
        )
    )

    # ── Paragraph 3 ──────────────────────────────────────────────────────────
    p3 = doc.add_text(
        label=DocItemLabel.TEXT,
        text="Operating margins improved to between 18% and 21% during the period.",
    )
    p3.meta = BaseMeta(
        entities=EntitiesMetaField(
            mentions=[
                DataPointMention(
                    text="18 to 21 percent",
                    orig="18%",
                    label="OPERATING_MARGIN",
                    charspan=(37, 40),
                    value=18.0,
                    unit="%",
                    display_dp=0,
                    precision="range_low",
                    range_end=21.0,
                    direction="increase",
                ),
            ]
        )
    )

    return doc


def test_data_point_roundtrip_and_html_rendering(
    doc_with_entity_mentions: DoclingDocument,
) -> None:
    doc = doc_with_entity_mentions

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    meta_p1 = roundtrip.texts[0].meta
    assert meta_p1 is not None and meta_p1.entities is not None
    assert meta_p1.has_content()

    mentions = meta_p1.entities.mentions
    # plain EntityMention (ORG) has no value field — falls back to base type in Union
    org = mentions[0]
    assert isinstance(org, EntityMention)
    assert not isinstance(org, DataPointMention)

    # DataPointMention (REVENUE) — exact, scaled value
    revenue = mentions[2]
    assert isinstance(revenue, DataPointMention)
    assert revenue.orig == "$16.3B"
    assert revenue.value == 16.3
    assert revenue.unit == "USD"
    assert revenue.scale == "billion"
    assert revenue.normalized_value == 16_300_000_000.0
    assert revenue.display_dp == 1
    assert revenue.precision == "exact"
    assert revenue.scale_factor == 1e9
    assert revenue.compute_normalized_value() == revenue.normalized_value

    html = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert 'data-meta-name="entities"' in html
    assert "IBM (ORG, [0,3])" in html
    assert "16.3 billion USD (REVENUE, [24,30])" in html


def test_data_point_range_and_direction(
    doc_with_entity_mentions: DoclingDocument,
) -> None:
    doc = doc_with_entity_mentions

    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))

    # GROWTH_RATE from paragraph 1 — direction + exact precision
    meta_p1 = roundtrip.texts[0].meta
    assert meta_p1 is not None and meta_p1.entities is not None
    growth = meta_p1.entities.mentions[3]
    assert isinstance(growth, DataPointMention)
    assert growth.value == 9.0
    assert growth.unit == "%"
    assert growth.display_dp == 0
    assert growth.precision == "exact"
    assert growth.direction == "increase"

    # OPERATING_MARGIN from paragraph 3 — range_low + range_end
    meta_p3 = roundtrip.texts[2].meta
    assert meta_p3 is not None and meta_p3.entities is not None
    margin = meta_p3.entities.mentions[0]
    assert isinstance(margin, DataPointMention)
    assert margin.value == 18.0
    assert margin.range_end == 21.0
    assert margin.precision == "range_low"
    assert margin.display_dp == 0


def test_data_point_numeric_helpers() -> None:
    # scale_factor maps canonical scale strings to their multipliers
    assert DataPointMention(text="4B", value=4.0, scale="billion").scale_factor == 1e9
    assert DataPointMention(text="5k", value=5.0, scale="k").scale_factor == 1e3
    assert DataPointMention(text="x", value=0.0).scale_factor is None

    # compute_normalized_value applies the scale factor on demand
    assert DataPointMention(text="4B", value=4.0, scale="billion").compute_normalized_value() == 4_000_000_000.0
    assert DataPointMention(text="30", value=30.0).compute_normalized_value() == 30.0
    # display_dp is independent of the numeric value: "3%" and "3.0%" share value=3.0
    assert DataPointMention(text="3%", value=3.0, display_dp=0).display_dp == 0
    assert DataPointMention(text="3.0%", value=3.0, display_dp=1).display_dp == 1


def test_data_point_range_end_requires_value() -> None:
    with pytest.raises(ValidationError, match="range_end requires value"):
        DataPointMention.model_validate({"text": "x", "value": None, "range_end": 20.0})


def test_entity_mentions_json_roundtrip(
    doc_with_entity_mentions: DoclingDocument,
) -> None:
    """Ground truth: JSON serialization and structural round-trip."""
    import json

    doc = doc_with_entity_mentions
    gt_path = _META_GT_DIR / "entity_mentions.json"

    result = json.dumps(doc.export_to_dict(), indent=2)
    assert_or_generate_ground_truth(result, gt_path)

    # Structural round-trip: reload from ground-truth JSON and verify mentions
    reloaded = DoclingDocument.model_validate_json(gt_path.read_text())

    meta_p1 = reloaded.texts[0].meta
    assert meta_p1 is not None and meta_p1.entities is not None
    mentions_p1 = meta_p1.entities.mentions
    assert isinstance(mentions_p1[0], EntityMention) and not isinstance(mentions_p1[0], DataPointMention)
    assert isinstance(mentions_p1[2], DataPointMention)
    assert mentions_p1[2].value == 16.3
    assert mentions_p1[2].normalized_value == 16_300_000_000.0

    meta_p2 = reloaded.texts[1].meta
    assert meta_p2 is not None and meta_p2.entities is not None
    mentions_p2 = meta_p2.entities.mentions
    assert isinstance(mentions_p2[2], DataPointMention)
    assert mentions_p2[2].precision == "approximate"

    meta_p3 = reloaded.texts[2].meta
    assert meta_p3 is not None and meta_p3.entities is not None
    mentions_p3 = meta_p3.entities.mentions
    assert isinstance(mentions_p3[0], DataPointMention)
    assert mentions_p3[0].range_end == 21.0
    assert mentions_p3[0].precision == "range_low"


def test_entity_mentions_markdown_serialization(
    doc_with_entity_mentions: DoclingDocument,
) -> None:
    """Ground truth: Markdown serialization (meta marked)."""
    doc = doc_with_entity_mentions
    gt_path = _META_GT_DIR / "entity_mentions.md"

    result = MarkdownDocSerializer(doc=doc, params=MarkdownParams(mark_meta=True)).serialize().text
    assert_or_generate_ground_truth(result, gt_path)


def test_entity_mentions_html_serialization(
    doc_with_entity_mentions: DoclingDocument,
) -> None:
    """Ground truth: HTML serialization."""
    doc = doc_with_entity_mentions
    gt_path = _META_GT_DIR / "entity_mentions.html"

    result = HTMLDocSerializer(doc=doc, params=HTMLParams()).serialize().text
    assert_or_generate_ground_truth(result, gt_path)


def test_entity_mentions_doctags_serialization(
    doc_with_entity_mentions: DoclingDocument,
) -> None:
    """Ground truth: DocTags (Doclang) serialization."""
    doc = doc_with_entity_mentions
    gt_path = _META_GT_DIR / "entity_mentions.dt"

    result = doc.export_to_doctags(add_location=False)
    assert_or_generate_ground_truth(result, gt_path)
