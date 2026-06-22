import pytest
from docling_core.types.doc import (
    DoclingDocument,
    DocItemLabel,
    DocumentOrigin,
    PageItem,
    Size,
)
from docling_core.proto import docling_document_to_proto
from docling_core.proto.gen.ai.docling.core.v1 import docling_document_pb2 as pb2
from docling_core.utils import conversion

def test_minimal_doc_conversion():
    doc = DoclingDocument(name="test_doc")
    proto = docling_document_to_proto(doc)
    
    assert proto.name == "test_doc"
    assert proto.body.name == "_root_"
    assert proto.furniture.name == "_root_"

def test_doc_with_text_conversion():
    doc = DoclingDocument(name="test_doc")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
    
    proto = docling_document_to_proto(doc)
    
    assert len(proto.texts) == 1
    assert proto.texts[0].text.base.text == "Hello world"
    assert proto.texts[0].text.base.label == pb2.DOC_ITEM_LABEL_PARAGRAPH

def test_doc_with_title_conversion():
    doc = DoclingDocument(name="test_doc")
    doc.add_title(text="Main Title")
    
    proto = docling_document_to_proto(doc)
    
    assert len(proto.texts) == 1
    assert proto.texts[0].title.base.text == "Main Title"
    assert proto.texts[0].title.base.label == pb2.DOC_ITEM_LABEL_TITLE


def test_pages_map_keys_are_ints_in_proto():
    doc = DoclingDocument(name="test_doc")
    doc.pages = {
        1: PageItem(size=Size(width=100.0, height=200.0), page_no=1),
        2: PageItem(size=Size(width=300.0, height=400.0), page_no=2),
    }

    proto = docling_document_to_proto(doc)
    assert 1 in proto.pages
    assert 2 in proto.pages
    assert proto.pages[1].page_no == 1
    assert proto.pages[2].size.width == 300.0


def test_structural_item_labels_are_enum_with_raw_fallback():
    for message_name in ("PictureItem", "TableItem", "KeyValueItem", "FormItem"):
        descriptor = pb2.DESCRIPTOR.message_types_by_name[message_name]
        label_field = descriptor.fields_by_name["label"]
        assert label_field.enum_type is not None
        assert label_field.enum_type.name == "DocItemLabel"
        assert "label_raw" in descriptor.fields_by_name


def test_doc_item_label_fallback_mapping():
    enum_value, raw = conversion._to_doc_item_label_enum_and_raw(DocItemLabel.PICTURE)
    assert enum_value == pb2.DOC_ITEM_LABEL_PICTURE
    assert raw is None

    enum_value, raw = conversion._to_doc_item_label_enum_and_raw("future_new_label")
    assert enum_value == pb2.DOC_ITEM_LABEL_UNSPECIFIED
    assert raw == "future_new_label"


def test_document_origin_binary_hash_uses_uint64_proto_field():
    doc = DoclingDocument(
        name="test_doc",
        origin=DocumentOrigin(
            mimetype="application/pdf",
            binary_hash=18446744073709551615,
            filename="sample.pdf",
        ),
    )

    proto = docling_document_to_proto(doc)
    assert proto.origin.binary_hash == 18446744073709551615


def test_picture_meta_code_field_round_trip():
    from docling_core.types.doc.document import CodeMetaField, PictureMeta
    from docling_core.types.doc.labels import CodeLanguageLabel

    meta = PictureMeta(
        code=CodeMetaField(
            text="print('hi')",
            language=CodeLanguageLabel.PYTHON,
            confidence=0.9,
            created_by="ocr",
        )
    )
    proto_meta = conversion._to_picture_meta(meta)
    assert proto_meta is not None
    assert proto_meta.code.text == "print('hi')"
    assert proto_meta.code.language == pb2.CODE_LANGUAGE_LABEL_PYTHON
    assert proto_meta.code.language_raw == ""
    assert proto_meta.code.confidence == pytest.approx(0.9)
    assert proto_meta.code.created_by == "ocr"


def test_code_language_fallback_for_unknown_value():
    enum_value, raw = conversion._to_code_language_enum_and_raw("BrandNewLang")
    assert enum_value == pb2.CODE_LANGUAGE_LABEL_UNSPECIFIED
    assert raw == "BrandNewLang"


def test_table_data_orientation_conversion():
    from docling_core.types.doc.document import Orientation, TableData

    proto = conversion._to_table_data(
        TableData(num_rows=2, num_cols=3, orientation=Orientation.ROT_90)
    )
    assert proto.orientation == pb2.ORIENTATION_ROT_90
    assert proto.orientation_raw == ""

    enum_value, raw = conversion._to_orientation_enum_and_raw("rot_future")
    assert enum_value == pb2.ORIENTATION_UNSPECIFIED
    assert raw == "rot_future"


def test_code_language_label_proto_covers_latex_tikz_doclang():
    descriptor = pb2.CodeLanguageLabel.DESCRIPTOR
    names = {v.name for v in descriptor.values}
    assert "CODE_LANGUAGE_LABEL_LATEX" in names
    assert "CODE_LANGUAGE_LABEL_TIKZ" in names
    assert "CODE_LANGUAGE_LABEL_DOCLANG" in names


def test_base_meta_language_and_entities_conversion():
    from docling_core.types.doc.document import (
        BaseMeta,
        EntitiesMetaField,
        EntityMention,
        LanguageMetaField,
    )
    from docling_core.types.doc.labels import HumanLanguageLabel

    meta = BaseMeta(
        language=LanguageMetaField(
            code=HumanLanguageLabel.EN,
            confidence=0.95,
            created_by="lang-detector",
        ),
        entities=EntitiesMetaField(
            mentions=[
                EntityMention(
                    text="IBM",
                    orig="IBM",
                    label="ORG",
                    charspan=(0, 3),
                    confidence=0.88,
                )
            ]
        ),
    )
    proto_meta = conversion._to_base_meta(meta)
    assert proto_meta is not None
    assert proto_meta.language.code == pb2.HUMAN_LANGUAGE_LABEL_EN
    assert proto_meta.language.code_raw == ""
    assert proto_meta.language.confidence == pytest.approx(0.95)
    assert len(proto_meta.entities.mentions) == 1
    assert proto_meta.entities.mentions[0].text == "IBM"
    assert proto_meta.entities.mentions[0].charspan.start == 0
    assert proto_meta.entities.mentions[0].charspan.end == 3


def test_base_meta_keywords_and_topics_conversion():
    from docling_core.types.doc.document import (
        BaseMeta,
        KeywordsMetaField,
        TopicsMetaField,
    )

    meta = BaseMeta(
        keywords=KeywordsMetaField(values=["transformer", "attention mechanism"]),
        topics=TopicsMetaField(values=["natural language processing"]),
    )
    proto_meta = conversion._to_base_meta(meta)
    assert proto_meta is not None
    assert list(proto_meta.keywords.values) == ["transformer", "attention mechanism"]
    assert list(proto_meta.topics.values) == ["natural language processing"]


def test_human_language_fallback_for_unknown_value():
    enum_value, raw = conversion._to_human_language_enum_and_raw("xx")
    assert enum_value == pb2.HUMAN_LANGUAGE_LABEL_UNSPECIFIED
    assert raw == "xx"


def test_language_meta_field_has_code_raw_fallback():
    descriptor = pb2.LanguageMetaField.DESCRIPTOR
    code_field = descriptor.fields_by_name["code"]
    assert code_field.enum_type.name == "HumanLanguageLabel"
    assert "code_raw" in descriptor.fields_by_name
