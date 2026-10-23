"""Unit tests for document tokens."""

from docling_core.types.doc.document import CodeItem, DoclingDocument, DocTagsDocument
from docling_core.types.doc.labels import CodeLanguageLabel
from docling_core.types.doc.tokens import DocumentToken


def test_get_location_normal_bbox():
    """A well-ordered bbox produces correctly-ordered location tokens."""
    loc = DocumentToken.get_location(
        bbox=(100.0, 200.0, 300.0, 400.0),
        page_w=1000.0,
        page_h=1000.0,
    )
    assert loc == "<loc_50><loc_100><loc_150><loc_200>"


def test_get_location_inverted_bbox_does_not_crash():
    """A near-degenerate, slightly inverted bbox must not raise.

    Regression test for #620: layout-model regression can emit a bbox with
    bbox[1] > bbox[3] (or bbox[0] > bbox[2]) on thin/degenerate elements.
    get_location already normalizes coordinates via min()/max() downstream,
    so it should serialize such boxes instead of crashing export_to_doctags.
    """
    # bbox from the issue: y is inverted by 7.7pt (633.39 > 625.68).
    inverted = DocumentToken.get_location(
        bbox=(0.0, 633.3925132751465, 100.0, 625.6789665222168),
        page_w=1000.0,
        page_h=1000.0,
    )
    # The result must equal the same box with y-coordinates ordered correctly.
    normalized = DocumentToken.get_location(
        bbox=(0.0, 625.6789665222168, 100.0, 633.3925132751465),
        page_w=1000.0,
        page_h=1000.0,
    )
    assert inverted == normalized


def test_get_location_fully_inverted_bbox():
    """Both axes inverted still normalizes to the ordered box."""
    inverted = DocumentToken.get_location(
        bbox=(300.0, 400.0, 100.0, 200.0),
        page_w=1000.0,
        page_h=1000.0,
    )
    ordered = DocumentToken.get_location(
        bbox=(100.0, 200.0, 300.0, 400.0),
        page_w=1000.0,
        page_h=1000.0,
    )
    assert inverted == ordered


def test_code_language_tokens_cover_all_labels():
    """Every CodeLanguageLabel value has a DocTags code-language token.

    Regression test: `JSON`, `DocLang`, `Latex` and `Tikz` were added to
    CodeLanguageLabel without a matching entry in the token table, so DocTags
    export of a code item with one of those languages raised
    `ValueError: '<_JSON_>' is not a valid _CodeLanguageToken`.
    """
    for code_language in CodeLanguageLabel:
        assert DocumentToken.get_code_language_token(code_language.value) == f"<_{code_language.value}_>"


def test_code_languages_roundtrip_through_doctags():
    """A code item of any valid language exports to DocTags and loads back."""
    for code_language in CodeLanguageLabel:
        doc = DoclingDocument(name="test")
        doc.add_code(text="code snippet", code_language=code_language)

        doctags = doc.export_to_doctags()
        assert f"<_{code_language.value}_>code snippet" in doctags

        reloaded = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([doctags], None))
        assert len(reloaded.texts) == 1
        reloaded_code = reloaded.texts[0]
        assert isinstance(reloaded_code, CodeItem)
        assert reloaded_code.code_language == code_language
