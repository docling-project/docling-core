"""Public metadata and media-reference access from DocLang source XML."""

import pytest

from dclq.document import DclqError, LoadedDocument

DOCUMENT = b"""<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.8">
  <head>
    <custom>
      <delivery__batch value="batch-42"/>
      <delivery__source>lab archive</delivery__source>
    </custom>
  </head>
  <group>
    <custom><docling_machine__group_name value="slide-4"/></custom>
    <text>On slide four</text>
    <picture><src uri="media/figure-1.png"/></picture>
  </group>
  <group><text>Outside a slide group</text></group>
</doclang>"""


def test_head_metadata_exposes_custom_attributes_and_text() -> None:
    loaded = LoadedDocument.load("-", stdin_bytes=DOCUMENT)

    assert loaded.head_metadata == {
        "delivery__batch": {"value": "batch-42"},
        "delivery__source": {"text": "lab archive"},
    }


def test_picture_uris_preserve_external_references_without_loading_images() -> None:
    loaded = LoadedDocument.load("-", stdin_bytes=DOCUMENT)

    picture = "/d:doclang/d:group[1]/d:picture[1]"
    assert loaded.picture_uris == {picture: "media/figure-1.png"}
    assert picture in loaded.raw_elements
    assert f"{picture}/d:src[1]" in loaded.raw_elements


def test_slide_number_resolves_enclosing_group_for_nested_items() -> None:
    loaded = LoadedDocument.load("-", stdin_bytes=DOCUMENT)

    assert loaded.slide_number_for_xpath("/d:doclang/d:group[1]") == 4
    assert loaded.slide_number_for_xpath("/d:doclang/d:group[1]/d:text[1]") == 4
    assert loaded.slide_number_for_xpath("/d:doclang/d:group[1]/d:picture[1]") == 4
    assert loaded.slide_number_for_xpath("/d:doclang/d:group[2]/d:text[1]") is None

    with pytest.raises(DclqError, match="unknown source XPath"):
        loaded.slide_number_for_xpath("/d:doclang/d:group[3]/d:text[1]")


def test_absent_head_and_picture_sources_have_empty_accessors() -> None:
    loaded = LoadedDocument.load("-", stdin_bytes=b'<doclang version="0.8"><text>Just text</text></doclang>')

    assert loaded.head_metadata == {}
    assert loaded.picture_uris == {}
    assert loaded.slide_number_for_xpath("/d:doclang/d:text[1]") is None
