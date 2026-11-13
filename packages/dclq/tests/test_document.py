"""Tests for dclq's public DocLang context-form addresses."""

from dataclasses import replace

import pytest
from dclq.document import DclqError, LoadedDocument

DOCUMENT = b"""<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7">
  <group>
    <text>Alpha</text>
    <text>Beta</text>
  </group>
  <group>
    <text>Gamma</text>
  </group>
</doclang>"""


def test_loaded_document_exposes_context_xpaths_without_changing_source_keys() -> None:
    loaded = LoadedDocument.load("-", stdin_bytes=DOCUMENT)

    assert loaded.context_xpath("/d:doclang") == "/doclang[1]"
    assert loaded.context_xpath("/d:doclang/d:group[1]/d:text[2]") == "/doclang[1]/group[1]/text[2]"
    assert loaded.context_xpath("/d:doclang/d:group[2]/d:text[1]") == "/doclang[1]/group[2]/text[1]"

    mapped = "/d:doclang/d:group[2]/d:text[1]"
    assert mapped in loaded.source_pages
    assert mapped in loaded.raw_elements
    assert loaded.selected_paths(loaded.context_xpath(mapped)) == {mapped}

    with pytest.raises(DclqError, match="unknown source XPath"):
        loaded.context_xpath("/d:doclang/d:group[3]")


def test_unit_context_xpaths_support_multiple_addresses_and_preserve_originals() -> None:
    loaded = LoadedDocument.load("-", stdin_bytes=DOCUMENT)
    first = "/d:doclang/d:group[1]/d:text[1]"
    second = "/d:doclang/d:group[1]/d:text[2]"

    unit = next(unit for unit in loaded.units if first in unit.xpaths)
    assert unit.context_xpaths == ("/doclang[1]/group[1]/text[1]",)
    assert unit.xpath == first

    multi_source = replace(unit, xpaths=(first, second))
    assert multi_source.context_xpaths == (
        "/doclang[1]/group[1]/text[1]",
        "/doclang[1]/group[1]/text[2]",
    )
    assert multi_source.xpaths == (first, second)
