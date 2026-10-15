"""Tests for block quotes: ``BlockQuoteGroup``, its serializers, chunking and the schema projector."""

from pathlib import Path

from pydantic import AnyUrl

from docling_core.compat import project_to
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.latex import LaTeXDocSerializer
from docling_core.types.doc import (
    BlockQuoteGroup,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
)

from .test_data_gen_flag import GEN_TEST_DATA

GT_DIR = Path(__file__).parent / "data" / "doc"

CITE = "https://example.org/judgments/2017/5#para45"
ATTRIBUTION = "Section 1, Example Act 2017"


def _verify(exp_file: Path, actual: str) -> None:
    if GEN_TEST_DATA:
        exp_file.write_text(f"{actual}\n", encoding="utf-8")
    else:
        assert actual == exp_file.read_text(encoding="utf-8").rstrip()


def _build_doc() -> DoclingDocument:
    """A judgment passage that quotes a provision: nested list, nested quote, attribution."""
    doc = DoclingDocument(name="block_quote")
    doc.add_title(text="Judgment")
    doc.add_text(label=DocItemLabel.TEXT, text="The provision reads as follows.")

    quote = doc.add_block_quote_group(cite=AnyUrl(CITE))
    doc.add_text(label=DocItemLabel.TEXT, text="A person commits an offence if the person:", parent=quote)
    lst = doc.add_list_group(parent=quote)
    doc.add_list_item(text="does an act; and", marker="(a)", enumerated=True, parent=lst)
    doc.add_list_item(text="intends the result.", marker="(b)", enumerated=True, parent=lst)
    inner = doc.add_block_quote_group(parent=quote)
    doc.add_text(label=DocItemLabel.TEXT, text="A nested quotation inside the provision.", parent=inner)
    attribution = doc.add_text(label=DocItemLabel.CAPTION, text=ATTRIBUTION, parent=quote)
    quote.captions.append(attribution.get_ref())

    doc.add_text(label=DocItemLabel.TEXT, text="The court agreed.")
    plain = doc.add_group(label=GroupLabel.BLOCK_QUOTE)
    doc.add_text(label=DocItemLabel.TEXT, text="A quote without source or attribution.", parent=plain)
    return doc


def test_add_group_and_insert_group_route_to_block_quote_group():
    doc = DoclingDocument(name="t")
    group = doc.add_group(label=GroupLabel.BLOCK_QUOTE, name="quote")
    assert isinstance(group, BlockQuoteGroup)
    assert group.name == "quote"
    assert group.cite is None
    assert group.captions == []

    inserted = doc.insert_group(sibling=group, label=GroupLabel.BLOCK_QUOTE, after=False)
    assert isinstance(inserted, BlockQuoteGroup)
    assert doc.body.children[0].resolve(doc) is inserted

    with_cite = doc.insert_block_quote_group(sibling=group, cite=AnyUrl(CITE))
    assert isinstance(with_cite, BlockQuoteGroup)
    assert str(with_cite.cite) == CITE
    assert doc.body.children[2].resolve(doc) is with_cite


def test_json_round_trip_keeps_subtype_cite_and_attribution():
    doc = _build_doc()
    loaded = DoclingDocument.model_validate_json(doc.model_dump_json())

    quote = loaded.groups[0]
    assert isinstance(quote, BlockQuoteGroup)
    assert str(quote.cite) == CITE
    assert quote.caption_text(loaded) == ATTRIBUTION
    assert quote.captions[0] == quote.children[-1]

    assert isinstance(loaded.groups[2], BlockQuoteGroup)  # the nested quote
    assert isinstance(loaded.groups[3], BlockQuoteGroup)  # the plain quote
    assert loaded.groups[3].cite is None


def test_html_export():
    _verify(GT_DIR / "block_quote.gt.html", _build_doc().export_to_html())


def test_html_cite_attribute_is_escaped():
    doc = DoclingDocument(name="t")
    quote = doc.add_block_quote_group(cite=AnyUrl("https://example.org/?a=1&b=2"))
    doc.add_text(label=DocItemLabel.TEXT, text="Quoted.", parent=quote)
    assert '<blockquote cite="https://example.org/?a=1&amp;b=2">' in doc.export_to_html()


def test_markdown_export():
    _verify(GT_DIR / "block_quote.gt.md", _build_doc().export_to_markdown())


def test_attribution_dash_is_normalized():
    """A source dash on the attribution is replaced, never doubled, and never a bare hyphen."""
    doc = DoclingDocument(name="t")
    for text in ("- Author", "\u2014 Author", "Author"):
        quote = doc.add_block_quote_group()
        doc.add_text(label=DocItemLabel.TEXT, text="Quoted.", parent=quote)
        attribution = doc.add_text(label=DocItemLabel.CAPTION, text=text, parent=quote)
        quote.captions.append(attribution.get_ref())
    markdown = doc.export_to_markdown()
    assert markdown.count("> \u2014 Author") == 3
    assert "> -" not in markdown
    latex = LaTeXDocSerializer(doc=doc).serialize().text
    assert latex.count("\\hfill--- Author") == 3


def test_latex_export():
    _verify(GT_DIR / "block_quote.gt.tex", LaTeXDocSerializer(doc=_build_doc()).serialize().text)


def test_doctags_export_is_transparent():
    _verify(GT_DIR / "block_quote.gt.dt", _build_doc().export_to_doctags())


def test_doclang_named_groups_round_trip():
    doc = _build_doc()
    text = doc.export_to_doclang(add_named_groups=True)
    _verify(GT_DIR / "block_quote.gt.dclang", text)

    loaded = DocLangDocDeserializer().deserialize_str(text)
    quote = loaded.groups[0]
    assert isinstance(quote, BlockQuoteGroup)
    child_texts = [child.resolve(loaded).text for child in quote.children if child.cref.startswith("#/texts/")]
    assert child_texts[0] == "A person commits an offence if the person:"
    assert child_texts[-1] == ATTRIBUTION


def test_hierarchical_chunker_keeps_quote_and_attribution_together():
    chunks = list(HierarchicalChunker().chunk(_build_doc()))
    texts = [chunk.text for chunk in chunks]
    quote_chunk = next(text for text in texts if "A person commits an offence" in text)
    assert "does an act" in quote_chunk
    assert "A nested quotation" in quote_chunk
    assert ATTRIBUTION in quote_chunk
    assert sum(ATTRIBUTION in text for text in texts) == 1


def test_delete_items_keeps_attribution_reference_in_sync():
    doc = _build_doc()
    intro = doc.texts[1]
    assert intro.text == "The provision reads as follows."
    doc.delete_items(node_items=[intro])

    quote = doc.groups[0]
    assert isinstance(quote, BlockQuoteGroup)
    assert quote.caption_text(doc) == ATTRIBUTION
    assert quote.captions[0] == quote.children[-1]


def test_projection_to_1_10_keeps_content_in_order():
    doc = _build_doc()
    projected = project_to(doc, target_version="1.10.0")

    assert not any(isinstance(group, BlockQuoteGroup) for group in projected.groups)
    assert all(group.label == GroupLabel.UNSPECIFIED for group in projected.groups if group.label != GroupLabel.LIST)
    markdown = projected.export_to_markdown()
    assert ">" not in markdown
    assert markdown.index("A nested quotation") < markdown.index(ATTRIBUTION) < markdown.index("The court agreed.")
