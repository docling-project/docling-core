"""Test publication metadata and chapter indexing in Markdown."""

from pathlib import Path

import pytest
import yaml

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.transforms.serializer.publication import PubMetadata, PubMetadataSerializer
from docling_core.types.doc.base import BoundingBox, ImageRefMode
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel


def _metadata() -> PubMetadata:
    return PubMetadata(
        title="Alice's Adventures in Wonderland",
        authors=["Lewis Carroll"],
        published="2008-06-27",
        language="en",
        source_file="alice.epub",
    )


def _frontmatter(markdown: str) -> dict:
    return yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])


def _book_document() -> DoclingDocument:
    doc = DoclingDocument(name="alice")
    doc.add_title("Alice's Adventures in Wonderland")
    doc.add_heading("CHAPTER I. Dîner à Oxford", level=1)
    doc.add_text(DocItemLabel.TEXT, "Down the rabbit-hole.")
    doc.add_heading("CHAPTER II. The Pool of Tears", level=1)
    doc.add_text(DocItemLabel.TEXT, "Curiouser and curiouser!")
    return doc


def test_frontmatter_holds_the_book_metadata():
    """Metadata supplied to the serializer is rendered as YAML frontmatter."""
    doc = DoclingDocument(name="alice")
    doc.add_text(DocItemLabel.TEXT, "Down the rabbit-hole.")

    ser = MarkdownDocSerializer(doc=doc, metadata_serializer=PubMetadataSerializer(metadata=_metadata()))
    parsed = _frontmatter(ser.serialize().text)

    assert parsed["title"] == "Alice's Adventures in Wonderland"
    assert parsed["authors"] == ["Lewis Carroll"]
    assert parsed["published"] == "2008-06-27"
    assert parsed["language"] == "en"
    assert parsed["source_file"] == "alice.epub"


def test_chapter_byte_offsets_seek_to_the_heading():
    """Each chapter byte offset lands on the first byte of its rendered heading."""
    ser = MarkdownDocSerializer(doc=_book_document(), metadata_serializer=PubMetadataSerializer(metadata=_metadata()))
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)

    assert [chapter["title"] for chapter in parsed["chapters"]] == [
        "Alice's Adventures in Wonderland",
        "CHAPTER I. Dîner à Oxford",
        "CHAPTER II. The Pool of Tears",
    ]
    encoded = markdown.encode("utf-8")
    for chapter in parsed["chapters"]:
        heading = encoded[chapter["byte"] :].split(b"\n", maxsplit=1)[0].decode("utf-8")
        assert heading.lstrip("# ") == chapter["title"]
        assert heading.startswith("#")


def test_chapters_titled_with_h1_are_indexed():
    """TitleItem chapters and level-1 section headers both appear in the index."""
    doc = DoclingDocument(name="novel")
    doc.add_title("Chapter One")
    doc.add_text(DocItemLabel.TEXT, "It was a bright cold day in April.")
    doc.add_title("Chapter Two")
    doc.add_text(DocItemLabel.TEXT, "The Ministry of Truth.")

    ser = MarkdownDocSerializer(doc=doc, metadata_serializer=PubMetadataSerializer(metadata=_metadata()))
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)

    assert [chapter["title"] for chapter in parsed["chapters"]] == ["Chapter One", "Chapter Two"]
    encoded = markdown.encode("utf-8")
    for chapter in parsed["chapters"]:
        assert encoded[chapter["byte"] :].startswith(f"# {chapter['title']}".encode())


def test_chapter_line_offsets_are_one_based():
    """Each chapter line offset indexes the heading line, counting from one."""
    ser = MarkdownDocSerializer(doc=_book_document(), metadata_serializer=PubMetadataSerializer(metadata=_metadata()))
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)
    lines = markdown.splitlines()

    for chapter in parsed["chapters"]:
        assert lines[chapter["line"] - 1].lstrip("# ") == chapter["title"]


@pytest.mark.parametrize("page_break", ["<PB>", "", "\n<é>\n"])
def test_chapter_offsets_account_for_page_break_replacement(page_break):
    """Page-break markers are replaced in the body, so offsets measure the replacement."""
    prov = [ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1)) for page in (1, 2)]
    doc = DoclingDocument(name="pages")
    doc.add_heading("One", level=1, prov=prov[0])
    doc.add_text(DocItemLabel.TEXT, "body", prov=prov[0])
    doc.add_heading("Two", level=1, prov=prov[1])

    ser = MarkdownDocSerializer(
        doc=doc,
        metadata_serializer=PubMetadataSerializer(metadata=_metadata()),
        params=MarkdownParams(page_break_placeholder=page_break),
    )
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)

    assert "#_#_DOCLING_DOC_PAGE_BREAK_" not in markdown
    if page_break:
        assert page_break in markdown
    encoded = markdown.encode("utf-8")
    lines = markdown.splitlines()
    for chapter in parsed["chapters"]:
        assert encoded[chapter["byte"] :].startswith(f"## {chapter['title']}".encode())
        assert lines[chapter["line"] - 1] == f"## {chapter['title']}"


def test_offsets_stay_correct_as_their_digit_width_grows():
    """Chapter offsets are advertised inside the block they measure, so rendering
    them must not resize that block whatever their digit width."""
    digit_widths = set()
    for body_size in (1, 2_000, 1_000_000):
        doc = DoclingDocument(name="offset-width")
        doc.add_text(DocItemLabel.TEXT, "x" * body_size)
        doc.add_heading("Chapter", level=1)

        ser = MarkdownDocSerializer(doc=doc, metadata_serializer=PubMetadataSerializer(metadata=_metadata()))
        markdown = ser.serialize().text
        chapter = _frontmatter(markdown)["chapters"][0]

        assert markdown.encode("utf-8")[chapter["byte"] :].startswith(b"## Chapter")
        digit_widths.add(len(str(chapter["byte"])))

    assert len(digit_widths) > 1, "the body sizes must span several offset digit widths"


def test_frontmatter_size_change_is_rejected(monkeypatch):
    """A frontmatter size change would invalidate every advertised byte offset."""
    original_render = PubMetadataSerializer._render_frontmatter
    render_count = 0

    def unstable_render(self, chapters):
        nonlocal render_count
        render_count += 1
        rendered = original_render(self, chapters)
        return f"{rendered} " if render_count == 2 else rendered

    monkeypatch.setattr(PubMetadataSerializer, "_render_frontmatter", unstable_render)
    serializer = MarkdownDocSerializer(
        doc=_book_document(), metadata_serializer=PubMetadataSerializer(metadata=_metadata())
    )

    with pytest.raises(ValueError, match="frontmatter changed size"):
        serializer.serialize()


def test_empty_serialization_parts_are_ignored():
    """An empty part must not add body separators or phantom chapters."""
    serializer = MarkdownDocSerializer(doc=DoclingDocument(name="empty"), metadata_serializer=PubMetadataSerializer())

    result = serializer.serialize_doc(parts=[SerializationResult()])

    assert result.text == "---\nchapters: []\n---\n\n"


def test_book_without_headings_has_an_empty_chapter_list():
    """A book with no top-level headings still advertises a parseable empty list."""
    doc = DoclingDocument(name="empty")
    doc.add_text(DocItemLabel.TEXT, "No headings here.")

    ser = MarkdownDocSerializer(doc=doc, metadata_serializer=PubMetadataSerializer(metadata=_metadata()))
    parsed = _frontmatter(ser.serialize().text)

    assert parsed["chapters"] == []


def test_publication_markdown_matches_reference():
    """The publication hook retains the existing metadata and chapter format."""
    serializer = MarkdownDocSerializer(
        doc=_book_document(), metadata_serializer=PubMetadataSerializer(metadata=_metadata())
    )
    expected = Path("tests/data/serialization/publication.md").read_bytes()
    assert serializer.serialize().text.encode("utf-8") == expected


def test_saved_book_offsets_address_the_bytes_on_disk(tmp_path):
    """The advertised offsets index the saved file, so the save must not rewrite
    newlines the way text mode does on Windows."""
    out = tmp_path / "book.md"
    serializer = MarkdownDocSerializer(
        doc=_book_document(), metadata_serializer=PubMetadataSerializer(metadata=_metadata())
    )
    out.write_bytes(serializer.serialize().text.encode("utf-8"))

    saved = out.read_bytes()
    parsed = yaml.safe_load(saved.decode("utf-8").split("---\n", maxsplit=2)[1])
    for chapter in parsed["chapters"]:
        heading = saved[chapter["byte"] :].split(b"\n", maxsplit=1)[0].decode("utf-8")
        assert heading.lstrip("# ") == chapter["title"]


@pytest.mark.parametrize("metadata", [None, PubMetadata(), PubMetadata(title="")])
def test_optional_publication_fields_are_omitted(metadata):
    serializer = MarkdownDocSerializer(
        doc=DoclingDocument(name="empty"), metadata_serializer=PubMetadataSerializer(metadata=metadata)
    )
    expected = {"chapters": []}
    if metadata is not None and metadata.title is not None:
        expected["title"] = metadata.title
    assert _frontmatter(serializer.serialize().text) == expected


def test_only_top_level_headings_are_chapters():
    doc = DoclingDocument(name="levels")
    doc.add_heading("Chapter", level=1)
    doc.add_heading("Section", level=2)
    serializer = MarkdownDocSerializer(doc=doc, metadata_serializer=PubMetadataSerializer())
    assert [entry["title"] for entry in _frontmatter(serializer.serialize().text)["chapters"]] == ["Chapter"]


def test_publication_body_and_spans_match_plain_markdown():
    doc = _book_document()
    plain = MarkdownDocSerializer(doc=doc).serialize()
    publication = MarkdownDocSerializer(doc=doc, metadata_serializer=PubMetadataSerializer()).serialize()
    assert publication.text.split("---\n", maxsplit=2)[2] == f"\n{plain.text}"
    assert publication.spans == plain.spans


def test_document_has_no_epub_export_methods():
    assert not hasattr(DoclingDocument, "export_to_epub")
    assert not hasattr(DoclingDocument, "save_as_epub")
