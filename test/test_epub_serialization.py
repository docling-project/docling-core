"""Test EPUB book Markdown serialization."""

import pytest
import yaml

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.epub import (
    EpubDocSerializer,
    EpubMetadata,
    EpubParams,
)
from docling_core.types.doc.base import BoundingBox, ImageRefMode
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel


def _metadata() -> EpubMetadata:
    return EpubMetadata(
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

    ser = EpubDocSerializer(doc=doc, params=EpubParams(metadata=_metadata()))
    parsed = _frontmatter(ser.serialize().text)

    assert parsed["title"] == "Alice's Adventures in Wonderland"
    assert parsed["authors"] == ["Lewis Carroll"]
    assert parsed["published"] == "2008-06-27"
    assert parsed["language"] == "en"
    assert parsed["source_file"] == "alice.epub"


def test_chapter_byte_offsets_seek_to_the_heading():
    """Each chapter byte offset lands on the first byte of its rendered heading."""
    ser = EpubDocSerializer(doc=_book_document(), params=EpubParams(metadata=_metadata()))
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
    """A spine document that titles its chapter with <h1> yields a TitleItem, not a
    level-1 section header, and must still appear in the chapter index."""
    doc = DoclingDocument(name="novel")
    doc.add_title("Chapter One")
    doc.add_text(DocItemLabel.TEXT, "It was a bright cold day in April.")
    doc.add_title("Chapter Two")
    doc.add_text(DocItemLabel.TEXT, "The Ministry of Truth.")

    ser = EpubDocSerializer(doc=doc, params=EpubParams(metadata=_metadata()))
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)

    assert [chapter["title"] for chapter in parsed["chapters"]] == ["Chapter One", "Chapter Two"]
    encoded = markdown.encode("utf-8")
    for chapter in parsed["chapters"]:
        assert encoded[chapter["byte"] :].startswith(f"# {chapter['title']}".encode())


def test_chapter_line_offsets_are_one_based():
    """Each chapter line offset indexes the heading line, counting from one."""
    ser = EpubDocSerializer(doc=_book_document(), params=EpubParams(metadata=_metadata()))
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)
    lines = markdown.splitlines()

    for chapter in parsed["chapters"]:
        assert lines[chapter["line"] - 1].lstrip("# ") == chapter["title"]


def test_chapter_offsets_account_for_page_break_replacement():
    """Page-break markers are replaced in the body, so offsets measure the replacement."""
    prov = [ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1)) for page in (1, 2)]
    doc = DoclingDocument(name="pages")
    doc.add_heading("One", level=1, prov=prov[0])
    doc.add_text(DocItemLabel.TEXT, "body", prov=prov[0])
    doc.add_heading("Two", level=1, prov=prov[1])

    ser = EpubDocSerializer(
        doc=doc,
        params=EpubParams(metadata=_metadata(), page_break_placeholder="<PB>"),
    )
    markdown = ser.serialize().text
    parsed = _frontmatter(markdown)

    assert "<PB>" in markdown
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

        ser = EpubDocSerializer(doc=doc, params=EpubParams(metadata=_metadata()))
        markdown = ser.serialize().text
        chapter = _frontmatter(markdown)["chapters"][0]

        assert markdown.encode("utf-8")[chapter["byte"] :].startswith(b"## Chapter")
        digit_widths.add(len(str(chapter["byte"])))

    assert len(digit_widths) > 1, "the body sizes must span several offset digit widths"


def test_frontmatter_size_change_is_rejected(monkeypatch):
    """A frontmatter size change would invalidate every advertised byte offset."""
    original_render = EpubDocSerializer._render_frontmatter
    render_count = 0

    def unstable_render(self, chapters):
        nonlocal render_count
        render_count += 1
        rendered = original_render(self, chapters)
        return f"{rendered} " if render_count == 2 else rendered

    monkeypatch.setattr(EpubDocSerializer, "_render_frontmatter", unstable_render)
    serializer = EpubDocSerializer(doc=_book_document(), params=EpubParams(metadata=_metadata()))

    with pytest.raises(ValueError, match="frontmatter changed size"):
        serializer.serialize()


def test_empty_serialization_parts_are_ignored():
    """An empty part must not add body separators or phantom chapters."""
    serializer = EpubDocSerializer(doc=DoclingDocument(name="empty"))

    result = serializer.serialize_doc(parts=[SerializationResult()])

    assert result.text == "---\nchapters: []\n---\n\n"


def test_book_without_headings_has_an_empty_chapter_list():
    """A book with no top-level headings still advertises a parseable empty list."""
    doc = DoclingDocument(name="empty")
    doc.add_text(DocItemLabel.TEXT, "No headings here.")

    ser = EpubDocSerializer(doc=doc, params=EpubParams(metadata=_metadata()))
    parsed = _frontmatter(ser.serialize().text)

    assert parsed["chapters"] == []


def test_export_to_epub_is_consistent_with_the_serializer():
    """export_to_epub() is consistent with EpubDocSerializer."""
    doc = _book_document()
    ser = EpubDocSerializer(doc=doc, params=EpubParams(metadata=_metadata()))

    assert doc.export_to_epub(metadata=_metadata()) == ser.serialize().text


def test_saved_book_offsets_address_the_bytes_on_disk(tmp_path):
    """The advertised offsets index the saved file, so the save must not rewrite
    newlines the way text mode does on Windows."""
    out = tmp_path / "book.md"
    _book_document().save_as_epub(str(out), metadata=_metadata())

    saved = out.read_bytes()
    parsed = yaml.safe_load(saved.decode("utf-8").split("---\n", maxsplit=2)[1])
    for chapter in parsed["chapters"]:
        heading = saved[chapter["byte"] :].split(b"\n", maxsplit=1)[0].decode("utf-8")
        assert heading.lstrip("# ") == chapter["title"]


def test_save_as_epub_creates_the_referenced_image_directory(tmp_path):
    """Referenced-image output creates its artifacts directory before serialization."""
    artifacts_dir = tmp_path / "images"

    _book_document().save_as_epub(
        tmp_path / "book.md",
        artifacts_dir=artifacts_dir,
        metadata=_metadata(),
        image_mode=ImageRefMode.REFERENCED,
    )

    assert artifacts_dir.is_dir()
