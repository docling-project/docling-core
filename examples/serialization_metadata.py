"""Add optional YAML front matter when serializing Markdown."""

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, YAMLMetadataSerializer
from docling_core.transforms.serializer.publication import PubMetadata, PubMetadataSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument


def main():
    """Show generic metadata and publication metadata with a chapter index."""
    doc = DoclingDocument(name="book")
    doc.add_title("My book")
    doc.add_heading("Chapter one", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Once upon a time...")

    generic = MarkdownDocSerializer(
        doc=doc,
        metadata_serializer=YAMLMetadataSerializer(metadata={"title": "My book", "tags": ["fiction"]}),
    )
    print(generic.serialize().text)

    publication = MarkdownDocSerializer(
        doc=doc,
        metadata_serializer=PubMetadataSerializer(metadata=PubMetadata(title="My book", authors=["A. Writer"])),
    )
    print(publication.serialize().text)
    # Chapter positions index the complete UTF-8 output. To preserve offsets on
    # disk, use Path("book.md").write_bytes(publication.serialize().text.encode("utf-8")).


if __name__ == "__main__":
    main()
