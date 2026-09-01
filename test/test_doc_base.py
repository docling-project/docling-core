import pytest
from pydantic import ValidationError

from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin, GraphData, TableData, TrackSource


@pytest.mark.parametrize(
    "mimetype",
    [
        "application/pdf",
        "application/vnd.box.boxnote",
        "application/vnd.docling.ebcdic",
        "application/x-ebcdic",
        "text/markdown",
    ],
)
def test_document_origin_mimetype(mimetype: str):
    """Test that DocumentOrigin accepts the supported MIME types."""
    origin = DocumentOrigin(mimetype=mimetype, binary_hash=42, filename="test")
    assert origin.mimetype == mimetype


def test_document_origin_invalid_mimetype():
    """Test that DocumentOrigin rejects unknown MIME types."""
    with pytest.raises(ValidationError, match="is not a valid MIME type"):
        DocumentOrigin(mimetype="application/x-not-a-mimetype", binary_hash=42, filename="test")


def test_track_source():
    """Test the class TrackSource."""
    valid_track = TrackSource(
        start_time=11.0,
        end_time=12.0,
        identifier="test",
        voice="Mary",
    )

    assert valid_track
    assert valid_track.start_time == 11.0
    assert valid_track.end_time == 12.0
    assert valid_track.identifier == "test"
    assert valid_track.voice == "Mary"

    with pytest.raises(ValidationError, match="end_time"):
        TrackSource(start_time=11.0)

    with pytest.raises(ValidationError, match="should be a valid string"):
        TrackSource(
            start_time=11.0,
            end_time=12.0,
            voice=["Mary"],
        )

    with pytest.raises(ValidationError, match="must be greater than start"):
        TrackSource(
            start_time=11.0,
            end_time=11.0,
        )

    doc = DoclingDocument(name="Unknown")
    item = doc.add_text(text="Hello world", label=DocItemLabel.TEXT, source=valid_track)
    assert item.source
    assert len(item.source) == 1
    assert item.source[0] == valid_track


def test_add_doc_items_with_source():
    """Test that add_* helpers append sources to created DocItems."""
    source = TrackSource(start_time=11.0, end_time=12.0)
    doc = DoclingDocument(name="Unknown")

    items = [
        doc.add_list_item(text="List item", source=source),
        doc.add_text(text="Text", label=DocItemLabel.TEXT, source=source),
        doc.add_comment(text="Comment", source=source),
        doc.add_table(data=TableData(), source=source),
        doc.add_picture(source=source),
        doc.add_title(text="Title", source=source),
        doc.add_code(text="Code", source=source),
        doc.add_formula(text="Formula", source=source),
        doc.add_heading(text="Heading", source=source),
        doc.add_key_values(graph=GraphData(), source=source),
        doc.add_form(graph=GraphData(), source=source),
        doc.add_field_region(source=source),
        doc.add_field_heading(text="Field heading", source=source),
        doc.add_field_item(source=source),
        doc.add_field_key(text="Field key", source=source),
        doc.add_field_value(text="Field value", source=source),
        doc.add_field_hint(text="Field hint", source=source),
        doc.add_marker(text="Marker", source=source),
    ]

    for item in items:
        assert item.source == [source]


@pytest.mark.parametrize(
    "label",
    [
        DocItemLabel.TITLE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.CODE,
        DocItemLabel.FORMULA,
        DocItemLabel.FIELD_HEADING,
        DocItemLabel.FIELD_VALUE,
    ],
)
def test_add_text_dispatch_preserves_source(label: DocItemLabel):
    """Test that add_text preserves source when dispatching to specialized helpers."""
    source = TrackSource(start_time=11.0, end_time=12.0)
    doc = DoclingDocument(name="Unknown")

    item = doc.add_text(label=label, text="Hello world", source=source)

    assert item.source == [source]
