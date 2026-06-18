import pytest
from pydantic import ValidationError

from docling_core.types.doc import DocItemLabel, DoclingDocument, TrackSource


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
