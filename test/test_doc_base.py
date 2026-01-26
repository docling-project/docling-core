import pytest
from pydantic import ValidationError

from docling_core.types.doc import DocItemLabel, DoclingDocument, TrackSource
from docling_core.types.legacy_doc.base import Prov, S3Reference


def test_s3_reference():
    """Validate data with Identifier model."""
    gold_dict = {"__ref_s3_data": "#/s3_data/figures/0"}
    data = S3Reference(__ref_s3_data="#/s3_data/figures/0")

    assert data.model_dump() == gold_dict
    assert data.model_dump(by_alias=True) == gold_dict

    with pytest.raises(ValidationError, match="required"):
        S3Reference()


def test_prov():
    prov = {
        "bbox": [
            48.19645328521729,
            644.2883926391602,
            563.6185592651367,
            737.4546043395997,
        ],
        "page": 2,
        "span": [0, 0],
    }

    assert Prov(**prov)

    with pytest.raises(ValidationError, match="valid integer"):
        prov["span"] = ["foo", 0]
        Prov(**prov)

    with pytest.raises(ValidationError, match="at least 2 items"):
        prov["span"] = [0]
        Prov(**prov)


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
