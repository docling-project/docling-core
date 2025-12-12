import pytest
from pydantic import ValidationError

from docling_core.types.doc.document import ProvenanceTrack
from docling_core.types.doc.webvtt import WebVTTTimestamp
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


def test_prov_track():
    """Test the class ProvenanceTrack."""

    valid_track = ProvenanceTrack(
        start_time=WebVTTTimestamp(raw="00:11.000"),
        end_time=WebVTTTimestamp(raw="00:12.000"),
        identifier="test",
        voice="Mary",
        languages=["en", "en-GB"],
        classes=["v.first.loud", "i.foreignphrase"],
    )

    assert valid_track
    assert valid_track.start_time == WebVTTTimestamp(raw="00:11.000")
    assert valid_track.end_time == WebVTTTimestamp(raw="00:12.000")
    assert valid_track.identifier == "test"
    assert valid_track.voice == "Mary"
    assert valid_track.languages == ["en", "en-GB"]
    assert valid_track.classes == ["v.first.loud", "i.foreignphrase"]

    with pytest.raises(ValidationError, match="end_time"):
        ProvenanceTrack(start_time=WebVTTTimestamp(raw="00:11.000"))

    with pytest.raises(ValidationError, match="should be a valid list"):
        ProvenanceTrack(
            start_time=WebVTTTimestamp(raw="00:11.000"),
            end_time=WebVTTTimestamp(raw="00:12.000"),
            languages="en",
        )
