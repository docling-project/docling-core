import pytest
from pydantic import ValidationError

from docling_core.types.doc import TrackProvenance
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
    """Test the class TrackProvenance."""

    valid_track = TrackProvenance(
        start_time=11.0,
        end_time=12.0,
        identifier="test",
        voice="Mary",
        languages=["en", "en-GB"],
        classes=["v.first.loud", "i.foreignphrase"],
    )

    assert valid_track
    assert valid_track.start_time == 11.0
    assert valid_track.end_time == 12.0
    assert valid_track.identifier == "test"
    assert valid_track.voice == "Mary"
    assert valid_track.languages == ["en", "en-GB"]
    assert valid_track.classes == ["v.first.loud", "i.foreignphrase"]

    with pytest.raises(ValidationError, match="end_time"):
        TrackProvenance(start_time=11.0)

    with pytest.raises(ValidationError, match="should be a valid list"):
        TrackProvenance(
            start_time=11.0,
            end_time=12.0,
            languages="en",
        )

    with pytest.raises(ValidationError, match="must be greater than start"):
        TrackProvenance(
            start_time=11.0,
            end_time=11.0,
        )
