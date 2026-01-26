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


def test_track_provenance():
    """Test the class TrackSource."""

    valid_track = TrackSource(
        start_time=11.0,
        end_time=12.0,
        identifier="test",
        tags = [
            {"name": "v", "annotation": "Mary", "classes": ["first", "loud"]},
            {"name": "lang", "annotation": "en"},
            {"name": "lang", "annotation": "en-GB"},
            {"name": "i", "classes": ["foreignphrase"]},
        ]
    )

    assert valid_track
    assert valid_track.start_time == 11.0
    assert valid_track.end_time == 12.0
    assert valid_track.identifier == "test"
    assert valid_track.tags
    assert valid_track.tags[0].annotation == "Mary"
    assert valid_track.tags[0].classes == ["first", "loud"]
    assert valid_track.tags[1].annotation == "en"
    assert valid_track.tags[2].annotation == "en-GB"
    assert valid_track.tags[3].classes == ["foreignphrase"]

    with pytest.raises(ValidationError, match="end_time"):
        TrackSource(start_time=11.0)

    with pytest.raises(ValidationError, match="should be a valid dictionary"):
        TrackSource(
            start_time=11.0,
            end_time=12.0,
            tags=["en"],
        )

    with pytest.raises(ValidationError, match="must be greater than start"):
        TrackSource(
            start_time=11.0,
            end_time=11.0,
        )

    doc = DoclingDocument(name="Unknown")
    item = doc.add_text(text="Hello world", label=DocItemLabel.TEXT)
    item.source = [valid_track]
    with pytest.raises(ValidationError, match="should be a valid list"):
        item.source = "Invalid source"
