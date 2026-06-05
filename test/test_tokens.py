"""Tests for location-token generation in DocumentToken."""

from docling_core.types.doc.tokens import DocumentToken
from docling_core.types.legacy_doc.tokens import (
    DocumentToken as LegacyDocumentToken,
)


def test_get_location_accepts_inverted_bbox():
    """A slightly inverted bbox must not crash get_location.

    Near-zero-height/width elements (e.g. a layout-model regression artifact)
    can produce bbox coordinates where the "min" corner is greater than the
    "max" corner. The method's own min()/max() normalization already orders the
    output, so such input must be tolerated rather than raising AssertionError.
    """
    page_w, page_h = 600.0, 800.0
    inverted = (100.0, 633.39, 110.0, 625.68)
    normalized = (100.0, 625.68, 110.0, 633.39)

    # Must not raise.
    inverted_loc = DocumentToken.get_location(bbox=inverted, page_w=page_w, page_h=page_h)
    normalized_loc = DocumentToken.get_location(bbox=normalized, page_w=page_w, page_h=page_h)

    # Inverted input yields the same correctly-ordered tokens as its
    # normalized counterpart.
    assert inverted_loc == normalized_loc


def test_get_location_inverted_x_axis():
    """Inversion on the x-axis is also tolerated and normalized."""
    page_w, page_h = 600.0, 800.0
    inverted = (110.0, 100.0, 100.0, 200.0)
    normalized = (100.0, 100.0, 110.0, 200.0)

    assert DocumentToken.get_location(bbox=inverted, page_w=page_w, page_h=page_h) == DocumentToken.get_location(
        bbox=normalized, page_w=page_w, page_h=page_h
    )


def test_legacy_get_location_accepts_inverted_bbox():
    """The legacy DocumentToken.get_location has the same contract."""
    page_w, page_h = 600.0, 800.0
    inverted = [100.0, 633.39, 110.0, 625.68]
    normalized = [100.0, 625.68, 110.0, 633.39]

    inverted_loc = LegacyDocumentToken.get_location(bbox=inverted, page_w=page_w, page_h=page_h)
    normalized_loc = LegacyDocumentToken.get_location(bbox=normalized, page_w=page_w, page_h=page_h)

    assert inverted_loc == normalized_loc
