"""Test the pydantic models in module data_types.base.py."""

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from docling_core.types.base import StrictDateTime


def test_strict_date_time() -> None:
    """Validate data with StrictDateTime model."""

    class Model(BaseModel):
        published: StrictDateTime

    # allowed formats
    Model(published=datetime.now(tz=timezone.utc))

    data = Model(published="2022-12-01T03:49:20.724435+00:00")
    assert data.published.isoformat() == "2022-12-01T03:49:20.724435+00:00"

    data = Model(published="2022-12-01T03:49:20.724435+03:00")
    assert data.published.isoformat() == "2022-12-01T03:49:20.724435+03:00"

    data = Model(published="2022-12-01T03:49:20.724435Z")
    assert data.published.isoformat() == "2022-12-01T03:49:20.724435+00:00"

    data = Model(published="2022-12-01T03:49:20")
    assert data.published.isoformat() == "2022-12-01T03:49:20"

    data = Model(published="2022-12-01")
    assert data.published.isoformat() == "2022-12-01T00:00:00"

    # invalid formats
    with pytest.raises(ValidationError, match="published"):
        Model(published="03:49:20")

    with pytest.raises(ValidationError, match="published"):
        Model(published=1679616000.0)
