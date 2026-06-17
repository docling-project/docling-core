"""Test the pydantic models in module types."""

import pytest
from pydantic import ValidationError

from docling_core.types import Generic


def test_generic():
    """Test the Generic model."""
    input_generic_0 = {
        "file-info": {
            "filename": "abc.xml",
            "filename-prov": "abc.xml.zip",
            "document-hash": "123457889",
        },
        "_name": "The ABC legacy_doc",
        "custom": ["The custom ABC content 1.", "The custom ABC content 2."],
    }
    Generic.model_validate(input_generic_0)

    input_generic_1 = {
        "file-info": {"filename": "abc.xml", "document-hash": "123457889"},
        "_name": "The ABC legacy_doc",
    }
    Generic.model_validate(input_generic_1)

    input_generic_2 = {
        "_name": "The ABC legacy_doc",
        "custom": ["The custom ABC content 1.", "The custom ABC content 2."],
    }
    with pytest.raises(ValidationError):
        Generic.model_validate(input_generic_2)
