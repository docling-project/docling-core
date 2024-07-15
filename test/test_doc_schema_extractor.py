#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module data_types.ccs."""
from pydantic import ValidationError
import json

from docling_core.types.doc.document import (
    CCSDocument,
)

def test_ccs_document_update():
    """Validate data with CCSDocument extract."""
    filename = "test/data/doc/ext-1.json"
    try:
        with open(filename) as f:
            raw_doc = json.load(f)
            for item in raw_doc["main-text"]:
                if "$ref" in item:
                    assert False, f"$ref should not be in file {filename}"

            doc = CCSDocument.model_validate(raw_doc)

            if doc.description.abstract:
                assert False, f"Abstract should not be present"

    except ValidationError as e:
        print(f"Validation error in file {filename}:\n{e.json()}")
        raise
