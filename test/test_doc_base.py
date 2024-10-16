#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#


import pytest
from pydantic import ValidationError

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
