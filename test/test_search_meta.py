#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module search.metadata.py."""
import glob
import os
from typing import Literal

from pydantic import ValidationError

from docling_core.search.meta import Meta


def test_meta():
    """Validate data with Meta schema."""
    taxonomy = Literal["Public", "PI"]
    domain = Literal[
        "Science", "Technology", "History", "Art", "Literature", "Geography"
    ]

    for filename in glob.glob("test/data/search/meta-*.json"):
        try:
            with open(filename, encoding="utf-8") as file_obj:
                file_json = file_obj.read()
            Meta[taxonomy, domain].model_validate_json(file_json)
        except ValidationError as e:
            print(f"Validation error in file {filename}", e.json())
            raise

    # test invalid documents
    gold_errors = {
        "error-meta-01.json": ["type", "version"],
        "error-meta-02.json": ["version", "domain", "$ref"],
        "error-meta-03.json": ["source", "extra"],
    }

    for filename in glob.glob("test/data/search/error-meta-*.json"):
        gold = gold_errors[os.path.basename(filename)]
        try:
            with open(filename, encoding="utf-8") as file_obj:
                file_json = file_obj.read()
            Meta[taxonomy, domain].model_validate_json(file_json)
            assert False, f"File {filename} should be an invalid metadata"
        except ValidationError as e:
            errors = e.errors()
            assert len(errors) == len(gold), f"Wrong number of errors in {filename}"
            assert all(errors[zdx]["loc"][0] == gold[zdx] for zdx in range(len(errors)))
