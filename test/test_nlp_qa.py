#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Test the pydantic models in module data_types.nlp.qa.py"""
import glob
import unittest

import pytest
from pydantic import ValidationError

from docling_core.types.nlp.qa import QAPair


class TestQAPair(unittest.TestCase):
    """Test QAPair model."""

    def test_qapair_read(self):
        """Validate data read from files."""
        for filename in glob.glob("test/data/nlp/qa-*.json"):
            try:
                with open(filename, encoding="utf-8") as file_obj:
                    file_json = file_obj.read()
                QAPair.model_validate_json(file_json)
            except ValidationError as e:
                print(f"Validation error in file {filename}", e.json())
                raise

    def test_qapair_wrong(self):
        """Validates wrong format from files."""
        filename = "test/data/nlp/error-qa-1.json"
        with (
            pytest.raises(ValidationError, match="Input should be a valid string"),
            open(filename, encoding="utf-8") as file_obj,
        ):
            file_json = file_obj.read()
            QAPair.model_validate_json(file_json)

        filename = "test/data/nlp/error-qa-3.json"
        with (
            pytest.raises(ValidationError, match="List must be unique"),
            open(filename, encoding="utf-8") as file_obj,
        ):
            file_json = file_obj.read()
            QAPair.model_validate_json(file_json)
