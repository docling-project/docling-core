#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Module for custom type validators."""
import json
import logging
from datetime import datetime
from importlib import resources
from typing import Hashable, TypeVar

import jsonschema
from pydantic_core import PydanticCustomError

logger = logging.getLogger("docling-core")

T = TypeVar("T", bound=Hashable)


def validate_schema(file_: dict, schema: dict) -> tuple[bool, str]:
    """Check wheter the workflow is properly formatted JSON and contains valid keys.

    Where possible, this also checks a few basic dependencies between properties, but
    this functionality is limited.
    """
    try:
        jsonschema.validate(file_, schema)
        return (True, "All good!")

    except jsonschema.ValidationError as err:
        return (False, err.message)


def validate_raw_schema(file_: dict) -> tuple[bool, str]:
    """Validate a RAW file."""
    logger.debug("validate RAW schema ... ")

    schema_txt = (
        resources.files("docling_core")
        .joinpath("resources/schemas/doc/RAW.json")
        .read_text("utf-8")
    )
    schema = json.loads(schema_txt)

    return validate_schema(file_, schema)


def validate_ann_schema(file_: dict) -> tuple[bool, str]:
    """Validate an annotated (ANN) file."""
    logger.debug("validate ANN schema ... ")

    schema_txt = (
        resources.files("docling_core")
        .joinpath("resources/schemas/doc/ANN.json")
        .read_text("utf-8")
    )
    schema = json.loads(schema_txt)

    return validate_schema(file_, schema)


def validate_ocr_schema(file_: dict) -> tuple[bool, str]:
    """Validate an OCR file."""
    logger.debug("validate OCR schema ... ")

    schema_txt = (
        resources.files("docling_core")
        .joinpath("resources/schemas/doc/OCR-output.json")
        .read_text("utf-8")
    )
    schema = json.loads(schema_txt)

    return validate_schema(file_, schema)


def validate_unique_list(v: list[T]) -> list[T]:
    """Validate that a list has unique values.

    Validator for list types, since pydantic V2 does not support the `unique_items`
    parameter from V1. More information on
    https://github.com/pydantic/pydantic-core/pull/820#issuecomment-1670475909

    Args:
        v: any list of hashable types

    Returns:
        The list, after checking for unique items.
    """
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


def validate_datetime(v, handler):
    """Validate that a value is a datetime or a non-numeric string."""
    if type(v) is datetime or (type(v) is str and not v.isnumeric()):
        return handler(v)
    else:
        raise ValueError("Value type must be a datetime or a non-numeric string")
