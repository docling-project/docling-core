"""Module for custom type validators."""

from collections.abc import Hashable
from datetime import datetime
from typing import Any, TypeVar

from pydantic_core import PydanticCustomError

T = TypeVar("T", bound=Hashable)


def ensure_unique_list(values: Any) -> Any:
    """Deduplicate a list while preserving order.

    `BeforeValidator` for list fields that should silently drop repeated values
    (as opposed to `validate_unique_list`, which raises on duplicates).

    Args:
        values: the value to validate; must be a list.

    Returns:
        The list with duplicates removed, keeping first occurrence order.
    """
    if not isinstance(values, list):
        raise ValueError("values must be a list of strings")
    return list(dict.fromkeys(values))


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
