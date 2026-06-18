"""Define common models across types."""

from collections.abc import Hashable
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Final, TypeVar

from pydantic import (
    AfterValidator,
    Field,
    PlainSerializer,
    WrapValidator,
)

from docling_core.utils.validators import validate_datetime, validate_unique_list

# (subset of) JSON Pointer URI fragment id format, e.g. "#/main-text/84":
_JSON_POINTER_REGEX: Final[str] = r"^#(?:/([\w-]+)(?:/(\d+))?)?$"

VERSION_PATTERN: Final = (
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+"
    r"(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

T = TypeVar("T", bound=Hashable)

UniqueList = Annotated[
    list[T],
    AfterValidator(validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]

StrictDateTime = Annotated[
    datetime,
    WrapValidator(validate_datetime),
    PlainSerializer(lambda x: x.astimezone(tz=timezone.utc).isoformat(), return_type=str),
]


class CollectionTypeEnum(str, Enum):
    """Enumeration of valid Docling collection types."""

    document = "Document"
    record = "Record"
