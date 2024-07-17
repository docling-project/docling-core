#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the model Predicate."""
from datetime import datetime
from typing import Annotated, Generic, Optional

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictStr,
    field_validator,
)

from docling_core.search.mapping import es_field
from docling_core.types.base import (
    Coordinates,
    PredicateKeyNameT,
    PredicateKeyTypeT,
    PredicateValueTypeT,
)
from docling_core.utils.alias import AliasModel


class NumericalValue(BaseModel, extra="forbid"):
    """Model for numerical values."""

    min: StrictFloat = Field(..., json_schema_extra=es_field(type="float"))
    max: StrictFloat = Field(..., json_schema_extra=es_field(type="float"))
    val: StrictFloat = Field(..., json_schema_extra=es_field(type="float"))
    err: StrictFloat = Field(..., json_schema_extra=es_field(type="float"))
    unit: StrictStr = Field(
        ..., json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )


class NominalValue(BaseModel, extra="forbid"):
    """Model for nominal (categorical) values."""

    value: StrictStr = Field(
        ..., json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )


class TextValue(BaseModel, extra="forbid"):
    """Model for textual values."""

    value: StrictStr = Field(..., json_schema_extra=es_field(type="text"))


class BooleanValue(BaseModel, extra="forbid"):
    """Model for boolean values."""

    value: StrictBool = Field(..., json_schema_extra=es_field(type="boolean"))


class DatetimeValue(BaseModel, extra="forbid"):
    """Model for datetime values."""

    value: datetime


class GeopointValue(BaseModel, extra="forbid"):
    """A representation of a geopoint (longitude and latitude coordinates)."""

    value: Coordinates
    conf: Optional[Annotated[float, Field(strict=True, ge=0.0, le=1.0)]] = Field(
        default=None, json_schema_extra=es_field(type="float")
    )

    @field_validator("value")
    @classmethod
    def validate_coordinates(cls, v):
        """Validate the reference field for indexes of type Document."""
        if abs(v[0]) > 180:
            raise ValueError("invalid longitude")
        if abs(v[1]) > 90:
            raise ValueError("invalid latitude")
        return v


class PredicateKey(
    AliasModel, Generic[PredicateKeyNameT, PredicateKeyTypeT], extra="forbid"
):
    """Model for the key (unique identifier) of a predicate."""

    name: PredicateKeyNameT = Field(
        description="Name of the predicate key.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    type_: PredicateKeyTypeT = Field(
        alias="type",
        title="Type",
        description="Type of predicate key.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )


class PredicateValue(AliasModel, Generic[PredicateValueTypeT], extra="forbid"):
    """Model for the value of a predicate."""

    name: StrictStr = Field(
        description="Name of the predicate value (actual value).",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    type_: PredicateValueTypeT = Field(
        alias="type",
        description="Type of predicate value.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )


class Predicate(
    AliasModel,
    Generic[PredicateValueTypeT, PredicateKeyNameT, PredicateKeyTypeT],
    extra="forbid",
):
    """Model for a predicate."""

    key: PredicateKey[PredicateKeyNameT, PredicateKeyTypeT]
    value: PredicateValue[PredicateValueTypeT]

    numerical_value: Optional[NumericalValue] = None
    numerical_value_si: Optional[NumericalValue] = None
    nominal_value: Optional[NominalValue] = None
    text_value: Optional[TextValue] = None
    boolean_value: Optional[BooleanValue] = None
    datetime_value: Optional[DatetimeValue] = None
    geopoint_value: Optional[GeopointValue] = None
