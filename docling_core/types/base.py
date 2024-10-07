#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define common models across types."""
from datetime import datetime, timezone
from enum import Enum
from typing import Final, Generic, Hashable, List, Literal, Optional, TypeVar

from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    Field,
    PlainSerializer,
    StrictStr,
    StringConstraints,
    ValidationInfo,
    WrapValidator,
    field_validator,
)
from pydantic.types import NonNegativeInt
from typing_extensions import Annotated

from docling_core.search.mapping import es_field
from docling_core.search.package import VERSION_PATTERN
from docling_core.utils.alias import AliasModel
from docling_core.utils.validators import validate_datetime, validate_unique_list

# (subset of) JSON Pointer URI fragment id format, e.g. "#/main-text/84":
_JSON_POINTER_REGEX: Final[str] = r"^#(?:/([\w-]+)(?:/(\d+))?)?$"

LanguageT = TypeVar("LanguageT", bound=str)
IdentifierTypeT = TypeVar("IdentifierTypeT", bound=str)
DescriptionAdvancedT = TypeVar("DescriptionAdvancedT", bound=BaseModel)
DescriptionAnalyticsT = TypeVar("DescriptionAnalyticsT", bound=BaseModel)
SubjectTypeT = TypeVar("SubjectTypeT", bound=str)
SubjectNameTypeT = TypeVar("SubjectNameTypeT", bound=str)
PredicateValueTypeT = TypeVar("PredicateValueTypeT", bound=str)
PredicateKeyNameT = TypeVar("PredicateKeyNameT", bound=str)
PredicateKeyTypeT = TypeVar("PredicateKeyTypeT", bound=str)
ProvenanceTypeT = TypeVar("ProvenanceTypeT", bound=str)
CollectionNameTypeT = TypeVar("CollectionNameTypeT", bound=str)
Coordinates = Annotated[
    list[float],
    Field(min_length=2, max_length=2, json_schema_extra=es_field(type="geo_point")),
]
T = TypeVar("T", bound=Hashable)

UniqueList = Annotated[
    List[T],
    AfterValidator(validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]

StrictDateTime = Annotated[
    datetime,
    WrapValidator(validate_datetime),
    PlainSerializer(
        lambda x: x.astimezone(tz=timezone.utc).isoformat(), return_type=str
    ),
]

ACQUISITION_TYPE = Literal[
    "API", "FTP", "Download", "Link", "Web scraping/Crawling", "Other"
]


class Identifier(AliasModel, Generic[IdentifierTypeT], extra="forbid"):
    """Unique identifier of a Docling data object."""

    type_: IdentifierTypeT = Field(
        alias="type",
        description=(
            "A string representing a collection or database that contains this "
            "data object."
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    value: StrictStr = Field(
        description=(
            "The identifier value of the data object within a collection or database."
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    name: str = Field(
        alias="_name",
        title="_Name",
        description=(
            "A unique identifier of the data object across Docling, consisting of "
            "the concatenation of type and value in lower case, separated by hash "
            "(#)."
        ),
        pattern=r"^.+#.+$",
        strict=True,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )

    @field_validator("name")
    @classmethod
    def name_from_type_value(cls, v, info: ValidationInfo):
        """Validate the reference field for indexes of type Document."""
        if (
            "type_" in info.data
            and "value" in info.data
            and v != f"{info.data['type_'].lower()}#{info.data['value'].lower()}"
        ):
            raise ValueError(
                "the _name field must be the concatenation of type and value in lower "
                "case, separated by hash (#)"
            )
        return v


class Log(AliasModel, extra="forbid"):
    """Log entry to describe an ETL task on a document."""

    task: Optional[StrictStr] = Field(
        default=None,
        description=(
            "An identifier of this task. It may be used to identify this task from "
            "other tasks of the same agent and type."
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    agent: StrictStr = Field(
        description="The Docling agent that performed the task, e.g., CCS or CXS.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    type_: StrictStr = Field(
        alias="type",
        description="A task category.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    comment: Optional[StrictStr] = Field(
        default=None,
        description="A description of the task or any comments in natural language.",
    )
    date: StrictDateTime = Field(
        description=(
            "A string representation of the task execution datetime in ISO 8601 format."
        )
    )


class FileInfoObject(AliasModel):
    """Filing information for any data object to be stored in a Docling database."""

    filename: StrictStr = Field(
        description="The name of a persistent object that created this data object",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    fileprov: Optional[StrictStr] = Field(
        default=None,
        description=(
            "The provenance of this data object, e.g. an archive file, a URL, or any"
            " other repository."
        ),
        alias="filename-prov",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    document_hash: StrictStr = Field(
        description=(
            "A unique identifier of this data object within a collection of a "
            "Docling database"
        ),
        alias="document-hash",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )


class CollectionTypeEnum(str, Enum):
    """Enumeration of valid Docling collection types."""

    generic = "Generic"
    document = "Document"
    record = "Record"


CollectionTypeT = TypeVar("CollectionTypeT", bound=CollectionTypeEnum)


class CollectionInfo(
    BaseModel, Generic[CollectionNameTypeT, CollectionTypeT], extra="forbid"
):
    """Information of a collection."""

    name: Optional[CollectionNameTypeT] = Field(
        default=None,
        description="Name of the collection.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    type: CollectionTypeT = Field(
        ...,
        description="The collection type.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    version: Optional[
        Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)]
    ] = Field(
        default=None,
        description="The version of this collection model.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    alias: Optional[list[StrictStr]] = Field(
        default=None,
        description="A list of tags (aliases) for the collection.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )


class CollectionDocumentInfo(
    CollectionInfo[CollectionNameTypeT, Literal[CollectionTypeEnum.document]],
    Generic[CollectionNameTypeT],
    extra="forbid",
):
    """Information of a collection of type Document."""


class CollectionRecordInfo(
    CollectionInfo[CollectionNameTypeT, Literal[CollectionTypeEnum.record]],
    Generic[CollectionNameTypeT],
    extra="forbid",
):
    """Information of a collection of type Record."""


class Acquisition(BaseModel, extra="forbid"):
    """Information on how the data was obtained."""

    type: ACQUISITION_TYPE = Field(
        description="The method to obtain the data.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    date: Optional[StrictDateTime] = Field(
        default=None,
        description=(
            "A string representation of the acquisition datetime in ISO 8601 format."
        ),
    )
    link: Optional[AnyUrl] = Field(
        default=None,
        description="Link to the data source of this document.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    size: Optional[NonNegativeInt] = Field(
        default=None,
        description="Size in bytes of the raw document from the data source.",
        json_schema_extra=es_field(type="long"),
    )
