#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the model Record."""
from typing import Generic, Optional

from pydantic import BaseModel, Field, StrictStr

from docling_core.search.mapping import es_field
from docling_core.types.base import (
    Acquisition,
    CollectionNameTypeT,
    CollectionRecordInfo,
    FileInfoObject,
    Identifier,
    IdentifierTypeT,
    Log,
    PredicateKeyNameT,
    PredicateKeyTypeT,
    PredicateValueTypeT,
    StrictDateTime,
    SubjectNameTypeT,
    SubjectTypeT,
)
from docling_core.types.rec.attribute import Attribute
from docling_core.types.rec.base import Provenance, ProvenanceTypeT
from docling_core.types.rec.subject import Subject


class RecordDescription(BaseModel, Generic[CollectionNameTypeT]):
    """Additional record metadata, including optional collection-specific fields."""

    logs: list[Log] = Field(
        description="Logs that describe the ETL tasks applied to this record."
    )
    publication_date: Optional[StrictDateTime] = Field(
        default=None,
        title="Publication date",
        description=(
            "The date that best represents the last publication time of a record."
        ),
    )
    collection: Optional[CollectionRecordInfo[CollectionNameTypeT]] = Field(
        default=None, description="The collection information of this record."
    )
    acquisition: Optional[Acquisition] = Field(
        default=None,
        description=(
            "Information on how the document was obtained, for data governance"
            " purposes."
        ),
    )


class Record(
    Provenance,
    Generic[
        IdentifierTypeT,
        PredicateValueTypeT,
        PredicateKeyNameT,
        PredicateKeyTypeT,
        ProvenanceTypeT,
        SubjectTypeT,
        SubjectNameTypeT,
        CollectionNameTypeT,
    ],
):
    """A representation of a structured record in an database."""

    file_info: FileInfoObject = Field(alias="file-info")
    description: RecordDescription
    subject: Subject[IdentifierTypeT, SubjectTypeT, SubjectNameTypeT]
    attributes: Optional[
        list[
            Attribute[
                IdentifierTypeT,
                PredicateValueTypeT,
                PredicateKeyNameT,
                PredicateKeyTypeT,
                ProvenanceTypeT,
            ]
        ]
    ] = None
    name: Optional[StrictStr] = Field(
        default=None,
        description="A short description or summary of the record.",
        alias="_name",
        json_schema_extra=es_field(type="text"),
    )
    identifiers: Optional[list[Identifier[IdentifierTypeT]]] = Field(
        default=None,
        description="A list of unique identifiers of this record in a database.",
    )
