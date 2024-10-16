#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the model Subject."""
from typing import Generic, Optional

from pydantic import Field, StrictStr

from docling_core.search.mapping import es_field
from docling_core.types.base import (
    Identifier,
    IdentifierTypeT,
    SubjectNameTypeT,
    SubjectTypeT,
)
from docling_core.types.legacy_doc.base import S3Reference
from docling_core.utils.alias import AliasModel


class SubjectNameIdentifier(Identifier[SubjectNameTypeT], Generic[SubjectNameTypeT]):
    """Identifier of subject names.""" ""


class Subject(
    AliasModel,
    Generic[IdentifierTypeT, SubjectTypeT, SubjectNameTypeT],
    extra="forbid",
):
    """A representation of a subject."""

    display_name: StrictStr = Field(
        title="Display Name",
        description=(
            "Name of the subject in natural language. It can be used for end-user "
            "applications to display a human-readable name. For instance, `B(2) Mg(1)` "
            "for `MgB2` or `International Business Machines` for `IBM`"
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    display_image: Optional[S3Reference] = Field(
        default=None,
        title="Display Image",
        description=(
            "Image representing the subject. It can be used for end-user applications."
            "For example, the chemical structure drawing of a compound "
            "or the eight bar IBM logo for IBM."
        ),
        json_schema_extra=es_field(suppress=True),
    )
    type_: SubjectTypeT = Field(
        alias="type",
        description=(
            "Main subject type. For instance, `material`, `material-class`, "
            "`material-device`, `company`, or `person`."
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    names: list[SubjectNameIdentifier[SubjectNameTypeT]] = Field(
        description=(
            "List of given names for this subject. They may not be unique across "
            "different subjects."
        )
    )
    identifiers: Optional[list[Identifier[IdentifierTypeT]]] = Field(
        default=None,
        description=(
            "List of unique identifiers in database. For instance, the `PubChem ID` "
            "of a record in the PubChem database."
        ),
    )
    labels: Optional[list[StrictStr]] = Field(
        default=None,
        description="List of labels or categories for this subject.",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
