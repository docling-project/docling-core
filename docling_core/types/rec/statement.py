#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the model Statement."""
from typing import Generic

from pydantic import Field

from docling_core.types.base import (
    IdentifierTypeT,
    PredicateKeyNameT,
    PredicateKeyTypeT,
    PredicateValueTypeT,
    ProvenanceTypeT,
    SubjectNameTypeT,
    SubjectTypeT,
)
from docling_core.types.rec.attribute import Attribute
from docling_core.types.rec.subject import Subject


class Statement(
    Attribute,
    Generic[
        IdentifierTypeT,
        PredicateValueTypeT,
        PredicateKeyNameT,
        PredicateKeyTypeT,
        ProvenanceTypeT,
        SubjectTypeT,
        SubjectNameTypeT,
    ],
    extra="allow",
):
    """A representation of a statement on a subject."""

    subject: Subject[IdentifierTypeT, SubjectTypeT, SubjectNameTypeT] = Field(
        description="The subject (entity) of this statement."
    )
