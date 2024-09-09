#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the model Statement."""
from enum import Enum
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


class StatementToken(Enum):
    """Class to represent an LLM friendly representation of statements."""

    BEG_STATEMENTS = "<statements>"
    END_STATEMENTS = "</statements>"

    BEG_STATEMENT = "<statement>"
    END_STATEMENT = "</statement>"

    BEG_PROV = "<prov>"
    END_PROV = "</prov>"

    BEG_SUBJECT = "<subject>"
    END_SUBJECT = "</subject>"

    BEG_PREDICATE = "<predicate>"
    END_PREDICATE = "</predicate>"

    BEG_PROPERTY = "<property>"
    END_PROPERTY = "</property>"

    BEG_VALUE = "<value>"
    END_VALUE = "</value>"

    BEG_UNIT = "<unit>"
    END_UNIT = "</unit>"

    @classmethod
    def get_special_tokens(cls):
        """Function to get all special statements tokens."""
        return [token.value for token in cls]


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
