#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the model Attribute."""
from typing import Generic, Optional

from pydantic import Field
from typing_extensions import Annotated

from docling_core.search.mapping import es_field
from docling_core.types.base import (
    IdentifierTypeT,
    PredicateKeyNameT,
    PredicateKeyTypeT,
    PredicateValueTypeT,
    ProvenanceTypeT,
)
from docling_core.types.rec.base import ProvenanceItem
from docling_core.types.rec.predicate import Predicate
from docling_core.utils.alias import AliasModel


class Attribute(
    AliasModel,
    Generic[
        IdentifierTypeT,
        PredicateValueTypeT,
        PredicateKeyNameT,
        PredicateKeyTypeT,
        ProvenanceTypeT,
    ],
    extra="forbid",
):
    """Attribute model that describes a list of characteristics."""

    conf: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)] = (
        Field(
            ...,
            title="Confidence",
            description="The confidence level of this attribute characteristics.",
            json_schema_extra=es_field(type="float"),
        )
    )

    prov: Optional[list[ProvenanceItem[IdentifierTypeT, ProvenanceTypeT]]] = Field(
        default=None,
        title="Provenance",
        description="The sources of this attribute characteristics.",
    )

    predicates: list[
        Predicate[PredicateValueTypeT, PredicateKeyNameT, PredicateKeyTypeT]
    ] = Field(..., description="A list of characteristics (type, value, and name).")
