#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the base models for the Record type."""
from typing import Generic, List, Optional

from pydantic import Field, StrictInt, StrictStr
from typing_extensions import Annotated

from docling_core.search.mapping import es_field
from docling_core.types.base import Identifier, IdentifierTypeT, ProvenanceTypeT
from docling_core.utils.alias import AliasModel


class ProvenanceItem(
    AliasModel, Generic[IdentifierTypeT, ProvenanceTypeT], extra="forbid"
):
    """A representation of an object provenance."""

    type_: Optional[ProvenanceTypeT] = Field(
        default=None,
        alias="type",
        title="The provenance type",
        description=(
            "Any string representing the type of provenance, e.g. `sentence`, "
            "`table`, or `doi`."
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )

    text: Optional[StrictStr] = Field(
        default=None,
        title="Evidence of the provenance",
        description=(
            "A text representing the evidence of the provenance, e.g. the sentence "
            "text or the content of a table cell"
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )

    reference: Optional[Identifier[IdentifierTypeT]] = Field(
        default=None,
        title="Reference to the provenance object",
        description=(
            "Reference to another object, e.g. record, statement, URL, or any other "
            "object that identifies the provenance"
        ),
    )

    path: Optional[StrictStr] = Field(
        default=None,
        title="The location of the provenance within the referenced object",
        description=(
            "A path that locates the evidence within the provenance object identified "
            "by the `reference` field using a JSON pointer notation, e.g., "
            "`#/main-text/5` to locate the `main-text` paragraph at index 5"
        ),
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )

    span: Optional[Annotated[List[StrictInt], Field(min_length=2, max_length=2)]] = (
        Field(
            default=None,
            title="The location of the item in the text/table",
            description=(
                "location of the item in the text/table referenced by the `path`,"
                " e.g., `[34, 67]`"
            ),
        )
    )


class Provenance(AliasModel, Generic[IdentifierTypeT, ProvenanceTypeT]):
    """A representation of an evidence, as a list of provenance objects."""

    conf: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] = Field(
        ...,
        title="The confidence of the evidence",
        description=(
            "This value represents a score to the data item. Items originating from "
            " databases will typically have a score 1.0, while items resulting from "
            " an NLP model may have a value between 0.0 and 1.0."
        ),
        json_schema_extra=es_field(type="float"),
    )
    prov: list[ProvenanceItem[IdentifierTypeT, ProvenanceTypeT]] = Field(
        title="Provenance", description="A list of provenance items."
    )
