#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models and methods to define the metadata fields in database index mappings."""
from pathlib import Path
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, StrictStr, ValidationInfo, field_validator

from docling_core.search.package import Package
from docling_core.types.base import CollectionTypeEnum, StrictDateTime, UniqueList
from docling_core.utils.alias import AliasModel

ClassificationT = TypeVar("ClassificationT", bound=str)
DomainT = TypeVar("DomainT", bound=str)


class S3Path(BaseModel, extra="forbid"):
    """The path details within a cloud object storage for CCS-parsed files."""

    bucket: StrictStr
    prefix: StrictStr
    infix: StrictStr

    def __hash__(self):
        """Return the hash value for this S3Path object."""
        return hash((type(self),) + tuple(self.__dict__.values()))


class S3CcsData(BaseModel, extra="forbid"):
    """The access details to a cloud object storage for CCS-parsed files."""

    endpoint: StrictStr
    paths: UniqueList[S3Path] = Field(min_length=1)


class DocumentLicense(BaseModel, extra="forbid"):
    """Document license for a search database index within the index mappings."""

    code: Optional[list[StrictStr]] = None
    text: Optional[list[StrictStr]] = None


class Meta(AliasModel, Generic[ClassificationT, DomainT], extra="forbid"):
    """Metadata of a search database index within the index mappings."""

    aliases: Optional[list[StrictStr]] = None
    created: StrictDateTime
    description: Optional[StrictStr] = None
    source: StrictStr
    storage: Optional[StrictStr] = None
    display_name: Optional[StrictStr] = None
    type: CollectionTypeEnum
    classification: Optional[list[ClassificationT]] = None
    version: UniqueList[Package] = Field(min_length=1)
    license: Optional[StrictStr] = None
    filename: Optional[Path] = None
    domain: Optional[list[DomainT]] = None
    reference: Optional[StrictStr] = Field(default=None, alias="$ref")
    ccs_s3_data: Optional[S3CcsData] = None
    document_license: Optional[DocumentLicense] = None
    index_key: Optional[StrictStr] = None
    project_key: Optional[StrictStr] = None

    @field_validator("reference")
    @classmethod
    def reference_for_document(cls, v, info: ValidationInfo):
        """Validate the reference field for indexes of type Document."""
        if "type" in info.data and info.data["type"] == "Document":
            if v and v != "ccs:schemas#/Document":
                raise ValueError("wrong reference value for Document type")
            else:
                return "ccs:schemas#/Document"
        else:
            return v

    @field_validator("version")
    @classmethod
    def version_has_schema(cls, v):
        """Validate that the docling-core library is always set in version field."""
        docling_core = [item for item in v if item.name == "docling-core"]
        if not docling_core:
            raise ValueError(
                "the version should include at least a valid docling-core package"
            )
        elif len(docling_core) > 1:
            raise ValueError(
                "the version must not include more than 1 docling-core package"
            )
        else:
            return v
