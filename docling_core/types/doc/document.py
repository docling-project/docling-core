#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models for the Docling Document data type."""

from datetime import datetime
from typing import Generic, Optional, Union

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    NonNegativeInt,
    StrictStr,
    model_validator,
)

from docling_core.search.mapping import es_field
from docling_core.types.base import (
    Acquisition,
    CollectionDocumentInfo,
    CollectionNameTypeT,
    DescriptionAdvancedT,
    DescriptionAnalyticsT,
    FileInfoObject,
    Identifier,
    IdentifierTypeT,
    LanguageT,
    Log,
)
from docling_core.types.doc.base import (
    BaseCell,
    BaseText,
    BitmapObject,
    PageDimensions,
    PageReference,
    Ref,
    S3Data,
    Table,
)
from docling_core.utils.alias import AliasModel


class CCSFileInfoDescription(BaseModel, extra="forbid"):
    """File info description."""

    author: Optional[list[StrictStr]] = None
    keywords: Optional[str] = None
    subject: Optional[str] = None
    title: Optional[StrictStr] = None
    creation_date: Optional[str] = None  # datetime


class CCSFileInfoObject(FileInfoObject, extra="forbid"):
    """File info object."""

    num_pages: Optional[int] = Field(default=None, alias="#-pages")

    collection_name: Optional[str] = Field(
        default=None,
        alias="collection-name",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    description: Optional[CCSFileInfoDescription] = Field(
        default=None, json_schema_extra=es_field(suppress=True)
    )
    page_hashes: Optional[list[PageReference]] = Field(
        default=None, alias="page-hashes"
    )


class Affiliation(BaseModel, extra="forbid"):
    """Affiliation."""

    name: str = Field(
        ...,
        json_schema_extra=es_field(
            fields={
                "lower": {
                    "normalizer": "lowercase_asciifolding",
                    "type": "keyword",
                    "ignore_above": 8191,
                },
                "keyword": {"type": "keyword", "ignore_above": 8191},
            },
        ),
    )
    id: Optional[str] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    source: Optional[str] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )


class Author(BaseModel, extra="forbid"):
    """Author."""

    name: str = Field(
        ...,
        json_schema_extra=es_field(
            type="text",
            fields={
                "lower": {
                    "normalizer": "lowercase_asciifolding",
                    "type": "keyword",
                    "ignore_above": 8191,
                },
                "keyword": {"type": "keyword", "ignore_above": 8191},
            },
        ),
    )
    id: Optional[str] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    source: Optional[str] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    affiliations: Optional[list[Affiliation]] = None


class Publication(BaseModel, Generic[IdentifierTypeT], extra="forbid"):
    """Publication details of a journal or venue."""

    identifiers: Optional[list[Identifier[IdentifierTypeT]]] = Field(
        default=None,
        description="Unique identifiers of a publication venue.",
    )
    name: StrictStr = Field(
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
        description="Name of the publication.",
    )
    alternate_names: Optional[list[StrictStr]] = Field(
        default=None,
        json_schema_extra=es_field(type="text"),
        title="Alternate Names",
        description="Other names or abbreviations of this publication.",
    )
    type: Optional[list[StrictStr]] = Field(
        default=None,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
        description="Type of publication (journal article, conference, review,...).",
    )
    pages: Optional[StrictStr] = Field(
        default=None,
        json_schema_extra=es_field(type="text"),
        description="Page range in the publication.",
    )
    issue: Optional[StrictStr] = Field(
        default=None,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
        description="Publication issue (issue number).",
    )
    volume: Optional[StrictStr] = Field(
        default=None,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
        description="Publication volume.",
    )
    url: Optional[AnyHttpUrl] = Field(
        default=None,
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
        description="URL on the publication site.",
    )


class DescriptionLicense(BaseModel, extra="forbid"):
    """Licence in document description."""

    code: Optional[StrictStr] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    text: Optional[StrictStr] = None


class CCSDocumentDescription(
    AliasModel,
    Generic[
        DescriptionAdvancedT,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ],
):
    """Description in document."""

    title: Optional[StrictStr] = None
    abstract: Optional[list[StrictStr]] = None
    authors: Optional[list[Author]] = None
    affiliations: Optional[list[Affiliation]] = None
    subjects: Optional[list[str]] = Field(
        default=None,
        json_schema_extra=es_field(
            fields={"keyword": {"ignore_above": 8191, "type": "keyword"}}
        ),
    )
    keywords: Optional[list[str]] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    publication_date: Optional[datetime] = None
    languages: Optional[list[LanguageT]] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    license_: Optional[DescriptionLicense] = Field(default=None, alias="license")
    publishers: Optional[list[StrictStr]] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    url_refs: Optional[list[str]] = Field(
        default=None, json_schema_extra=es_field(type="keyword", ignore_above=8191)
    )
    references: Optional[list[Identifier[IdentifierTypeT]]] = None
    publication: Optional[list[Publication]] = Field(
        default=None, description="List of publication journals or venues."
    )
    reference_count: Optional[NonNegativeInt] = Field(
        default=None,
        title="Reference Count",
        description="Total number of documents referenced by this document.",
        json_schema_extra=es_field(type="integer"),
    )
    citation_count: Optional[NonNegativeInt] = Field(
        default=None,
        title="Citation Count",
        description=(
            "Total number of citations that this document has received (number "
            "of documents in whose bibliography this document appears)."
        ),
        json_schema_extra=es_field(type="integer"),
    )
    citation_date: Optional[datetime] = Field(
        default=None,
        title="Citation Count Date",
        description="Last update date of the citation count.",
    )
    advanced: Optional[DescriptionAdvancedT] = None
    analytics: Optional[DescriptionAnalyticsT] = None
    logs: list[Log]
    collection: Optional[CollectionDocumentInfo[CollectionNameTypeT]] = Field(
        default=None, description="The collection information of this document."
    )
    acquisition: Optional[Acquisition] = Field(
        default=None,
        description=(
            "Information on how the document was obtained, for data governance"
            " purposes."
        ),
    )


class MinimalDocument(
    AliasModel,
    Generic[
        DescriptionAdvancedT,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ],
):
    """Minimal model for a document."""

    name: StrictStr = Field(alias="_name")
    obj_type: StrictStr = Field("document", alias="type")
    description: CCSDocumentDescription[
        DescriptionAdvancedT,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ]
    file_info: FileInfoObject = Field(alias="file-info")
    main_text: Optional[list[Union[Ref, BaseText]]] = Field(
        default=None, alias="main-text"
    )
    figures: Optional[list[BaseCell]] = None
    tables: Optional[list[Table]] = None


class CCSDocument(
    MinimalDocument,
    Generic[
        DescriptionAdvancedT,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ],
):
    """Model for a CCS-generated document."""

    obj_type: StrictStr = Field("pdf-document", alias="type")
    bitmaps: Optional[list[BitmapObject]] = None
    equations: Optional[list[BaseCell]] = None
    footnotes: Optional[list[BaseText]] = None
    file_info: CCSFileInfoObject = Field(alias="file-info")
    main_text: Optional[list[Union[Ref, BaseText]]] = Field(
        default=None,
        alias="main-text",
    )
    page_dimensions: Optional[list[PageDimensions]] = Field(
        default=None, alias="page-dimensions"
    )
    page_footers: Optional[list[BaseText]] = Field(default=None, alias="page-footers")
    page_headers: Optional[list[BaseText]] = Field(default=None, alias="page-headers")
    s3_data: Optional[S3Data] = Field(default=None, alias="_s3_data")

    @model_validator(mode="before")
    @classmethod
    def from_dict(cls, data):
        """Validates and fixes the input data."""
        description_collection = data["description"].get("collection")
        if not description_collection:
            data["description"].setdefault("collection", {})

        data["description"]["collection"].setdefault("type", "Document")
        logs = data["description"].get("logs")
        if not logs:
            data["description"].setdefault("logs", [])

        abstract = data["description"].get("abstract")
        if abstract is not None and not isinstance(abstract, list):
            if isinstance(abstract, str):
                data["description"]["abstract"] = [abstract]
            else:
                data["description"].pop("abstract")

        for key in ["affiliations", "authors"]:
            descr = data["description"].get(key)
            if descr is not None and not isinstance(descr, list):
                if isinstance(descr, dict):
                    data["description"][key] = [descr]
                else:
                    data["description"].pop(key)

        if data.get("main-text"):
            for item in data["main-text"]:
                if ref := item.pop("__ref", None):
                    item["$ref"] = ref

        return data


class ExportedCCSDocument(
    MinimalDocument,
    Generic[
        DescriptionAdvancedT,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ],
):
    """Document model for Docling."""

    obj_type: StrictStr = Field(
        "pdf-document",
        alias="type",
        json_schema_extra=es_field(type="keyword", ignore_above=8191),
    )
    bitmaps: Optional[list[BitmapObject]] = None
    equations: Optional[list[BaseCell]] = None
    footnotes: Optional[list[BaseText]] = None
    description: CCSDocumentDescription[
        DescriptionAdvancedT,
        DescriptionAnalyticsT,
        IdentifierTypeT,
        LanguageT,
        CollectionNameTypeT,
    ]
    file_info: CCSFileInfoObject = Field(alias="file-info")
    main_text: Optional[list[Union[Ref, BaseText]]] = Field(
        default=None, alias="main-text"
    )
    page_dimensions: Optional[list[PageDimensions]] = Field(
        default=None, alias="page-dimensions"
    )
    page_footers: Optional[list[BaseText]] = Field(default=None, alias="page-footers")
    page_headers: Optional[list[BaseText]] = Field(default=None, alias="page-headers")
    s3_data: Optional[S3Data] = Field(default=None, alias="_s3_data")
    identifiers: Optional[list[Identifier[IdentifierTypeT]]] = None

    @model_validator(mode="before")
    @classmethod
    def from_dict(cls, data):
        """Fix ref in main-text."""
        if data.get("main-text"):
            for item in data["main-text"]:
                if ref := item.pop("__ref", None):
                    item["$ref"] = ref

        return data
