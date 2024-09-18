#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models for the Docling Document data type."""

from datetime import datetime
from enum import Enum
from typing import Generic, Optional, Tuple, Union

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    NonNegativeInt,
    StrictStr,
    model_validator,
)
from tabulate import tabulate

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
    Figure,
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
    obj_type: Optional[StrictStr] = Field("document", alias="type")
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
    figures: Optional[list[Figure]] = None
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

    obj_type: Optional[StrictStr] = Field("pdf-document", alias="type")
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
        if not isinstance(data, dict):
            return data
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


class DocumentToken(Enum):
    """Class to represent an LLM friendly representation of a Document."""

    BEG_DOCUMENT = "<document>"
    END_DOCUMENT = "</document>"

    BEG_TITLE = "<title>"
    END_TITLE = "</title>"

    BEG_ABSTRACT = "<abstract>"
    END_ABSTRACT = "</abstract>"

    BEG_DOI = "<doi>"
    END_DOI = "</doi>"
    BEG_DATE = "<date>"
    END_DATE = "</date>"

    BEG_AUTHORS = "<authors>"
    END_AUTHORS = "</authors>"
    BEG_AUTHOR = "<author>"
    END_AUTHOR = "</author>"

    BEG_AFFILIATIONS = "<affiliations>"
    END_AFFILIATIONS = "</affiliations>"
    BEG_AFFILIATION = "<affiliation>"
    END_AFFILIATION = "</affiliation>"

    BEG_HEADER = "<section-header>"
    END_HEADER = "</section-header>"
    BEG_TEXT = "<text>"
    END_TEXT = "</text>"
    BEG_PARAGRAPH = "<paragraph>"
    END_PARAGRAPH = "</paragraph>"
    BEG_TABLE = "<table>"
    END_TABLE = "</table>"
    BEG_FIGURE = "<figure>"
    END_FIGURE = "</figure>"
    BEG_CAPTION = "<caption>"
    END_CAPTION = "</caption>"
    BEG_EQUATION = "<equation>"
    END_EQUATION = "</equation>"
    BEG_LIST = "<list>"
    END_LIST = "</list>"
    BEG_LISTITEM = "<list-item>"
    END_LISTITEM = "</list-item>"

    BEG_LOCATION = "<location>"
    END_LOCATION = "</location>"
    BEG_GROUP = "<group>"
    END_GROUP = "</group>"

    @classmethod
    def get_special_tokens(
        cls,
        max_rows: int = 100,
        max_cols: int = 100,
        max_pages: int = 1000,
        page_dimension: Tuple[int, int] = (100, 100),
    ):
        """Function to get all special document tokens."""
        special_tokens = [token.value for token in cls]

        # Adding dynamically generated row and col tokens
        for i in range(0, max_rows + 1):
            special_tokens += [f"<row_{i}>", f"</row_{i}>"]

        for i in range(0, max_cols + 1):
            special_tokens += [f"<col_{i}>", f"</col_{i}>"]

        for i in range(6):
            special_tokens += [f"<section-header-{i}>", f"</section-header-{i}>"]

        # Adding dynamically generated page-tokens
        for i in range(0, max_pages + 1):
            special_tokens.append(f"<page_{i}>")

        # Adding dynamically generated location-tokens
        for i in range(0, max(page_dimension[0] + 1, page_dimension[1] + 1)):
            special_tokens.append(f"<loc_{i}>")

        return special_tokens

    @staticmethod
    def get_page_token(page: int):
        """Function to get page tokens."""
        return f"<page_{page}>"

    @staticmethod
    def get_location_token(val: float, rnorm: int = 100):
        """Function to get location tokens."""
        val_ = round(rnorm * val)

        if val_ < 0:
            return "<loc_0>"

        if val_ > rnorm:
            return f"<loc_{rnorm}>"

        return f"<loc_{val_}>"


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

    obj_type: Optional[StrictStr] = Field(
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
        if not isinstance(data, dict):
            return data
        if data.get("main-text"):
            for item in data["main-text"]:
                if ref := item.pop("__ref", None):
                    item["$ref"] = ref

        return data

    def _resolve_ref(self, item: Ref) -> Optional[Union[BaseCell, BaseText]]:
        """Return the resolved reference.

        Resolved the Ref object within the document.
        If the object is not found, None is returned.
        """
        result: Optional[Union[BaseCell, BaseText]] = None

        # NOTE: currently only resolves refs explicitely, such that we can make
        # assumptions on ref parts
        if item.obj_type == "table" and self.tables:
            parts = item.ref.split("/")
            result = self.tables[int(parts[2])]
        elif item.obj_type == "figure" and self.figures:
            parts = item.ref.split("/")
            result = self.figures[int(parts[2])]
        elif item.obj_type == "equation" and self.equations:
            parts = item.ref.split("/")
            result = self.equations[int(parts[2])]
        elif item.obj_type == "footnote" and self.footnotes:
            parts = item.ref.split("/")
            result = self.footnotes[int(parts[2])]

        return result

    def export_to_markdown(
        self,
        delim: str = "\n\n",
        main_text_start: int = 0,
        main_text_stop: Optional[int] = None,
        main_text_labels: list[str] = [
            "title",
            "subtitle-level-1",
            "paragraph",
            "caption",
            "table",
        ],
        strict_text: bool = False,
    ) -> str:
        r"""Serialize to Markdown.

        Operates on a slice of the document's main_text as defined through arguments
        main_text_start and main_text_stop; defaulting to the whole main_text.

        Args:
            delim (str, optional): Delimiter to use when concatenating the various
                Markdown parts. Defaults to "\n\n".
            main_text_start (int, optional): Main-text slicing start index (inclusive).
                Defaults to 0.
            main_text_end (Optional[int], optional): Main-text slicing stop index
                (exclusive). Defaults to None.

        Returns:
            str: The exported Markdown representation.
        """
        has_title = False
        prev_text = ""
        md_texts: list[str] = []

        if self.main_text is not None:
            for orig_item in self.main_text[main_text_start:main_text_stop]:
                markdown_text = ""

                item = (
                    self._resolve_ref(orig_item)
                    if isinstance(orig_item, Ref)
                    else orig_item
                )
                if item is None:
                    continue

                item_type = item.obj_type
                if isinstance(item, BaseText) and item_type in main_text_labels:
                    text = item.text

                    # ignore repeated text
                    if prev_text == text:
                        continue
                    else:
                        prev_text = text

                    # first title match
                    if item_type == "title" and not has_title:
                        if strict_text:
                            markdown_text = f"{text}"
                        else:
                            markdown_text = f"# {text}"
                        has_title = True

                    # secondary titles
                    elif item_type in {"title", "subtitle-level-1"} or (
                        has_title and item_type == "title"
                    ):
                        if strict_text:
                            markdown_text = f"{text}"
                        else:
                            markdown_text = f"## {text}"

                    # normal text
                    else:
                        markdown_text = text

                elif (
                    isinstance(item, Table)
                    and item.data
                    and item_type in main_text_labels
                    and not strict_text
                ):
                    table = []
                    for row in item.data:
                        tmp = []
                        for col in row:
                            tmp.append(col.text)
                        table.append(tmp)

                    if len(table) > 1 and len(table[0]) > 0:
                        try:
                            md_table = tabulate(
                                table[1:], headers=table[0], tablefmt="github"
                            )
                        except ValueError:
                            md_table = tabulate(
                                table[1:],
                                headers=table[0],
                                tablefmt="github",
                                disable_numparse=True,
                            )

                        markdown_text = md_table

                if markdown_text:
                    md_texts.append(markdown_text)

        result = delim.join(md_texts)
        return result

    def export_to_document_tokens(
        self,
        delim: str = "\n\n",
        main_text_start: int = 0,
        main_text_stop: Optional[int] = None,
        main_text_labels: list[str] = [
            "title",
            "subtitle-level-1",
            "paragraph",
            "caption",
            "table",
            "figure",
        ],
        page_tagging: bool = True,
        location_tagging: bool = True,
        location_dimensions: Tuple[int, int] = (100, 100),
        add_new_line: bool = True,
    ) -> str:
        r"""Exports the document content to an DocumentToken format.

        Operates on a slice of the document's main_text as defined through arguments
        main_text_start and main_text_stop; defaulting to the whole main_text.

        Args:
            delim (str, optional): The delimiter used to separate text blocks in the
                exported XML. Default is two newline characters ("\n\n").
            main_text_start (int, optional): The starting index of the main text to
                be included in the XML. Default is 0 (the beginning of the text).
            main_text_stop (Optional[int], optional): The stopping index of the main
                text. If set to None, the export includes text up to the end.
                Default is None.
            main_text_labels (list[str], optional): A list of text labels that
                categorize the different sections of the document (e.g., "title",
                "subtitle-level-1", "paragraph", "caption"). Default labels are
                "title", "subtitle-level-1", "paragraph", and "caption".
            location_tagging (bool, optional): Determines whether to include
                location-based tagging in the XML. If True, the exported XML will
                contain information about the locations of the text elements.
                Default is True.
            location_dimensions (Tuple[int, int], optional): Specifies the dimensions
                (width and height) for the location tagging, if enabled.
                Default is [100, 100].
            add_new_line (bool, optional): Whether to add new line characters after
                each text block. If True, a new line is added after each block of
                text in the XML. Default is True.

        Returns:
            str: The content of the document formatted as an XML string.
        """
        xml_str = DocumentToken.BEG_DOCUMENT.value

        new_line = ""
        if add_new_line:
            new_line = "\n"

        if self.main_text is not None:
            for orig_item in self.main_text[main_text_start:main_text_stop]:

                item = (
                    self._resolve_ref(orig_item)
                    if isinstance(orig_item, Ref)
                    else orig_item
                )

                if item is None:
                    continue

                prov = item.prov

                loc_str = ""  # default is zero
                if (
                    location_tagging
                    and self.page_dimensions is not None
                    and prov is not None
                    and len(prov) > 0
                ):

                    page = prov[0].page
                    page_dim = self.page_dimensions[page - 1]

                    page_w = float(page_dim.width)
                    page_h = float(page_dim.height)

                    x0 = float(prov[0].bbox[0]) / float(page_w)
                    y0 = float(prov[0].bbox[1]) / float(page_h)
                    x1 = float(prov[0].bbox[2]) / float(page_w)
                    y1 = float(prov[0].bbox[3]) / float(page_h)

                    page_tok = ""
                    if page_tagging:
                        page_tok = DocumentToken.get_page_token(page=page)

                    x0_tok = DocumentToken.get_location_token(
                        val=min(x0, x1), rnorm=location_dimensions[0]
                    )
                    y0_tok = DocumentToken.get_location_token(
                        val=min(y0, y1), rnorm=location_dimensions[1]
                    )
                    x1_tok = DocumentToken.get_location_token(
                        val=max(x0, x1), rnorm=location_dimensions[0]
                    )
                    y1_tok = DocumentToken.get_location_token(
                        val=max(y0, y1), rnorm=location_dimensions[1]
                    )

                    # update
                    loc_str = f"{DocumentToken.BEG_LOCATION.value}"
                    loc_str += f"{page_tok}"
                    loc_str += f"{x0_tok}{y0_tok}{x1_tok}{y1_tok}"
                    loc_str += f"{DocumentToken.END_LOCATION.value}"

                item_type = item.obj_type
                if isinstance(item, BaseText) and (item_type in main_text_labels):
                    text = item.text

                    xml_str += f"<{item_type}>{loc_str}{text}</{item_type}>{new_line}"

                elif isinstance(item, Table) and (item_type in main_text_labels):

                    xml_str += f"<{item_type}>{loc_str}"

                    if item.text is not None and len(item.text) > 0:
                        xml_str += f"{DocumentToken.BEG_CAPTION.value}"
                        xml_str += (
                            f"{item.text}{DocumentToken.END_CAPTION.value}{new_line}"
                        )

                    if item.data is not None and len(item.data) > 0:
                        for i, row in enumerate(item.data):
                            xml_str += f"<row_{i}>"
                            for j, col in enumerate(row):
                                text = col.text
                                xml_str += f"<col_{j}>{text}</col_{j}>"

                            xml_str += f"</row_{i}>{new_line}"

                    xml_str += f"</{item_type}>{new_line}"

                elif isinstance(item, Figure) and (item_type in main_text_labels):

                    xml_str += f"<{item_type}>{loc_str}"

                    if item.text is not None and len(item.text) > 0:
                        xml_str += f"{DocumentToken.BEG_CAPTION.value}"
                        xml_str += (
                            f"{item.text}{DocumentToken.END_CAPTION.value}{new_line}"
                        )

                    xml_str += f"</{item_type}>{new_line}"

        xml_str += DocumentToken.END_DOCUMENT.value

        return xml_str
