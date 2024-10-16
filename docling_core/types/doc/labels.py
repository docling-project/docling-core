"""Models for the labels types."""

from enum import Enum


class DocItemLabel(str, Enum):
    """DocItemLabel."""

    # DocLayNet v2
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PICTURE = "picture"
    SECTION_HEADER = "section_header"
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    DOCUMENT_INDEX = "document_index"
    CODE = "code"
    CHECKBOX_SELECTED = "checkbox_selected"
    CHECKBOX_UNSELECTED = "checkbox_unselected"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"

    # Additional labels for markup-based formats (e.g. HTML, Word)
    PARAGRAPH = "paragraph"  # explicitly a paragraph and not arbitrary text
    REFERENCE = "reference"


class GroupLabel(str, Enum):
    """GroupLabel."""

    UNSPECIFIED = "unspecified"
    LIST = (
        "list"  # group label for list container (not the list-items) (e.g. HTML <ul/>)
    )
    ORDERED_LIST = "ordered_list"  # List with enumeration (e.g. HTML <ol/>)
    CHAPTER = "chapter"
    SECTION = "section"
    SHEET = "sheet"
    SLIDE = "slide"


class TableCellLabel(str, Enum):
    """TableCellLabel."""

    COLUMN_HEADER = "col_header"
    ROW_HEADER = "row_header"
    ROW_SECTION = "row_section"
    BODY = "body"
