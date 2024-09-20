#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Tokens used in the docling document model."""

from enum import Enum
from typing import Annotated, Tuple

from pydantic import Field


class TableToken(Enum):
    """Class to represent an LLM friendly representation of a Table."""

    CELL_LABEL_COLUMN_HEADER = "<column_header>"
    CELL_LABEL_ROW_HEADER = "<row_header>"
    CELL_LABEL_SECTION_HEADERE = "<section_header>"
    CELL_LABEL_DATA = "<data>"

    OTSL_ECEL = "<ecel>"  # empty cell
    OTSL_FCEL = "<fcel>"  # cell with content
    OTSL_LCEL = "<lcel>"  # left looking cell,
    OTSL_UCEL = "<ucel>"  # up looking cell,
    OTSL_XCEL = "<xcel>"  # 2d extension cell (cross cell),
    OTSL_NL = "<nl>"  # new line,
    OTSL_CHED = "<ched>"  # - column header cell,
    OTSL_RHED = "<rhed>"  # - row header cell,
    OTSL_SROW = "<srow>"  # - section row cell

    @classmethod
    def get_special_tokens(cls):
        """Function to get all special document tokens."""
        special_tokens = [token.value for token in cls]
        return special_tokens

    @staticmethod
    def is_known_token(label):
        """Function to check if label is in tokens."""
        return label in TableToken.get_special_tokens()


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

        # FIXME: this is synonym of section header
        for i in range(6):
            special_tokens += [f"<subtitle-level-{i}>", f"</subtitle-level-{i}>"]

        # Adding dynamically generated page-tokens
        for i in range(0, max_pages + 1):
            special_tokens.append(f"<page_{i}>")
            special_tokens.append(f"</page_{i}>")

        # Adding dynamically generated location-tokens
        for i in range(0, max(page_dimension[0] + 1, page_dimension[1] + 1)):
            special_tokens.append(f"<loc_{i}>")

        return special_tokens

    @staticmethod
    def is_known_token(label):
        """Function to check if label is in tokens."""
        return label in DocumentToken.get_special_tokens()

    @staticmethod
    def get_row_token(row: int, beg=bool) -> str:
        """Function to get page tokens."""
        if beg:
            return f"<row_{row}>"
        else:
            return f"</row_{row}>"

    @staticmethod
    def get_col_token(col: int, beg=bool) -> str:
        """Function to get page tokens."""
        if beg:
            return f"<col_{col}>"
        else:
            return f"</col_{col}>"

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

    @staticmethod
    def get_location(
        # bbox: Tuple[float, float, float, float],
        bbox: Annotated[list[float], Field(min_length=4, max_length=4)],
        page_w: float,
        page_h: float,
        xsize: int = 100,
        ysize: int = 100,
        page_i: int = -1,
    ):
        """Get the location string give bbox and page-dim."""
        assert bbox[0] <= bbox[2], f"bbox[0]<=bbox[2] => {bbox[0]}<={bbox[2]}"
        assert bbox[1] <= bbox[3], f"bbox[1]<=bbox[3] => {bbox[1]}<={bbox[3]}"

        x0 = bbox[0] / page_w
        y0 = bbox[1] / page_h
        x1 = bbox[2] / page_w
        y1 = bbox[3] / page_h

        page_tok = ""
        if page_i != -1:
            page_tok = DocumentToken.get_page_token(page=page_i)

        x0_tok = DocumentToken.get_location_token(val=min(x0, x1), rnorm=xsize)
        y0_tok = DocumentToken.get_location_token(val=min(y0, y1), rnorm=ysize)
        x1_tok = DocumentToken.get_location_token(val=max(x0, x1), rnorm=xsize)
        y1_tok = DocumentToken.get_location_token(val=max(y0, y1), rnorm=ysize)

        loc_str = f"{DocumentToken.BEG_LOCATION.value}"
        loc_str += f"{page_tok}{x0_tok}{y0_tok}{x1_tok}{y1_tok}"
        loc_str += f"{DocumentToken.END_LOCATION.value}"

        return loc_str
