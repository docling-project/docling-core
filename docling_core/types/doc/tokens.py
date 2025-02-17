#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Tokens used in the docling document model."""

from enum import Enum
from typing import Tuple

from docling_core.types.doc.labels import PictureClassificationLabel


class TableToken(Enum):
    """Class to represent an LLM friendly representation of a Table."""

    CELL_LABEL_COLUMN_HEADER = "<column_header>"
    CELL_LABEL_ROW_HEADER = "<row_header>"
    CELL_LABEL_SECTION_HEADER = "<shed>"
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

    DOCUMENT = "doctag"
    OTSL = "otsl"
    ORDERED_LIST = "ordered_list"
    UNORDERED_LIST = "unordered_list"
    LOC = "loc_"
    PAGE_BREAK = "page_break"

    @classmethod
    def get_special_tokens(
        cls,
        page_dimension: Tuple[int, int] = (100, 100),
    ):
        """Function to get all special document tokens."""
        special_tokens = [token.value for token in cls]

        for i in range(6):
            special_tokens += [
                f"<section_header_level_{i}>",
                f"</section_header_level_{i}>",
            ]

        # Add dynamically picture classification tokens
        for _, member in PictureClassificationLabel.__members__.items():
            special_tokens.append(f"<{member}>")

        # Adding dynamically generated location-tokens
        for i in range(0, max(page_dimension[0] + 1, page_dimension[1] + 1)):
            special_tokens.append(f"<loc_{i}>")

        return special_tokens

    @staticmethod
    def is_known_token(label):
        """Function to check if label is in tokens."""
        return label in DocumentToken.get_special_tokens()

    @staticmethod
    def get_picture_classification_token(classification: str) -> str:
        """Function to get picture classification tokens."""
        return f"<{classification}>"

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
        bbox: tuple[float, float, float, float],
        page_w: float,
        page_h: float,
        xsize: int = 100,
        ysize: int = 100,
    ):
        """Get the location string give bbox and page-dim."""
        assert bbox[0] <= bbox[2], f"bbox[0]<=bbox[2] => {bbox[0]}<={bbox[2]}"
        assert bbox[1] <= bbox[3], f"bbox[1]<=bbox[3] => {bbox[1]}<={bbox[3]}"

        x0 = bbox[0] / page_w
        y0 = bbox[1] / page_h
        x1 = bbox[2] / page_w
        y1 = bbox[3] / page_h

        x0_tok = DocumentToken.get_location_token(val=min(x0, x1), rnorm=xsize)
        y0_tok = DocumentToken.get_location_token(val=min(y0, y1), rnorm=ysize)
        x1_tok = DocumentToken.get_location_token(val=max(x0, x1), rnorm=xsize)
        y1_tok = DocumentToken.get_location_token(val=max(y0, y1), rnorm=ysize)

        loc_str = f"{x0_tok}{y0_tok}{x1_tok}{y1_tok}"

        return loc_str
