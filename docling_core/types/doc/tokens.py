#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Tokens used in the docling document model."""

from enum import Enum
from typing import Tuple

from docling_core.types.doc.labels import DocItemLabel


class TableToken(str, Enum):
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


_LOC_PREFIX = "loc_"
_SECTION_HEADER_PREFIX = "section_header_level_"


class _PictureClassificationToken(str, Enum):
    """PictureClassificationToken."""

    OTHER = "<other>"

    # If more than one picture is grouped together, it
    # is generally not possible to assign a label
    PICTURE_GROUP = "<picture_group>"

    # General
    PIE_CHART = "<pie_chart>"
    BAR_CHART = "<bar_chart>"
    STACKED_BAR_CHART = "<stacked_bar_chart>"
    LINE_CHART = "<line_chart>"
    FLOW_CHART = "<flow_chart>"
    SCATTER_CHART = "<scatter_chart>"
    HEATMAP = "<heatmap>"
    REMOTE_SENSING = "<remote_sensing>"

    NATURAL_IMAGE = "<natural_image>"

    # Chemistry
    MOLECULAR_STRUCTURE = "<chemistry_molecular_structure>"
    MARKUSH_STRUCTURE = "<chemistry_markush_structure>"

    # Company
    ICON = "<icon>"
    LOGO = "<logo>"
    SIGNATURE = "<signature>"
    STAMP = "<stamp>"
    QR_CODE = "<qr_code>"
    BAR_CODE = "<bar_code>"
    SCREENSHOT = "<screenshot>"

    # Geology/Geography
    GEOGRAPHIC_MAP = "<map>"
    STRATIGRAPHIC_CHART = "<stratigraphic_chart>"

    # Engineering
    CAD_DRAWING = "<cad_drawing>"
    ELECTRICAL_DIAGRAM = "<electrical_diagram>"


class _CodeLanguageToken(str, Enum):
    """CodeLanguageToken."""

    ADA = "<_Ada_>"
    AWK = "<_Awk_>"
    BASH = "<_Bash_>"
    BC = "<_bc_>"
    C = "<_C_>"
    C_SHARP = "<_C#_>"
    C_PLUS_PLUS = "<_C++_>"
    CMAKE = "<_CMake_>"
    COBOL = "<_COBOL_>"
    CSS = "<_CSS_>"
    CEYLON = "<_Ceylon_>"
    CLOJURE = "<_Clojure_>"
    CRYSTAL = "<_Crystal_>"
    CUDA = "<_Cuda_>"
    CYTHON = "<_Cython_>"
    D = "<_D_>"
    DART = "<_Dart_>"
    DC = "<_dc_>"
    DOCKERFILE = "<_Dockerfile_>"
    ELIXIR = "<_Elixir_>"
    ERLANG = "<_Erlang_>"
    FORTRAN = "<_FORTRAN_>"
    FORTH = "<_Forth_>"
    GO = "<_Go_>"
    HTML = "<_HTML_>"
    HASKELL = "<_Haskell_>"
    HAXE = "<_Haxe_>"
    JAVA = "<_Java_>"
    JAVASCRIPT = "<_JavaScript_>"
    JULIA = "<_Julia_>"
    KOTLIN = "<_Kotlin_>"
    LISP = "<_Lisp_>"
    LUA = "<_Lua_>"
    MATLAB = "<_Matlab_>"
    MOONSCRIPT = "<_MoonScript_>"
    NIM = "<_Nim_>"
    OCAML = "<_OCaml_>"
    OBJECTIVEC = "<_ObjectiveC_>"
    OCTAVE = "<_Octave_>"
    PHP = "<_PHP_>"
    PASCAL = "<_Pascal_>"
    PERL = "<_Perl_>"
    PROLOG = "<_Prolog_>"
    PYTHON = "<_Python_>"
    RACKET = "<_Racket_>"
    RUBY = "<_Ruby_>"
    RUST = "<_Rust_>"
    SML = "<_SML_>"
    SQL = "<_SQL_>"
    SCALA = "<_Scala_>"
    SCHEME = "<_Scheme_>"
    SWIFT = "<_Swift_>"
    TYPESCRIPT = "<_TypeScript_>"
    UNKNOWN = "<_unknown_>"
    VISUALBASIC = "<_VisualBasic_>"
    XML = "<_XML_>"
    YAML = "<_YAML_>"


class DocumentToken(str, Enum):
    """Class to represent an LLM friendly representation of a Document."""

    DOCUMENT = "doctag"
    OTSL = "otsl"
    CHART = "chart"
    ORDERED_LIST = "ordered_list"
    UNORDERED_LIST = "unordered_list"
    PAGE_BREAK = "page_break"
    SMILES = "smiles"
    INLINE = "inline"

    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PICTURE = "picture"
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    DOCUMENT_INDEX = "document_index"
    CODE = "code"
    CHECKBOX_SELECTED = "checkbox_selected"
    CHECKBOX_UNSELECTED = "checkbox_unselected"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"

    PARAGRAPH = "paragraph"
    REFERENCE = "reference"

    @classmethod
    def get_special_tokens(
        cls,
        page_dimension: Tuple[int, int] = (500, 500),
    ):
        """Function to get all special document tokens."""
        special_tokens: list[str] = []
        for token in cls:
            special_tokens.append(f"<{token.value}>")
            special_tokens.append(f"</{token.value}>")

        for i in range(6):
            special_tokens += [
                f"<{_SECTION_HEADER_PREFIX}{i}>",
                f"</{_SECTION_HEADER_PREFIX}{i}>",
            ]

        special_tokens.extend([t.value for t in _PictureClassificationToken])
        special_tokens.extend([t.value for t in _CodeLanguageToken])

        special_tokens.extend(TableToken.get_special_tokens())

        # Adding dynamically generated location-tokens
        for i in range(0, max(page_dimension[0], page_dimension[1])):
            special_tokens.append(f"<{_LOC_PREFIX}{i}>")

        return special_tokens

    @classmethod
    def create_token_name_from_doc_item_label(cls, label: str, level: int = 1) -> str:
        """Get token corresponding to passed doc item label."""
        doc_token_by_item_label = {
            DocItemLabel.CAPTION: DocumentToken.CAPTION,
            DocItemLabel.FOOTNOTE: DocumentToken.FOOTNOTE,
            DocItemLabel.FORMULA: DocumentToken.FORMULA,
            DocItemLabel.LIST_ITEM: DocumentToken.LIST_ITEM,
            DocItemLabel.PAGE_FOOTER: DocumentToken.PAGE_FOOTER,
            DocItemLabel.PAGE_HEADER: DocumentToken.PAGE_HEADER,
            DocItemLabel.PICTURE: DocumentToken.PICTURE,
            DocItemLabel.TABLE: DocumentToken.TABLE,
            DocItemLabel.TEXT: DocumentToken.TEXT,
            DocItemLabel.TITLE: DocumentToken.TITLE,
            DocItemLabel.DOCUMENT_INDEX: DocumentToken.DOCUMENT_INDEX,
            DocItemLabel.CODE: DocumentToken.CODE,
            DocItemLabel.CHECKBOX_SELECTED: DocumentToken.CHECKBOX_SELECTED,
            DocItemLabel.CHECKBOX_UNSELECTED: DocumentToken.CHECKBOX_UNSELECTED,
            DocItemLabel.FORM: DocumentToken.FORM,
            DocItemLabel.KEY_VALUE_REGION: DocumentToken.KEY_VALUE_REGION,
            DocItemLabel.PARAGRAPH: DocumentToken.PARAGRAPH,
            DocItemLabel.REFERENCE: DocumentToken.REFERENCE,
            DocItemLabel.CHART: DocumentToken.CHART,
        }

        res: str
        if label == DocItemLabel.SECTION_HEADER:
            res = f"{_SECTION_HEADER_PREFIX}{level}"
        else:
            try:
                res = doc_token_by_item_label[DocItemLabel(label)].value
            except KeyError as e:
                raise RuntimeError(f"Unexpected DocItemLabel: {label}") from e
        return res

    @staticmethod
    def is_known_token(label):
        """Function to check if label is in tokens."""
        return label in DocumentToken.get_special_tokens()

    @staticmethod
    def get_picture_classification_token(classification: str) -> str:
        """Function to get the token for a given picture classification value."""
        return _PictureClassificationToken(f"<{classification}>").value

    @staticmethod
    def get_code_language_token(code_language: str) -> str:
        """Function to get the token for a given code language."""
        return _CodeLanguageToken(f"<_{code_language}_>").value

    @staticmethod
    def get_location_token(val: float, rnorm: int = 500):  # TODO review
        """Function to get location tokens."""
        val_ = round(rnorm * val)
        val_ = max(val_, 0)
        val_ = min(val_, rnorm - 1)
        return f"<{_LOC_PREFIX}{val_}>"

    @staticmethod
    def get_location(
        bbox: tuple[float, float, float, float],
        page_w: float,
        page_h: float,
        xsize: int = 500,  # TODO review
        ysize: int = 500,  # TODO review
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
