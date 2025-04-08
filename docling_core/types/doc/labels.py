"""Models for the labels types."""

from enum import Enum
from typing import Tuple


class DocItemLabel(str, Enum):
    """DocItemLabel."""

    CAPTION = "caption"
    CHART = "chart"
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
    PARAGRAPH = "paragraph"
    REFERENCE = "reference"

    def __str__(self):
        """Get string value."""
        return str(self.value)

    @staticmethod
    def get_color(label: "DocItemLabel") -> Tuple[int, int, int]:
        """Return the RGB color associated with a given label."""
        color_map = {
            DocItemLabel.CAPTION: (255, 204, 153),
            DocItemLabel.FOOTNOTE: (200, 200, 255),
            DocItemLabel.FORMULA: (192, 192, 192),
            DocItemLabel.LIST_ITEM: (153, 153, 255),
            DocItemLabel.PAGE_FOOTER: (204, 255, 204),
            DocItemLabel.PAGE_HEADER: (204, 255, 204),
            DocItemLabel.PICTURE: (255, 204, 164),
            DocItemLabel.SECTION_HEADER: (255, 153, 153),
            DocItemLabel.TABLE: (255, 204, 204),
            DocItemLabel.TEXT: (255, 255, 153),
            DocItemLabel.TITLE: (255, 153, 153),
            DocItemLabel.DOCUMENT_INDEX: (220, 220, 220),
            DocItemLabel.CODE: (125, 125, 125),
            DocItemLabel.CHECKBOX_SELECTED: (255, 182, 193),
            DocItemLabel.CHECKBOX_UNSELECTED: (255, 182, 193),
            DocItemLabel.FORM: (200, 255, 255),
            DocItemLabel.KEY_VALUE_REGION: (183, 65, 14),
            DocItemLabel.PARAGRAPH: (255, 255, 153),
            DocItemLabel.REFERENCE: (176, 224, 230),
        }
        return color_map[label]


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
    FORM_AREA = "form_area"
    KEY_VALUE_AREA = "key_value_area"
    COMMENT_SECTION = "comment_section"
    INLINE = "inline"

    def __str__(self):
        """Get string value."""
        return str(self.value)


class PictureClassificationLabel(str, Enum):
    """PictureClassificationLabel."""

    OTHER = "other"

    # If more than one picture is grouped together, it
    # is generally not possible to assign a label
    PICTURE_GROUP = "picture_group"

    # General
    PIE_CHART = "pie_chart"
    BAR_CHART = "bar_chart"
    STACKED_BAR_CHART = "stacked_bar_chart"
    LINE_CHART = "line_chart"
    FLOW_CHART = "flow_chart"
    SCATTER_CHART = "scatter_chart"
    HEATMAP = "heatmap"
    REMOTE_SENSING = "remote_sensing"

    NATURAL_IMAGE = "natural_image"

    # Chemistry
    MOLECULAR_STRUCTURE = "chemistry_molecular_structure"
    MARKUSH_STRUCTURE = "chemistry_markush_structure"

    # Company
    ICON = "icon"
    LOGO = "logo"
    SIGNATURE = "signature"
    STAMP = "stamp"
    QR_CODE = "qr_code"
    BAR_CODE = "bar_code"
    SCREENSHOT = "screenshot"

    # Geology/Geography
    GEOGRAPHIC_MAP = "map"
    STRATIGRAPHIC_CHART = "stratigraphic_chart"

    # Engineering
    CAD_DRAWING = "cad_drawing"
    ELECTRICAL_DIAGRAM = "electrical_diagram"

    def __str__(self):
        """Get string value."""
        return str(self.value)


class TableCellLabel(str, Enum):
    """TableCellLabel."""

    COLUMN_HEADER = "col_header"
    ROW_HEADER = "row_header"
    ROW_SECTION = "row_section"
    BODY = "body"

    def __str__(self):
        """Get string value."""
        return str(self.value)


class GraphCellLabel(str, Enum):
    """GraphCellLabel."""

    UNSPECIFIED = "unspecified"

    KEY = "key"
    VALUE = "value"

    CHECKBOX = "checkbox"


class GraphLinkLabel(str, Enum):
    """GraphLinkLabel."""

    UNSPECIFIED = "unspecified"

    TO_VALUE = "to_value"
    TO_KEY = "to_key"

    TO_PARENT = "to_parent"
    TO_CHILD = "to_child"


class CodeLanguageLabel(str, Enum):
    """CodeLanguageLabel."""

    ADA = "Ada"
    AWK = "Awk"
    BASH = "Bash"
    BC = "bc"
    C = "C"
    C_SHARP = "C#"
    C_PLUS_PLUS = "C++"
    CMAKE = "CMake"
    COBOL = "COBOL"
    CSS = "CSS"
    CEYLON = "Ceylon"
    CLOJURE = "Clojure"
    CRYSTAL = "Crystal"
    CUDA = "Cuda"
    CYTHON = "Cython"
    D = "D"
    DART = "Dart"
    DC = "dc"
    DOCKERFILE = "Dockerfile"
    ELIXIR = "Elixir"
    ERLANG = "Erlang"
    FORTRAN = "FORTRAN"
    FORTH = "Forth"
    GO = "Go"
    HTML = "HTML"
    HASKELL = "Haskell"
    HAXE = "Haxe"
    JAVA = "Java"
    JAVASCRIPT = "JavaScript"
    JULIA = "Julia"
    KOTLIN = "Kotlin"
    LISP = "Lisp"
    LUA = "Lua"
    MATLAB = "Matlab"
    MOONSCRIPT = "MoonScript"
    NIM = "Nim"
    OCAML = "OCaml"
    OBJECTIVEC = "ObjectiveC"
    OCTAVE = "Octave"
    PHP = "PHP"
    PASCAL = "Pascal"
    PERL = "Perl"
    PROLOG = "Prolog"
    PYTHON = "Python"
    RACKET = "Racket"
    RUBY = "Ruby"
    RUST = "Rust"
    SML = "SML"
    SQL = "SQL"
    SCALA = "Scala"
    SCHEME = "Scheme"
    SWIFT = "Swift"
    TYPESCRIPT = "TypeScript"
    UNKNOWN = "unknown"
    VISUALBASIC = "VisualBasic"
    XML = "XML"
    YAML = "YAML"

    def __str__(self):
        """Get string value."""
        return str(self.value)
