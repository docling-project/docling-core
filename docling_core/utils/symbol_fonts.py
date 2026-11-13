"""Map text set in a symbol-encoded font to Unicode.

Symbol, ZapfDingbats, Wingdings, Wingdings 2, Wingdings 3 and Webdings place their
glyphs on the byte codes 0x20-0xFF, not on the Unicode characters they show. Office
files and PDFs carry such text either as the byte itself (``a`` set in Symbol shows
an alpha) or as the byte raised into the Private Use Area (U+F020-U+F0FF), which is
the cmap these fonts carry and how Word writes a ``w:sym`` element. Neither is
readable without the font.

The table is data only, in ``symbol_fonts.json`` next to this module, so that other
readers (the docling-parse PDF parser, docling.rs) can ship the same file. It is
built by ``scripts/generate_symbol_fonts.py`` from the Unicode Consortium sources
named in the file, and it never maps to a private-use code point.
"""

import json
import re
from enum import Enum
from functools import cache
from pathlib import Path

SYMBOL_FONTS_TABLE_PATH = Path(__file__).with_name("symbol_fonts.json")

_SUBSET_TAG = re.compile(r"^[A-Z]{6}\+")


class SymbolFont(str, Enum):
    """A symbol-encoded font whose byte codes the table maps to Unicode."""

    SYMBOL = "Symbol"
    ZAPF_DINGBATS = "ZapfDingbats"
    WINGDINGS = "Wingdings"
    WINGDINGS_2 = "Wingdings 2"
    WINGDINGS_3 = "Wingdings 3"
    WEBDINGS = "Webdings"

    @classmethod
    def from_name(cls, name: str) -> "SymbolFont | None":
        """Return the symbol font that a font name denotes, if any.

        Accepts the names Office files use (``Wingdings 2``) and the PostScript
        names PDFs use (``ABCDEF+Wingdings-Regular``, ``SymbolMT``, ``Symbol,Bold``):
        the subset tag and the style suffix are ignored, and so are case and spaces.
        Fonts that encode their glyphs as Unicode (``Symbola``, ``Segoe UI Symbol``)
        are not symbol-encoded and give ``None``.
        """
        family = _SUBSET_TAG.sub("", name.strip()).split(",", 1)[0].split("-", 1)[0]
        return _FONT_NAMES.get(family.replace(" ", "").lower())


_FONT_NAMES = {
    "symbol": SymbolFont.SYMBOL,
    "symbolmt": SymbolFont.SYMBOL,
    "zapfdingbats": SymbolFont.ZAPF_DINGBATS,
    "itczapfdingbats": SymbolFont.ZAPF_DINGBATS,
    "zapfdingbatsitc": SymbolFont.ZAPF_DINGBATS,
    "wingdings": SymbolFont.WINGDINGS,
    "wingdings2": SymbolFont.WINGDINGS_2,
    "wingdings3": SymbolFont.WINGDINGS_3,
    "webdings": SymbolFont.WEBDINGS,
}


@cache
def _tables() -> dict[SymbolFont, dict[int, str]]:
    document = json.loads(SYMBOL_FONTS_TABLE_PATH.read_text(encoding="utf-8"))
    return {
        SymbolFont(font): {int(code, 16): chr(int(cp, 16)) for code, cp in table.items()}
        for font, table in document["fonts"].items()
    }


def symbol_font_char(font: SymbolFont, code: int) -> str | None:
    """Return the Unicode character that glyph ``code`` of ``font`` shows.

    Args:
        font: The symbol font the glyph is set in.
        code: The byte code (0x20-0xFF) or its private-use form (U+F020-U+F0FF).

    Returns:
        The character, or ``None`` when the code is out of range or the font has
        no glyph with a Unicode equivalent there.
    """
    if 0xF020 <= code <= 0xF0FF:
        code -= 0xF000
    return _tables()[font].get(code)


def decode_symbol_font_text(font: SymbolFont, text: str) -> str:
    """Return ``text`` set in ``font`` with every glyph code replaced by its Unicode character.

    Characters without a mapping are kept as they are, so nothing is lost. Call it
    only for text that is set in a symbol font: elsewhere 0x80-0xFF is ordinary
    Latin-1 text.
    """
    return "".join(symbol_font_char(font, ord(char)) or char for char in text)
