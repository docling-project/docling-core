"""Tests for the symbol-font to Unicode table."""

import json
import unicodedata

import pytest

from docling_core.utils.symbol_fonts import (
    SYMBOL_FONTS_TABLE_PATH,
    SymbolFont,
    decode_symbol_font_text,
    symbol_font_char,
)


@pytest.mark.parametrize(
    ("font", "text", "expected"),
    [
        # The rows of docling#4610 / docling.rs#588 (w:t text and w:sym codes).
        (SymbolFont.SYMBOL, "\uf0b7", "\u2022"),
        (SymbolFont.SYMBOL, "\uf061\uf062\uf070", "\u03b1\u03b2\u03c0"),
        (SymbolFont.SYMBOL, "a b p", "\u03b1 \u03b2 \u03c0"),
        (SymbolFont.WINGDINGS, "\uf0a7", "\u25aa"),
        (SymbolFont.WINGDINGS, "\uf0e8", "\U0001f87a"),
        (SymbolFont.WINGDINGS, "\u008d", "\u278b"),
        (SymbolFont.WINGDINGS, "\uf0fc", "\u2713"),
        # Greek letters, not the compatibility signs the Adobe file also lists.
        (SymbolFont.SYMBOL, "\uf044\uf057\uf06d", "\u0394\u03a9\u03bc"),
        # Corporate-use glyphs resolve to standard characters.
        (SymbolFont.SYMBOL, "\uf0d2\uf0e3", "\u00ae\u00a9"),
        (SymbolFont.SYMBOL, "\uf0e6\uf0e8", "\u239b\u239d"),
        (SymbolFont.ZAPF_DINGBATS, "\x80\x81", "\u2768\u2769"),
        # No Unicode equivalent: the radical extender and the Windows logo stay.
        (SymbolFont.SYMBOL, "\uf060", "\uf060"),
        (SymbolFont.WINGDINGS, "\uf0ff", "\uf0ff"),
        # Characters outside the glyph range stay.
        (SymbolFont.WEBDINGS, "\u4e2d\U0001f600", "\u4e2d\U0001f600"),
    ],
)
def test_decode_symbol_font_text(font: SymbolFont, text: str, expected: str):
    assert decode_symbol_font_text(font, text) == expected


def test_space_glyph_is_a_plain_space():
    for font in SymbolFont:
        assert symbol_font_char(font, 0xF020) == " "


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Symbol", SymbolFont.SYMBOL),
        ("SymbolMT", SymbolFont.SYMBOL),
        ("Symbol,Bold", SymbolFont.SYMBOL),
        ("ABCDEF+Wingdings-Regular", SymbolFont.WINGDINGS),
        ("Wingdings 2", SymbolFont.WINGDINGS_2),
        ("Wingdings3", SymbolFont.WINGDINGS_3),
        (" webdings ", SymbolFont.WEBDINGS),
        ("ZapfDingbatsITC", SymbolFont.ZAPF_DINGBATS),
        # Unicode-encoded symbol fonts and unknown symbol fonts are not mapped.
        ("Symbola", None),
        ("Segoe UI Symbol", None),
        ("Monotype Sorts", None),
        ("Arial", None),
    ],
)
def test_symbol_font_from_name(name: str, expected: SymbolFont | None):
    assert SymbolFont.from_name(name) == expected


def test_table_maps_to_assigned_public_characters():
    """Readers rely on the table never producing private-use or unassigned code points."""
    fonts = json.loads(SYMBOL_FONTS_TABLE_PATH.read_text(encoding="utf-8"))["fonts"]
    assert set(fonts) == {font.value for font in SymbolFont}
    for font, table in fonts.items():
        for code, cp in table.items():
            assert 0x20 <= int(code, 16) <= 0xFF, (font, code)
            assert unicodedata.category(chr(int(cp, 16))) not in ("Co", "Cn"), (font, code, cp)
