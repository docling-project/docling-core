#!/usr/bin/env python3
"""Generate ``docling_core/utils/symbol_fonts.json`` from the Unicode Consortium sources.

The table maps the byte codes of the symbol-encoded fonts (Symbol, ZapfDingbats,
Wingdings, Wingdings 2, Wingdings 3, Webdings) to Unicode. It is data only, so
that the Python, C++ (docling-parse) and Rust (docling.rs) readers can share one
copy instead of keeping their own.

Sources:

- Symbol and ZapfDingbats: Adobe's encodings as published by Unicode in
  ``MAPPINGS/VENDORS/ADOBE/symbol.txt`` and ``zdingbat.txt``. Where a code has
  several Unicode values, ``PREFERRED`` picks one. Corporate Use Subarea values
  are resolved through their Unicode name or ``CORPORATE_USE``; the table never
  maps to a private-use code point.
- Wingdings, Wingdings 2, Wingdings 3 and Webdings: the normative "Source
  references for wingdings and webding symbols" at the end of L2/12-368 (WG2
  N4384, "Status of encoding of Wingdings and Webdings Symbols"), a
  ``UCS;ID[;ID]`` list that gives the UCS code of 866 of the 867 glyphs. The PDF
  is read with ``pdftotext`` (poppler). The glyph count per font is checked, and
  every code point must be assigned in the running Python's Unicode database.

Usage::

    uv run python scripts/generate_symbol_fonts.py            # rewrite the table
    uv run python scripts/generate_symbol_fonts.py --check    # exit 1 if it is stale
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

ADOBE_URL = "https://www.unicode.org/Public/MAPPINGS/VENDORS/ADOBE/{name}.txt"
L2_12_368_URL = "https://www.unicode.org/L2/L2012/12368-n4384.pdf"
OUTPUT = Path(__file__).resolve().parent.parent / "docling_core" / "utils" / "symbol_fonts.json"

# Codes the Adobe files map to several Unicode values: keep the character a
# reader of the text expects, not the compatibility sign.
PREFERRED = {
    "Symbol": {
        0x20: 0x0020,  # SPACE, not NO-BREAK SPACE
        0x44: 0x0394,  # GREEK CAPITAL LETTER DELTA, not INCREMENT
        0x57: 0x03A9,  # GREEK CAPITAL LETTER OMEGA, not OHM SIGN
        0x6D: 0x03BC,  # GREEK SMALL LETTER MU, not MICRO SIGN
        0xA4: 0x2044,  # FRACTION SLASH, not DIVISION SLASH
    },
    "ZapfDingbats": {0x20: 0x0020},
}

# Symbol glyphs that Adobe placed in the Corporate Use Subarea under names that
# are not Unicode names. The delimiter pieces use the Miscellaneous Technical
# characters added in Unicode 3.2 (the same rewrite docling-parse applies to the
# Adobe Glyph List); the serif and sans serif variants of the registered,
# copyright and trade mark signs use the plain signs. RADICAL EXTENDER (0x60)
# has no Unicode equivalent and stays out of the table.
CORPORATE_USE = {
    "Symbol": {
        0xBD: 0x23D0,  # VERTICAL LINE EXTENSION
        0xBE: 0x23AF,  # HORIZONTAL LINE EXTENSION
        0xD2: 0x00AE,  # REGISTERED SIGN
        0xD3: 0x00A9,  # COPYRIGHT SIGN
        0xD4: 0x2122,  # TRADE MARK SIGN
        0xE2: 0x00AE,
        0xE3: 0x00A9,
        0xE4: 0x2122,
        0xE6: 0x239B,  # LEFT PARENTHESIS UPPER HOOK
        0xE7: 0x239C,  # LEFT PARENTHESIS EXTENSION
        0xE8: 0x239D,  # LEFT PARENTHESIS LOWER HOOK
        0xE9: 0x23A1,  # LEFT SQUARE BRACKET UPPER CORNER
        0xEA: 0x23A2,  # LEFT SQUARE BRACKET EXTENSION
        0xEB: 0x23A3,  # LEFT SQUARE BRACKET LOWER CORNER
        0xEC: 0x23A7,  # LEFT CURLY BRACKET UPPER HOOK
        0xED: 0x23A8,  # LEFT CURLY BRACKET MIDDLE PIECE
        0xEE: 0x23A9,  # LEFT CURLY BRACKET LOWER HOOK
        0xEF: 0x23AA,  # CURLY BRACKET EXTENSION
        0xF4: 0x23AE,  # INTEGRAL EXTENSION
        0xF6: 0x239E,  # RIGHT PARENTHESIS UPPER HOOK
        0xF7: 0x239F,  # RIGHT PARENTHESIS EXTENSION
        0xF8: 0x23A0,  # RIGHT PARENTHESIS LOWER HOOK
        0xF9: 0x23A4,  # RIGHT SQUARE BRACKET UPPER CORNER
        0xFA: 0x23A5,  # RIGHT SQUARE BRACKET EXTENSION
        0xFB: 0x23A6,  # RIGHT SQUARE BRACKET LOWER CORNER
        0xFC: 0x23AB,  # RIGHT CURLY BRACKET UPPER HOOK
        0xFD: 0x23AC,  # RIGHT CURLY BRACKET MIDDLE PIECE
        0xFE: 0x23AD,  # RIGHT CURLY BRACKET LOWER HOOK
    },
}

# L2/12-368 identifies a glyph as <set digit><decimal byte>, e.g. w-1108 is
# Wingdings 0x6C. Its tables start at byte 33; byte 32 is the space glyph in
# every one of these fonts.
L2_SETS = {"0": "Webdings", "1": "Wingdings", "2": "Wingdings 2", "3": "Wingdings 3"}

# Glyphs per font in the annex: w-1255 (Wingdings) is the one glyph not encoded.
L2_COUNTS = {"Webdings": 222, "Wingdings": 221, "Wingdings 2": 216, "Wingdings 3": 207}

NOTICE = (
    "Symbol and ZapfDingbats are derived from the Adobe mapping files published by Unicode, Inc. under "
    "this notice: Copyright (c) 1991-2011 Unicode, Inc. All Rights reserved. This file is provided as-is "
    "by Unicode, Inc. (The Unicode Consortium). No claims are made as to fitness for any particular purpose. "
    "No warranties of any kind are expressed or implied. The recipient agrees to determine applicability of "
    "information provided. If this file has been provided on magnetic media by Unicode, Inc., the sole remedy "
    "for any claim will be exchange of defective media within 90 days of receipt. Unicode, Inc. hereby grants "
    "the right to freely use the information supplied in this file in the creation of products supporting "
    "the Unicode Standard, and to make copies of this file in any form for internal or external distribution "
    "as long as this notice remains attached."
)


def fetch(url: str, cache_dir: Path) -> Path:
    """Download ``url`` into ``cache_dir`` once and return the local path."""
    path = cache_dir / url.rsplit("/", 1)[1]
    if not path.exists():
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


def is_private_use(cp: int) -> bool:
    """Whether ``cp`` is a private-use code point."""
    return unicodedata.category(chr(cp)) == "Co"


def read_adobe(path: Path, font: str) -> dict[int, int]:
    """Read an Adobe-encoding-to-Unicode file of the Unicode MAPPINGS tree."""
    candidates: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        name = fields[2].removeprefix("# ").strip()
        candidates[int(fields[1], 16)].append((int(fields[0], 16), name))

    table: dict[int, int] = {}
    for code, values in sorted(candidates.items()):
        if code in PREFERRED.get(font, {}):
            chosen = PREFERRED[font][code]
            if chosen not in [cp for cp, _ in values]:
                raise ValueError(f"{font} 0x{code:02X}: preferred U+{chosen:04X} is not a listed value")
        elif len(values) > 1:
            raise ValueError(f"{font} 0x{code:02X} has several values {values}: add it to PREFERRED")
        else:
            chosen, name = values[0]
            if is_private_use(chosen):
                resolved = CORPORATE_USE.get(font, {}).get(code) or resolve_name(name)
                if resolved is None:
                    continue
                chosen = resolved
        table[code] = chosen
    return table


def resolve_name(name: str) -> int | None:
    """The code point of the Unicode character called ``name``, if there is one."""
    try:
        return ord(unicodedata.lookup(name))
    except KeyError:
        return None


def read_l2_12_368(pdf: Path) -> dict[str, dict[int, int]]:
    """Read the normative source references (``UCS;ID[;ID]``) at the end of L2/12-368."""
    text = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    annex = text[text.index("Source references for wingdings and webding symbols") :]

    tables: dict[str, dict[int, int]] = defaultdict(dict)
    for ucs, *glyph_ids in re.findall(r"\b([0-9A-F]{5});(\d{4})(?:;(\d{4}))?", annex):
        for glyph_id in filter(None, glyph_ids):
            font, byte = L2_SETS[glyph_id[0]], int(glyph_id[1:])
            if byte in tables[font]:
                raise ValueError(f"w-{glyph_id} is listed twice")
            tables[font][byte] = int(ucs, 16)

    counts = {font: len(table) for font, table in tables.items()}
    if counts != L2_COUNTS:
        raise ValueError(f"read {counts} glyphs, expected {L2_COUNTS}")
    for table in tables.values():
        table[0x20] = 0x0020
    return dict(tables)


def build(cache_dir: Path) -> dict:
    """Build the JSON document."""
    fonts = {
        "Symbol": read_adobe(fetch(ADOBE_URL.format(name="symbol"), cache_dir), "Symbol"),
        "ZapfDingbats": read_adobe(fetch(ADOBE_URL.format(name="zdingbat"), cache_dir), "ZapfDingbats"),
    }
    l2 = read_l2_12_368(fetch(L2_12_368_URL, cache_dir))
    for font in ("Wingdings", "Wingdings 2", "Wingdings 3", "Webdings"):
        fonts[font] = l2[font]

    for font, table in fonts.items():
        for code, cp in table.items():
            if is_private_use(cp) or unicodedata.name(chr(cp), None) is None:
                raise ValueError(f"{font} 0x{code:02X} maps to U+{cp:04X}, which is private-use or unassigned")

    return {
        "description": (
            "Byte code (0x20-0xFF) of a symbol-encoded font to Unicode code point, both in hexadecimal. "
            "Generated by scripts/generate_symbol_fonts.py; do not edit."
        ),
        "sources": {
            "Symbol": ADOBE_URL.format(name="symbol"),
            "ZapfDingbats": ADOBE_URL.format(name="zdingbat"),
            "Wingdings": L2_12_368_URL,
            "Wingdings 2": L2_12_368_URL,
            "Wingdings 3": L2_12_368_URL,
            "Webdings": L2_12_368_URL,
        },
        "notice": NOTICE,
        "fonts": {
            font: {f"{code:02X}": f"{cp:04X}" for code, cp in sorted(table.items())} for font, table in fonts.items()
        },
    }


def main() -> int:
    """Rewrite the table, or check that it is up to date."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed table differs")
    parser.add_argument("--cache-dir", type=Path, help="keep the downloaded sources here")
    args = parser.parse_args()

    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        document = build(args.cache_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            document = build(Path(tmp))

    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT} is out of date; run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}: " + ", ".join(f"{f} {len(t)}" for f, t in document["fonts"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
