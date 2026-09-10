"""Server-side downgrade projectors for DoclingDocument schema versions.

This module provides the machinery that allows a server running a *newer*
docling-core (v2.x) to return a DoclingDocument that is still parseable by a
client running an *older* docling-core (v2.x).

Scope
-----
Projectors are provided for every schema minor-version step within the
docling-core **v2.x** release line, starting from the oldest v2.x schema
(1.5.0, introduced in v2.38.2).  Schema versions 1.0-1.4 were produced by
the v1.x release line and are outside the supported downgrade range.

Protocol
--------
Every time ``CURRENT_VERSION`` in
``docling_core.types.doc.common.constants`` is bumped (minor or major), a
matching ``@register_projector`` function **must** be added in this file.
The CI enforcement test ``test/test_compat.py::TestProjectorCoverage``
and the script ``scripts/check_compat_projectors.py`` verify this invariant
automatically.

Usage (server side)
-------------------
::

    from docling_core.compat import project_to
    from docling_core.types.doc.document import DoclingDocument

    doc: DoclingDocument = converter.convert(pdf)
    client_version = request.headers.get("Accept-Schema-Version")
    if client_version:
        doc = project_to(doc, target_version=client_version)
    return doc.model_dump_json()

Adding a new projector (contributor guide)
------------------------------------------
When you bump ``CURRENT_VERSION`` from ``1.N.0`` to ``1.(N+1).0``:

1. Implement the changes to the Pydantic models.
2. Bump ``CURRENT_VERSION`` in
   ``docling_core/types/doc/common/constants.py``.
3. Append a new entry to ``SCHEMA_VERSION_HISTORY`` in the same file.
4. Add a function decorated with
   ``@register_projector(from_minor=N+1, to_minor=N)`` **in this file**,
   below the existing projectors.
5. The function receives the raw ``dict`` produced by
   ``DoclingDocument.model_dump(mode="python")`` and must return a
   modified ``dict`` that is valid against schema ``1.N.0``.
   Always set ``data["version"] = f"1.{N}.0"`` before returning.
6. Add a test in ``test/test_compat.py`` that exercises the new projector.

See the existing projectors below for examples.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps (from_minor, to_minor) → projector callable.
# Keys always satisfy from_minor == to_minor + 1  (single-step downgrade).
_ProjectorFn = Callable[[dict], dict]
_projectors: dict[tuple[int, int], _ProjectorFn] = {}

_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def register_projector(*, from_minor: int, to_minor: int) -> Callable[[_ProjectorFn], _ProjectorFn]:
    """Decorator: register a single-step downgrade projector.

    Args:
        from_minor: The schema minor version that the projector *reads*.
        to_minor: The schema minor version that the projector *produces*.
            Must equal ``from_minor - 1``.

    Raises:
        ValueError: If ``to_minor != from_minor - 1``.
    """
    if to_minor != from_minor - 1:
        raise ValueError(
            f"Projectors must step exactly one minor version at a time; "
            f"got from_minor={from_minor}, to_minor={to_minor}."
        )

    def decorator(fn: _ProjectorFn) -> _ProjectorFn:
        _projectors[(from_minor, to_minor)] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def project_to(doc: DoclingDocument, target_version: str) -> DoclingDocument:
    """Return a copy of *doc* projected to be parseable by *target_version* SDK.

    The function walks the projector chain from the document's current schema
    version down to *target_version*, applying each registered projector in
    sequence.  If a projector is missing for a given step, a ``RuntimeError``
    is raised rather than silently producing an incompatible document.

    The minimum supported *target_version* is the oldest schema version
    produced by docling-core v2.x (currently ``1.5.0``).  Requesting a
    target below that boundary raises ``ValueError``.

    Args:
        doc: The source ``DoclingDocument`` (current schema version).
        target_version: Semver string of the schema version required by the
            consumer (e.g. ``"1.8.0"``).

    Returns:
        A new ``DoclingDocument`` validated at *target_version*.

    Raises:
        ValueError: If *target_version* is unparseable, has a different
            major version than *doc*, or is below the oldest supported
            schema version (1.5.0).
        RuntimeError: If a required projector step has not been registered.
    """
    from docling_core.types.doc.common.constants import FIRST_SUPPORTED_MINOR
    from docling_core.types.doc.document import DoclingDocument  # avoid circular at module level

    doc_match = _VERSION_RE.match(doc.version)
    tgt_match = _VERSION_RE.match(target_version)
    if doc_match is None:
        raise ValueError(f"Cannot parse document version {doc.version!r}")
    if tgt_match is None:
        raise ValueError(f"Cannot parse target version {target_version!r}")

    doc_major = int(doc_match["major"])
    tgt_major = int(tgt_match["major"])
    doc_minor = int(doc_match["minor"])
    tgt_minor = int(tgt_match["minor"])

    if doc_major != tgt_major:
        raise ValueError(
            f"Cannot project across major versions: document is {doc.version}, target is {target_version}."
        )

    if tgt_minor >= doc_minor:
        # Nothing to do — the target is equal or newer.
        return doc

    if tgt_minor < FIRST_SUPPORTED_MINOR:
        raise ValueError(
            f"Target version {target_version!r} is below the oldest supported schema version "
            f"1.{FIRST_SUPPORTED_MINOR}.0 (docling-core v2.x baseline). "
            f"Schema versions 1.0-1.{FIRST_SUPPORTED_MINOR - 1} were produced by docling-core v1.x "
            f"and are not supported by the downgrade-projector system."
        )

    data = doc.model_dump(mode="python")
    for current_minor in range(doc_minor, tgt_minor, -1):
        step_key = (current_minor, current_minor - 1)
        fn = _projectors.get(step_key)
        if fn is None:
            raise RuntimeError(
                f"No projector registered for schema 1.{current_minor} -> 1.{current_minor - 1}. "
                f"A projector must be added to docling_core/compat.py before this downgrade can be performed."
            )
        _logger.debug("Applying downgrade projector %d -> %d", current_minor, current_minor - 1)
        data = fn(data)

    # The current SDK's version validator normalises the version field to
    # CURRENT_VERSION on success.  We restore the target version afterwards so
    # callers and downstream serialisation correctly reflect the projected schema.
    result = DoclingDocument.model_validate(data)
    object.__setattr__(result, "version", data["version"])
    return result


def list_projectors() -> list[tuple[int, int]]:
    """Return a sorted list of registered projector keys ``(from_minor, to_minor)``."""
    return sorted(_projectors)


# ---------------------------------------------------------------------------
# Projectors — one per schema minor bump within docling-core v2.x
# ---------------------------------------------------------------------------
# These cover every schema version step from the current schema (1.10) down
# to the oldest v2.x schema (1.5).  Each projector represents the *minimal*
# structural downgrade that allows an old client's Pydantic model to parse
# the serialised dict without error.  Full semantic fidelity is not always
# possible; the goal is a document that validates rather than crashes.

# --- 1.10 → 1.9 -----------------------------------------------------------
# Schema 1.10 (docling-core v2.69.0, PR #519):
#   • Added FieldRegionItem, FieldHeadingItem, FieldItem, FieldValueItem
#     in the texts union (new discriminator label values).
#   • Added field_regions, field_items as new top-level lists.
#   • Added 7 new DocItemLabel values: FIELD_REGION, FIELD_HEADING,
#     FIELD_ITEM, FIELD_KEY, FIELD_VALUE, FIELD_HINT, MARKER.
# v2.70.1 (no bump): PictureClassificationLabel sync — removed 11 old values
#   but kept them as aliases; added 14 new values.
# v2.70.2 (no bump): CodeMetaField, FloatingMeta.code; HANDWRITTEN_TEXT label;
#   _validate_unique_refs; keywords/topics meta fields; Orientation enum.
#
# Downgrade strategy:
#   • Drop top-level "field_regions" and "field_items" keys (unknown to 1.9).
#   • In all item lists that could contain field-related items, convert them
#     to a generic TextItem with label "text" (safest common type).
#   • Strip unknown "code" key from any "floating_meta" dicts inside items.
#   • Strip "orientation" from table data.
#   • Strip "language", "entities", "keywords", "topics" from "meta" objects.
#   • New CodeLanguageLabel values (DOCLANG, LATEX, TIKZ, JSON, …) and
#     PictureClassificationLabel values: fall back to None / strip.
# ---------------------------------------------------------------------------

_FIELD_LABELS_1_10 = {
    "field_region",
    "field_heading",
    "field_item",
    "field_key",
    "field_value",
    "field_hint",
    "marker",
    "handwritten_text",
}

# DocItemLabel values that 1.9 clients do not know.
_NEW_DOC_ITEM_LABELS_1_10 = _FIELD_LABELS_1_10

# CodeLanguageLabel values first appearing at or after 1.9 era (no bump taken).
_NEW_CODE_LANG_1_9_ERA = {
    "JSON",  # v2.50.0
    "doclang",
    "latex",
    "tikz",
    # values added in v2.60.2 (alignment with Linguist):
    "abap",
    "actionscript",
    "ada",
    "agda",
    "alloy",
    "antlr",
    "apex",
    "applescript",
    "arc",
    "arduino",
    "asciidoc",
    "aspx",
    "asm",
    "aspect_j",
    "awk",
    "ballerina",
    "batchfile",
    "bicep",
    "bluespec",
    "boo",
    "brainfuck",
    "brightscript",
    "builders",
    "c",
    "c#",
    "c++",
    "cfg",
    "circom",
    "clojure",
    "cmake",
    "cobol",
    "coffeescript",
    "common_lisp",
    "css",
    "cuda",
    "d",
    "dart",
    "datalog",
    "dhall",
    "dockerfile",
    "eiffel",
    "elixir",
    "elm",
    "emacs_lisp",
    "erlang",
    "f#",
    "f*",
    "fidl",
    "forth",
    "fortran",
    "gedcom",
    "glsl",
    "go",
    "groovy",
    "haskell",
    "hlsl",
    "html",
    "idl",
    "isabelle",
    "java",
    "javascript",
    "jinja",
    "json5",
    "jsonld",
    "julia",
    "kotlin",
    "lean",
    "literate_agda",
    "literate_coffeescript",
    "literate_haskell",
    "lua",
    "makefile",
    "maple",
    "markdown",
    "matlab",
    "max_msp",
    "mediawiki",
    "mercury",
    "meson",
    "mlir",
    "modula-3",
    "moonscript",
    "nasm",
    "nix",
    "nu",
    "objective-c",
    "objective-c++",
    "ocaml",
    "openedge_abl",
    "pascal",
    "perl",
    "php",
    "plsql",
    "powershell",
    "prolog",
    "protobuf",
    "public_key",
    "puppet",
    "python",
    "q",
    "qml",
    "r",
    "racket",
    "raku",
    "reasonml",
    "restructuredtext",
    "robotframework",
    "ruby",
    "rust",
    "sas",
    "scala",
    "scheme",
    "scss",
    "shell",
    "smalltalk",
    "solidity",
    "sql",
    "stan",
    "standard_ml",
    "stata",
    "swift",
    "systemverilog",
    "tcl",
    "tex",
    "thrift",
    "toml",
    "tsql",
    "typescript",
    "vba",
    "verilog",
    "vhdl",
    "viml",
    "visual_basic",
    "webassembly",
    "xml",
    "xojo",
    "xquery",
    "yaml",
    "yara",
    "zig",
}

# PictureClassificationLabel values added in v2.70.1 (not present in 1.9 era).
_NEW_PICTURE_CLASS_LABELS_1_10 = {
    "bar_chart",
    "box_plot",
    "flow_chart",
    "line_chart",
    "pie_chart",
    "scatter_plot",
    "other_chart",
    "full_page_image",
    "page_thumbnail",
    "photograph",
    "chemistry_structure",
    "bar_code",
    "icon",
    "logo",
    "qr_code",
    "signature",
    "stamp",
    "engineering_drawing",
    "screenshot_from_computer",
    "screenshot_from_manual",
    "geographical_map",
    "topographical_map",
    "calendar",
    "crossword_puzzle",
    "music",
}

# Meta field keys added in 1.10 era (strip when downgrading to 1.9).
_NEW_META_KEYS_1_10 = {"language", "entities", "keywords", "topics", "code"}


def _strip_unknown_meta_fields(meta: dict, unknown_keys: set[str]) -> dict:
    """Return a copy of *meta* with *unknown_keys* removed (non-recursive)."""
    return {k: v for k, v in meta.items() if k not in unknown_keys}


def _downgrade_item_to_1_9(item: dict) -> dict:
    """Return a copy of *item* safe for schema 1.9 clients."""
    item = dict(item)

    # Strip new meta fields.
    if "meta" in item and isinstance(item["meta"], dict):
        item["meta"] = _strip_unknown_meta_fields(item["meta"], _NEW_META_KEYS_1_10)

    # Strip orientation from data dicts (TableData).
    if "data" in item and isinstance(item["data"], dict):
        item["data"] = {k: v for k, v in item["data"].items() if k != "orientation"}

    # Normalise unknown code_language to "unknown".
    if "code_language" in item and item["code_language"] in _NEW_CODE_LANG_1_9_ERA:
        item["code_language"] = "unknown"

    # Normalise unknown picture classification labels inside data.
    if "data" in item and isinstance(item["data"], dict):
        data = item["data"]
        if "classification" in data and isinstance(data["classification"], list):
            cleaned = []
            for cls_entry in data["classification"]:
                if isinstance(cls_entry, dict):
                    lbl = cls_entry.get("predicted_class")
                    if lbl in _NEW_PICTURE_CLASS_LABELS_1_10:
                        cls_entry = {**cls_entry, "predicted_class": "other"}
                cleaned.append(cls_entry)
            data = {**data, "classification": cleaned}
        item["data"] = data

    return item


@register_projector(from_minor=10, to_minor=9)
def _project_1_10_to_1_9(data: dict) -> dict:
    """Downgrade a schema 1.10 document dict to schema 1.9.

    Handles:
    - Removes top-level ``field_regions`` and ``field_items`` lists
      (unknown to 1.9 clients).
    - Converts field-related items still present in ``texts`` (e.g.
      FieldHeadingItem, FieldValueItem that have a text payload) to
      generic ``TextItem`` with label ``"text"``.
    - Strips ``orientation`` from ``TableData`` dicts.
    - Strips new meta sub-fields (``language``, ``entities``,
      ``keywords``, ``topics``, ``code``) from all ``BaseMeta`` dicts.
    - Falls back unknown ``CodeLanguageLabel`` values to ``"unknown"``.
    - Falls back unknown ``PictureClassificationLabel`` values to
      ``"other"``.
    """
    data = dict(data)

    # Drop new top-level lists.
    data.pop("field_regions", None)
    data.pop("field_items", None)

    # Convert field-label items in texts to generic TextItem.
    new_texts = []
    for item in data.get("texts", []):
        item = _downgrade_item_to_1_9(item)
        lbl = item.get("label", "")
        if lbl in _NEW_DOC_ITEM_LABELS_1_10:
            # Preserve text payload if present; otherwise omit.
            if "text" in item and "orig" in item:
                # Strip fields that belong only to specific subtypes
                # (e.g. FieldValueItem.kind, FieldHeadingItem.level) so
                # the generic TextItem union arm validates cleanly.
                item = {k: v for k, v in item.items() if k not in {"kind", "level"}}
                item = {**item, "label": "text"}
            else:
                # No text payload — skip the item entirely (safest option).
                continue
        new_texts.append(item)
    data["texts"] = new_texts

    # Downgrade remaining item lists (pictures, tables, key_value_items, form_items).
    for list_key in ("pictures", "tables", "key_value_items", "form_items"):
        data[list_key] = [_downgrade_item_to_1_9(item) for item in data.get(list_key, [])]

    # Downgrade groups.
    data["groups"] = [_downgrade_item_to_1_9(item) for item in data.get("groups", [])]

    data["version"] = "1.9.0"
    return data


# --- 1.9 → 1.8 ------------------------------------------------------------
# Schema 1.9 (docling-core v2.57.0, PR #465):
#   • Added DocItem.comments: list[FineRef] (new field with strict model).
#   • Added FineRef subclass with a "range" field.
#   • Added DocItem.source: list[SourceType] (PR #426, no bump but
#     present in 1.9 era).
#
# Downgrade strategy:
#   • Strip "comments" and "source" from every item dict.
# ---------------------------------------------------------------------------


def _downgrade_item_to_1_8(item: dict) -> dict:
    item = dict(item)
    item.pop("comments", None)
    item.pop("source", None)
    return item


@register_projector(from_minor=9, to_minor=8)
def _project_1_9_to_1_8(data: dict) -> dict:
    """Downgrade a schema 1.9 document dict to schema 1.8.

    Handles:
    - Strips ``comments`` and ``source`` fields from all ``DocItem`` dicts.
    """
    data = dict(data)

    for list_key in ("texts", "pictures", "tables", "key_value_items", "form_items", "groups"):
        data[list_key] = [_downgrade_item_to_1_8(item) for item in data.get(list_key, [])]

    data["version"] = "1.8.0"
    return data


# --- 1.8 → 1.7 ------------------------------------------------------------
# Schema 1.8 (docling-core v2.49.0, PR #408):
#   • Introduced BasePrediction, BaseMeta, FloatingMeta.
#   • Added "meta" optional field on NodeItem / FloatingItem.
#   NodeItem had extra="forbid", so the "meta" key would crash old clients.
#
# Downgrade strategy:
#   • Strip "meta" from all item dicts.
# ---------------------------------------------------------------------------


@register_projector(from_minor=8, to_minor=7)
def _project_1_8_to_1_7(data: dict) -> dict:
    """Downgrade a schema 1.8 document dict to schema 1.7.

    Handles:
    - Strips ``meta`` field from all node dicts.
    """
    data = dict(data)

    def _strip_meta(item: dict) -> dict:
        item = dict(item)
        item.pop("meta", None)
        return item

    for list_key in ("texts", "pictures", "tables", "key_value_items", "form_items", "groups"):
        data[list_key] = [_strip_meta(item) for item in data.get(list_key, [])]

    data["version"] = "1.7.0"
    return data


# --- 1.7 → 1.6 ------------------------------------------------------------
# Schema 1.7 (docling-core v2.47.0, PR #384):
#   • Added TableCell.fillable: bool = False.
#   Old strict clients crash on the unknown "fillable" field.
#
# Downgrade strategy:
#   • Strip "fillable" from every table cell dict inside every table.
# ---------------------------------------------------------------------------


@register_projector(from_minor=7, to_minor=6)
def _project_1_7_to_1_6(data: dict) -> dict:
    """Downgrade a schema 1.7 document dict to schema 1.6.

    Handles:
    - Strips ``fillable`` field from all ``TableCell`` dicts.
    """
    data = dict(data)

    new_tables = []
    for table in data.get("tables", []):
        table = dict(table)
        if "data" in table and isinstance(table["data"], dict):
            tdata = dict(table["data"])
            tdata["table_cells"] = [
                {k: v for k, v in cell.items() if k != "fillable"} for cell in tdata.get("table_cells", [])
            ]
            table["data"] = tdata
        new_tables.append(table)
    data["tables"] = new_tables

    data["version"] = "1.6.0"
    return data


# --- 1.6 → 1.5 ------------------------------------------------------------
# Schema 1.6 (docling-core v2.45.0, PR #368):
#   • Added RichTableCell with a "ref" field.
#   • TableData.table_cells changed to list[AnyTableCell] (union).
#   Old clients only know the plain TableCell model with extra="forbid",
#   so the "ref" field from RichTableCell crashes them.
#
# Downgrade strategy:
#   • For every table cell that has a "ref" key, remove "ref" so it
#     degrades cleanly to a plain TableCell.
# ---------------------------------------------------------------------------


@register_projector(from_minor=6, to_minor=5)
def _project_1_6_to_1_5(data: dict) -> dict:
    """Downgrade a schema 1.6 document dict to schema 1.5.

    Handles:
    - Strips ``ref`` field from ``RichTableCell`` dicts, turning them
      into plain ``TableCell`` dicts parseable by old clients.
    """
    data = dict(data)

    new_tables = []
    for table in data.get("tables", []):
        table = dict(table)
        if "data" in table and isinstance(table["data"], dict):
            tdata = dict(table["data"])
            tdata["table_cells"] = [
                {k: v for k, v in cell.items() if k != "ref"} for cell in tdata.get("table_cells", [])
            ]
            table["data"] = tdata
        new_tables.append(table)
    data["tables"] = new_tables

    data["version"] = "1.5.0"
    return data
