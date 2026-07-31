#!/usr/bin/env python3
"""CI enforcement script — verify projector coverage for DoclingDocument schema.

This script is run as a pre-commit hook and in CI to ensure that every
schema minor-version bump is accompanied by a registered downgrade projector
AND a versioned JSON Schema snapshot.

What it does
------------
1. Checks projector count and step coverage (same as before).
2. Checks SCHEMA_VERSION_HISTORY is up-to-date.
3. Ensures ``docs/schemas/DoclingDocument_1_{N}.json`` exists for every
   registered projector's target minor version N.
4. **Auto-generates the current-version snapshot** when CURRENT_VERSION was
   just bumped: copies ``docs/DoclingDocument.json`` (which always reflects
   the *current* model) to ``docs/schemas/DoclingDocument_1_{CURRENT_MINOR}.json``
   if that file is missing.

The key invariant
-----------------
``docs/DoclingDocument.json`` is regenerated on every commit by the ``docs``
pre-commit hook (``uv run python -m docling_core.utils.generate_docs docs``).
When CURRENT_VERSION is bumped from 1.N to 1.(N+1):

  1. The contributor adds the model changes.
  2. The contributor bumps CURRENT_VERSION.
  3. This script runs (triggered by the change to constants.py).
  4. It detects that ``docs/schemas/DoclingDocument_1_N.json`` does not yet
     exist — i.e., the snapshot of the *old* schema was not captured yet.

**The snapshot must be taken BEFORE ``generate_docs`` overwrites
``docs/DoclingDocument.json``.**  The pre-commit hook order is:

  check-compat-projectors  ← runs first, copies the snapshot
  docs                     ← runs after, overwrites docs/DoclingDocument.json

This ordering is guaranteed by the hook order in ``.pre-commit-config.yaml``.

Exit codes:
    0  — all checks pass (snapshots may have been auto-generated)
    1  — one or more checks failed (prints diagnostics to stderr)

Usage::

    python scripts/check_compat_projectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SCHEMAS_DIR = _REPO_ROOT / "docs" / "schemas"
_LIVE_SCHEMA = _REPO_ROOT / "docs" / "DoclingDocument.json"


def _ensure_schemas_dir() -> None:
    _SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)


def _snapshot_path(minor: int) -> Path:
    return _SCHEMAS_DIR / f"DoclingDocument_1_{minor}.json"


# ---------------------------------------------------------------------------
# Semantic schema diff — detect only breaking changes
# ---------------------------------------------------------------------------


def _breaking_changes(old: dict, new: dict) -> list[str]:
    """Return a list of human-readable breaking-change descriptions.

    A breaking change is any diff in the JSON Schema that would cause an old
    client's ``DoclingDocument.model_validate()`` to raise when given a
    document produced by the new server.

    Precisely, Pydantic raises in these four situations:

    1. **New enum value** — a field value that is valid under the new schema
       is not a member of the old enum, so Pydantic's enum validator raises.
       Detected as: a value present in a ``$defs[X].enum`` array in ``new``
       that is absent from the same array in ``old``.

    2. **New optional field on a strict model** — every concrete item model
       (``TextItem``, ``TableItem``, …) has ``extra="forbid"`` (rendered as
       ``additionalProperties: false`` in the JSON Schema).  A new field sent
       by the server will be rejected by the old client's strict model.
       Detected as: a key present in ``$defs[X].properties`` in ``new`` but
       absent in ``old``, where ``old.$defs[X].additionalProperties == false``.

    3. **New required field** — the old client raises ``missing`` if a field
       in ``required`` is absent from the document.  A new required field added
       by the server may not be sent by old servers, but it will be present in
       new server output and the old client must accept it.  More precisely:
       if the new server adds it to ``required``, old clients that receive a
       doc *without* it (from any source) will break.  We flag it.
       Detected as: a value present in ``$defs[X].required`` in ``new`` but
       absent in ``old``.

    4. **New DocItem subtype in a union** — a new branch in an ``anyOf``
       discriminator union means the new server can produce documents with
       item types the old client's union does not recognise, causing a
       discriminator-lookup failure.
       Detected as: a new ``$ref`` entry in an ``anyOf`` array at any path
       in ``new`` that is absent in ``old``.

    Changes that are NOT breaking (and are therefore not reported):

    - ``title``, ``description``, ``examples`` changes — metadata only.
    - ``default`` changes on optional fields — only affects Python-side
      construction, not wire deserialization.
    - Enum value *removed* in ``new`` — the new server never produces that
      value, so old clients never see it.
    - New ``$defs`` entry that is not yet referenced anywhere — dead code.
    - New ``properties`` key on a model where ``additionalProperties`` is
      not ``false`` — the old client ignores extra fields.
    - Any change inside ``$defs`` entries that are not reachable from the
      top-level ``DoclingDocument`` properties.
    """
    findings: list[str] = []
    old_defs = old.get("$defs", {})
    new_defs = new.get("$defs", {})

    # --- 1 & 2 & 3: walk every $defs entry that exists in both schemas -------
    for def_name, new_def in new_defs.items():
        old_def = old_defs.get(def_name)
        if old_def is None:
            # Entirely new def — only breaking if it is immediately used in a
            # union (covered by check 4 below via the anyOf walk).
            continue

        # 1. Enum values added
        old_enum = set(old_def.get("enum", []))
        new_enum = set(new_def.get("enum", []))
        added_vals = new_enum - old_enum
        if added_vals:
            findings.append(
                f"New enum value(s) in {def_name!r}: {sorted(added_vals)}\n"
                f"    Old clients will raise on documents containing these values.\n"
                f"    -> Bump CURRENT_VERSION and add a downgrade projector that\n"
                f"       remaps these values to the nearest old equivalent."
            )

        # 2. New properties on strict models (additionalProperties: false)
        if old_def.get("additionalProperties") is False:
            old_props = set(old_def.get("properties", {}).keys())
            new_props = set(new_def.get("properties", {}).keys())
            added_props = new_props - old_props
            if added_props:
                findings.append(
                    f"New field(s) on strict model {def_name!r}: {sorted(added_props)}\n"
                    f"    Old clients have extra='forbid' and will raise on these fields.\n"
                    f"    -> Bump CURRENT_VERSION and add a downgrade projector that\n"
                    f"       strips these fields."
                )

        # 3. New required fields
        old_required = set(old_def.get("required", []))
        new_required = set(new_def.get("required", []))
        added_required = new_required - old_required
        if added_required:
            findings.append(
                f"New required field(s) in {def_name!r}: {sorted(added_required)}\n"
                f"    Old clients will raise if these fields are missing.\n"
                f"    -> Bump CURRENT_VERSION and add a downgrade projector that\n"
                f"       supplies safe defaults for these fields."
            )

    # --- 4. New branches in anyOf unions (new DocItem subtypes) ---------------
    def _collect_any_of_refs(schema_obj: dict, path: str = "") -> dict[str, set[str]]:
        """Return {json_path: {$ref, ...}} for every anyOf found."""
        result: dict[str, set[str]] = {}
        if isinstance(schema_obj, dict):
            if "anyOf" in schema_obj:
                refs = {branch["$ref"] for branch in schema_obj["anyOf"] if "$ref" in branch}
                if refs:
                    result[path] = refs
            for key, val in schema_obj.items():
                if key in ("title", "description", "examples", "default"):
                    continue
                result.update(_collect_any_of_refs(val, f"{path}.{key}"))
        elif isinstance(schema_obj, list):
            for i, item in enumerate(schema_obj):
                result.update(_collect_any_of_refs(item, f"{path}[{i}]"))
        return result

    old_unions = _collect_any_of_refs(old)
    new_unions = _collect_any_of_refs(new)

    for path, new_refs in new_unions.items():
        old_refs = old_unions.get(path, set())
        added_refs = new_refs - old_refs
        if added_refs:
            type_names = [r.split("/")[-1] for r in sorted(added_refs)]
            findings.append(
                f"New subtype(s) in union at {path!r}: {type_names}\n"
                f"    Old clients' discriminator union does not include these types\n"
                f"    and will raise on documents containing them.\n"
                f"    -> Bump CURRENT_VERSION and add a downgrade projector that\n"
                f"       converts or removes these items."
            )

    return findings


def main() -> int:
    errors: list[str] = []
    generated: list[str] = []

    # --- 1. Load the constants module ----------------------------------------
    try:
        from docling_core.types.doc.common.constants import (
            _CURRENT_MINOR,
            CURRENT_VERSION,
            FIRST_SUPPORTED_MINOR,
            SCHEMA_VERSION_HISTORY,
        )
    except ImportError as exc:
        print(f"ERROR: Could not import docling_core constants: {exc}", file=sys.stderr)
        return 1

    # --- 2. Load the compat module -------------------------------------------
    try:
        from docling_core.compat import _projectors
    except ImportError as exc:
        print(f"ERROR: Could not import docling_core.compat: {exc}", file=sys.stderr)
        return 1

    _ensure_schemas_dir()

    # --- 3. Projector count --------------------------------------------------
    expected_count = _CURRENT_MINOR - FIRST_SUPPORTED_MINOR
    actual_count = len(_projectors)
    if actual_count != expected_count:
        errors.append(
            f"Projector count mismatch: expected {expected_count} "
            f"(CURRENT_MINOR={_CURRENT_MINOR} - FIRST_SUPPORTED_MINOR={FIRST_SUPPORTED_MINOR}), "
            f"got {actual_count}.\n"
            f"  -> Add a @register_projector(from_minor={_CURRENT_MINOR}, to_minor={_CURRENT_MINOR - 1}) "
            f"function to docling_core/compat.py."
        )

    # --- 4. Every step covered -----------------------------------------------
    missing_steps = []
    for n in range(FIRST_SUPPORTED_MINOR + 1, _CURRENT_MINOR + 1):
        if (n, n - 1) not in _projectors:
            missing_steps.append(f"1.{n} -> 1.{n - 1}")
    if missing_steps:
        errors.append(
            f"Missing projectors for: {', '.join(missing_steps)}.\n"
            f"  -> Each step requires a @register_projector in docling_core/compat.py."
        )

    # --- 5. SCHEMA_VERSION_HISTORY up-to-date --------------------------------
    history_count = len(SCHEMA_VERSION_HISTORY)
    expected_history_count = _CURRENT_MINOR - FIRST_SUPPORTED_MINOR + 1
    if history_count != expected_history_count:
        errors.append(
            f"SCHEMA_VERSION_HISTORY has {history_count} entries; "
            f"expected {expected_history_count} "
            f"(one per schema version 1.{FIRST_SUPPORTED_MINOR}..1.{_CURRENT_MINOR}).\n"
            f"  -> Append an entry for CURRENT_VERSION={CURRENT_VERSION!r} in "
            f"docling_core/types/doc/common/constants.py."
        )

    # --- 6. Last history entry matches CURRENT_VERSION -----------------------
    if SCHEMA_VERSION_HISTORY:
        last = SCHEMA_VERSION_HISTORY[-1]["schema"]
        if last != CURRENT_VERSION:
            errors.append(
                f"Last SCHEMA_VERSION_HISTORY entry is {last!r}, "
                f"expected {CURRENT_VERSION!r}.\n"
                f"  -> The last entry must always match CURRENT_VERSION."
            )

    # --- 7. Projector keys are single-step downgrades ------------------------
    for key in _projectors:
        if key[0] != key[1] + 1:
            errors.append(f"Projector key {key} is not a single-step downgrade (from_minor must equal to_minor + 1).")

    # --- 8. JSON Schema snapshots --------------------------------------------
    # For every projector (from_minor -> to_minor), a snapshot for `to_minor`
    # must exist at docs/schemas/DoclingDocument_1_{to_minor}.json.
    #
    # Additionally, a snapshot for CURRENT_MINOR must exist (used to validate
    # the output of any *future* projector that targets the current version).
    #
    # Auto-generation strategy:
    #   - The snapshot for CURRENT_MINOR is produced by copying the live
    #     docs/DoclingDocument.json (which always = current schema).
    #   - Snapshots for older target versions (FIRST_SUPPORTED_MINOR..
    #     CURRENT_MINOR-1) cannot be auto-generated because their schemas
    #     predate this system.  If missing, we copy the live schema as a
    #     permissive bootstrap (the current schema is a superset of all older
    #     schemas, so it will accept — but not strictly validate — old dicts).
    #     A warning is printed so maintainers know these are bootstrap copies.
    #
    # Going forward, the correct flow is:
    #   1. Before bumping CURRENT_VERSION from 1.N to 1.(N+1):
    #      this script captures docs/DoclingDocument.json → DoclingDocument_1_N.json
    #   2. The contributor bumps CURRENT_VERSION.
    #   3. generate_docs regenerates docs/DoclingDocument.json for 1.(N+1).
    #   4. This script captures it as DoclingDocument_1_{N+1}.json.
    # This means each snapshot is the exact schema that clients at that version saw.

    all_target_minors = {to_minor for _, to_minor in _projectors} | {_CURRENT_MINOR}

    for minor in sorted(all_target_minors):
        snap = _snapshot_path(minor)
        if snap.exists():
            # Verify it is valid JSON.
            try:
                json.loads(snap.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"Snapshot {snap.name} is not valid JSON: {exc}")
            continue

        # Snapshot missing — attempt auto-generation.
        if not _LIVE_SCHEMA.exists():
            errors.append(
                f"Snapshot docs/schemas/DoclingDocument_1_{minor}.json is missing and "
                f"docs/DoclingDocument.json does not exist either — cannot auto-generate.\n"
                f"  -> Run `uv run python -m docling_core.utils.generate_docs docs` first."
            )
            continue

        snap.write_text(_LIVE_SCHEMA.read_text(), encoding="utf-8")

        if minor == _CURRENT_MINOR:
            generated.append(
                f"  docs/schemas/DoclingDocument_1_{minor}.json  (snapshot of current schema {CURRENT_VERSION})"
            )
        else:
            generated.append(
                f"  docs/schemas/DoclingDocument_1_{minor}.json  "
                f"[bootstrap copy of {CURRENT_VERSION} schema — "
                f"replace with exact 1.{minor} schema if available]"
            )

    # --- 9. Breaking schema diff against current snapshot --------------------
    # Compare docs/DoclingDocument.json (live, regenerated each commit by the
    # `docs` hook) against docs/schemas/DoclingDocument_1_{CURRENT_MINOR}.json
    # (the snapshot taken at the last CURRENT_VERSION bump).
    #
    # If the live schema has breaking changes relative to the snapshot AND
    # CURRENT_VERSION has not been bumped, the contributor forgot to announce
    # the breaking change.  We report exactly what changed and why it matters.
    #
    # If CURRENT_VERSION *was* bumped (the snapshot for CURRENT_MINOR was just
    # auto-generated in step 8 above and reflects the new schema), this diff
    # will be empty — the snapshot IS the current schema.
    current_snap = _snapshot_path(_CURRENT_MINOR)
    if _LIVE_SCHEMA.exists() and current_snap.exists():
        try:
            live_schema = json.loads(_LIVE_SCHEMA.read_text())
            snap_schema = json.loads(current_snap.read_text())
            breaking = _breaking_changes(old=snap_schema, new=live_schema)
            if breaking:
                msg = (
                    f"Breaking schema change(s) detected without a CURRENT_VERSION bump.\n"
                    f"  docs/DoclingDocument.json differs from "
                    f"docs/schemas/DoclingDocument_1_{_CURRENT_MINOR}.json "
                    f"in {len(breaking)} breaking way(s):\n\n"
                )
                for i, b in enumerate(breaking, 1):
                    msg += f"  [{i}] {b}\n\n"
                msg += (
                    "  If this change is intentional, follow the checklist in\n"
                    "  CONTRIBUTING.md §'Breaking changes and schema bumps':\n"
                    "  bump CURRENT_VERSION, add a projector, and add a test class."
                )
                errors.append(msg)
        except json.JSONDecodeError:
            pass  # corrupt snapshot — already caught in step 8

    # --- Report --------------------------------------------------------------
    if errors:
        print("FAIL: docling_core/compat.py projector coverage check\n", file=sys.stderr)
        for i, err in enumerate(errors, 1):
            print(f"  [{i}] {err}\n", file=sys.stderr)
        print(
            "See the contributor guide in CONTRIBUTING.md §'Breaking changes and schema bumps'.",
            file=sys.stderr,
        )
        return 1

    if generated:
        print(
            f"OK: {actual_count} projector(s) registered. Auto-generated {len(generated)} schema snapshot(s):",
            file=sys.stdout,
        )
        for msg in generated:
            print(msg, file=sys.stdout)
        print(
            "  -> Add the new snapshot file(s) to your commit.",
            file=sys.stdout,
        )
    else:
        print(
            f"OK: {actual_count} projector(s) registered for "
            f"schema 1.{FIRST_SUPPORTED_MINOR}..1.{_CURRENT_MINOR}, "
            f"all snapshots present.",
            file=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
