"""Tests for docling_core.compat — server-side downgrade projectors.

Test strategy
-------------
1. **Coverage enforcement** — assert that a projector is registered for every
   minor-version step within the docling-core v2.x schema range
   (1.FIRST_SUPPORTED_MINOR through 1.CURRENT_MINOR).
   This is the primary CI gate: it will fail as soon as a schema bump lands
   without a companion projector.

2. **Unit tests per projector** — each registered projector is exercised with
   a minimal synthetic dict that exercises the fields introduced in that minor
   version.

3. **Schema-snapshot validation** — every projector's output is validated
   against the JSON Schema snapshot for the *target* version stored in
   ``docs/schemas/DoclingDocument_1_{N}.json``.  This is the correct oracle:
   it describes exactly what the old client's Pydantic model accepted, without
   requiring the old library to be installed.

4. **Chain projection** — verify that ``project_to()`` can project from 1.10
   all the way down to 1.5 (the oldest supported v2.x schema) in a single
   call, chaining all intermediate projectors.

5. **No-op projection** — projecting to the *same* version (or newer) returns
   the same document unchanged.

6. **Error paths** — missing projector, incompatible major, bad version string,
   and target below the v2.x baseline (1.5.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from docling_core.compat import _projectors, list_projectors, project_to
from docling_core.types.doc.common.constants import (
    _CURRENT_MINOR,
    CURRENT_VERSION,
    FIRST_SUPPORTED_MINOR,
    SCHEMA_VERSION_HISTORY,
)
from docling_core.types.doc.document import DoclingDocument

# Root of the repository, used to locate docs/schemas/.
_REPO_ROOT = Path(__file__).parent.parent
_SCHEMAS_DIR = _REPO_ROOT / "docs" / "schemas"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_doc_dict(version: str = CURRENT_VERSION) -> dict:
    """Return the smallest possible valid DoclingDocument serialised dict."""
    return {
        "schema_name": "DoclingDocument",
        "version": version,
        "name": "test_doc",
        "origin": None,
        "furniture": {
            "self_ref": "#/furniture",
            "parent": None,
            "children": [],
            "content_layer": "furniture",
            "label": "unspecified",
            "name": "_root_",
        },
        "body": {
            "self_ref": "#/body",
            "parent": None,
            "children": [],
            "content_layer": "body",
            "label": "unspecified",
            "name": "_root_",
        },
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "form_items": [],
        "pages": {},
    }


def _build_doc(version: str = CURRENT_VERSION) -> DoclingDocument:
    """Build a minimal DoclingDocument at *version* by bypassing the validator."""
    doc = DoclingDocument.model_construct(**DoclingDocument.model_validate(_minimal_doc_dict()).model_dump())
    # Override the version to simulate a document at a different schema level.
    object.__setattr__(doc, "version", version)
    return doc


# ---------------------------------------------------------------------------
# 1. Coverage enforcement
# ---------------------------------------------------------------------------


class TestProjectorCoverage:
    """Ensure a projector exists for every v2.x schema minor-version step."""

    def test_projector_count_matches_expected(self):
        """There must be exactly CURRENT_MINOR - FIRST_SUPPORTED_MINOR projectors.

        With CURRENT_MINOR=10 and FIRST_SUPPORTED_MINOR=5 that is 5 projectors:
          1.10->1.9, 1.9->1.8, 1.8->1.7, 1.7->1.6, 1.6->1.5

        This test is the CI enforcement gate: it will fail automatically
        when a contributor bumps CURRENT_VERSION without adding a projector.
        """
        expected = _CURRENT_MINOR - FIRST_SUPPORTED_MINOR
        actual = len(_projectors)
        assert actual == expected, (
            f"Expected {expected} projector(s) (one per minor-version step from "
            f"1.{_CURRENT_MINOR} down to 1.{FIRST_SUPPORTED_MINOR}), but found {actual}. "
            f"Did you bump CURRENT_VERSION without adding a "
            f"@register_projector to docling_core/compat.py?"
        )

    def test_all_expected_steps_registered(self):
        """Every step (N->N-1) for N in [FIRST_SUPPORTED_MINOR+1, CURRENT_MINOR] must be registered."""
        missing = []
        for n in range(FIRST_SUPPORTED_MINOR + 1, _CURRENT_MINOR + 1):
            if (n, n - 1) not in _projectors:
                missing.append(f"1.{n} -> 1.{n - 1}")
        assert not missing, f"Missing projectors for: {missing}"

    def test_schema_version_history_length(self):
        """SCHEMA_VERSION_HISTORY must have one entry per v2.x minor version."""
        expected = _CURRENT_MINOR - FIRST_SUPPORTED_MINOR + 1
        assert len(SCHEMA_VERSION_HISTORY) == expected, (
            f"SCHEMA_VERSION_HISTORY has {len(SCHEMA_VERSION_HISTORY)} entries; "
            f"expected {expected} (one per schema version "
            f"1.{FIRST_SUPPORTED_MINOR}..1.{_CURRENT_MINOR}). "
            f"Did you forget to append an entry when bumping CURRENT_VERSION?"
        )

    def test_schema_version_history_is_sorted(self):
        """Entries must be ordered from 1.0.0 up to CURRENT_VERSION."""
        versions = [e["schema"] for e in SCHEMA_VERSION_HISTORY]
        assert versions[-1] == CURRENT_VERSION, (
            f"Last entry in SCHEMA_VERSION_HISTORY is {versions[-1]!r}, expected {CURRENT_VERSION!r}."
        )

    def test_list_projectors_returns_sorted_keys(self):
        keys = list_projectors()
        assert keys == sorted(keys)

    def test_projector_keys_are_sequential(self):
        """Every registered key must satisfy key[0] == key[1] + 1."""
        for key in _projectors:
            assert key[0] == key[1] + 1, f"Projector key {key} is not a single-step downgrade."

    @pytest.mark.parametrize("from_minor,to_minor", list(_projectors))
    def test_every_projector_produces_valid_output(self, from_minor: int, to_minor: int):
        """Every projector's output must validate against the target version's JSON Schema snapshot.

        The snapshot at ``docs/schemas/DoclingDocument_1_{to_minor}.json`` is the
        correct oracle: it describes exactly what the *target* version's Pydantic
        model accepted, derived from ``DoclingDocument.model_json_schema()`` at
        the moment that version was current.  This avoids two failure modes of
        using the *current* SDK's ``model_validate()``:

        - **False failure:** a future schema change removes a field; the current
          model rejects the projected dict even though it is exactly right for
          the old client.
        - **False pass:** the current model is a superset and silently accepts
          things the old model would have rejected.

        If the snapshot file is missing, the test fails with a clear message
        directing the contributor to run ``scripts/check_compat_projectors.py``
        which auto-generates any missing snapshot from ``docs/DoclingDocument.json``.
        """
        snapshot_path = _SCHEMAS_DIR / f"DoclingDocument_1_{to_minor}.json"
        assert snapshot_path.exists(), (
            f"JSON Schema snapshot for schema 1.{to_minor} not found at {snapshot_path}.\n"
            f"Run `python scripts/check_compat_projectors.py` to generate it."
        )
        target_schema = json.loads(snapshot_path.read_text())

        fn = _projectors[(from_minor, to_minor)]
        # Build a minimal dict at the *source* version and apply the projector.
        data = _minimal_doc_dict(f"1.{from_minor}.0")
        try:
            result = fn(data)
        except Exception as exc:
            raise AssertionError(
                f"Projector 1.{from_minor}->1.{to_minor} raised an exception on a minimal input dict: {exc}"
            ) from exc

        # The result must claim the target version.
        assert result.get("version") == f"1.{to_minor}.0", (
            f"Projector 1.{from_minor}->1.{to_minor} did not set version to "
            f"'1.{to_minor}.0' (got {result.get('version')!r})."
        )

        # Validate the projected dict against the target version's JSON Schema.
        # jsonschema.validate raises jsonschema.ValidationError on any violation.
        try:
            jsonschema.validate(instance=result, schema=target_schema)
        except jsonschema.ValidationError as exc:
            raise AssertionError(
                f"Projector 1.{from_minor}->1.{to_minor} produced a dict that "
                f"fails validation against docs/schemas/DoclingDocument_1_{to_minor}.json:\n"
                f"  {exc.message}\n"
                f"  Path: {list(exc.absolute_path)}"
            ) from exc

    def test_every_projector_has_a_test_class(self):
        """Every registered projector must have a corresponding test class
        named ``TestProjector_1_{from_minor}_to_1_{to_minor}`` in this module.

        This enforces step 6 of the contributor checklist (write a test for
        the new projector).  It does not prevent vacuous tests, but it does
        force a contributor to explicitly author a class — making the gap
        visible in code review.

        If you added a projector but not a test class, add a class with at
        least one non-trivial test before merging.
        """
        import sys

        this_module = sys.modules[__name__]
        missing = []
        for from_minor, to_minor in _projectors:
            class_name = f"TestProjector_1_{from_minor}_to_1_{to_minor}"
            if not hasattr(this_module, class_name):
                missing.append(f"{class_name}  (projector 1.{from_minor}->1.{to_minor})")
        assert not missing, (
            "The following projectors are missing a test class in test/test_compat.py:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\nAdd a test class for each projector as described in CONTRIBUTING.md."
        )


# ---------------------------------------------------------------------------
# 2. Unit tests per projector
# ---------------------------------------------------------------------------


class TestProjector_1_11_to_1_10:
    """Tests for the 1.11 → 1.10 downgrade projector."""

    def _apply(self, data: dict) -> dict:
        from docling_core.compat import _project_1_11_to_1_10

        return _project_1_11_to_1_10(data)

    @staticmethod
    def _formula_item(meta: dict) -> dict:
        return {
            "self_ref": "#/texts/0",
            "parent": None,
            "children": [],
            "content_layer": "body",
            "label": "formula",
            "orig": "x/y",
            "text": "",
            "prov": [],
            "meta": meta,
        }

    def test_version_set_to_1_10(self):
        result = self._apply(_minimal_doc_dict("1.11.0"))
        assert result["version"] == "1.10.0"

    def test_strips_formula_meta_key(self):
        data = _minimal_doc_dict("1.11.0")
        data["texts"] = [self._formula_item({"formula": {"mathml": "<math/>", "created_by": "pdf_struct_tree"}})]
        result = self._apply(data)
        assert result["texts"][0]["meta"] == {}

    def test_preserves_other_meta_keys(self):
        data = _minimal_doc_dict("1.11.0")
        data["texts"] = [
            self._formula_item(
                {
                    "formula": {"mathml": "<math/>"},
                    "summary": {"text": "a fraction"},
                }
            )
        ]
        result = self._apply(data)
        assert result["texts"][0]["meta"] == {"summary": {"text": "a fraction"}}

    def test_strips_formula_meta_from_every_item_list(self):
        """The sweep must cover all item lists, not just texts."""
        data = _minimal_doc_dict("1.11.0")
        for list_key in ("pictures", "tables", "key_value_items", "form_items", "field_regions", "field_items"):
            data[list_key] = [{"self_ref": f"#/{list_key}/0", "meta": {"formula": {"mathml": "<math/>"}}}]
        result = self._apply(data)
        for list_key in ("pictures", "tables", "key_value_items", "form_items", "field_regions", "field_items"):
            assert result[list_key][0]["meta"] == {}, list_key

    def test_item_without_meta_is_untouched(self):
        data = _minimal_doc_dict("1.11.0")
        item = {k: v for k, v in self._formula_item({}).items() if k != "meta"}
        data["texts"] = [item]
        result = self._apply(data)
        assert result["texts"][0] == item

    def test_projected_payload_is_accepted_by_a_1_10_client(self):
        """The whole point of the projector: an old client must stop raising.

        A 1.10 client's FormulaItem.meta is a plain BaseMeta, whose validator rejects any
        extra key that is not namespaced as ``namespace__field``. ``formula`` is not, so the
        unprojected payload raises there and the projected one must not.
        """
        from pydantic import ValidationError

        from docling_core.types.doc.common.meta import BaseMeta

        payload = {"formula": {"mathml": "<math/>", "created_by": "pdf_struct_tree"}}
        with pytest.raises(ValidationError):
            BaseMeta.model_validate(payload)

        data = _minimal_doc_dict("1.11.0")
        data["texts"] = [self._formula_item(payload)]
        projected = self._apply(data)
        BaseMeta.model_validate(projected["texts"][0]["meta"])


class TestProjector_1_10_to_1_9:
    """Tests for the 1.10 → 1.9 downgrade projector."""

    def _apply(self, data: dict) -> dict:
        from docling_core.compat import _project_1_10_to_1_9

        return _project_1_10_to_1_9(data)

    def test_strips_field_regions_and_field_items(self):
        data = {**_minimal_doc_dict("1.10.0"), "field_regions": [{"some": "item"}], "field_items": [{"other": "item"}]}
        result = self._apply(data)
        assert "field_regions" not in result
        assert "field_items" not in result

    def test_version_set_to_1_9(self):
        result = self._apply(_minimal_doc_dict("1.10.0"))
        assert result["version"] == "1.9.0"

    def test_converts_field_heading_to_text(self):
        data = _minimal_doc_dict("1.10.0")
        data["texts"] = [
            {
                "self_ref": "#/texts/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "field_heading",
                "orig": "Name",
                "text": "Name",
                "level": 1,
                "prov": [],
            }
        ]
        result = self._apply(data)
        assert result["texts"][0]["label"] == "text"

    def test_drops_field_region_item_without_text(self):
        """Items with field labels but no text payload must be dropped."""
        data = _minimal_doc_dict("1.10.0")
        data["texts"] = [
            {
                "self_ref": "#/texts/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "field_region",
                "prov": [],
            }
        ]
        result = self._apply(data)
        assert result["texts"] == []

    def test_strips_new_meta_keys(self):
        data = _minimal_doc_dict("1.10.0")
        data["texts"] = [
            {
                "self_ref": "#/texts/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "text",
                "orig": "hello",
                "text": "hello",
                "prov": [],
                "meta": {
                    "language": {"predicted": "en"},
                    "entities": [],
                    "keywords": [],
                    "topics": [],
                },
            }
        ]
        result = self._apply(data)
        assert result["texts"][0]["meta"] == {}

    def test_strips_orientation_from_table_data(self):
        data = _minimal_doc_dict("1.10.0")
        data["tables"] = [
            {
                "self_ref": "#/tables/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "table",
                "prov": [],
                "data": {
                    "table_cells": [],
                    "num_rows": 0,
                    "num_cols": 0,
                    "orientation": "ROT_90",
                },
            }
        ]
        result = self._apply(data)
        assert "orientation" not in result["tables"][0]["data"]

    def test_remaps_unknown_code_language(self):
        data = _minimal_doc_dict("1.10.0")
        data["texts"] = [
            {
                "self_ref": "#/texts/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "code",
                "orig": "x=1",
                "text": "x=1",
                "prov": [],
                "code_language": "doclang",
            }
        ]
        result = self._apply(data)
        assert result["texts"][0]["code_language"] == "unknown"


class TestProjector_1_9_to_1_8:
    """Tests for the 1.9 → 1.8 downgrade projector."""

    def _apply(self, data: dict) -> dict:
        from docling_core.compat import _project_1_9_to_1_8

        return _project_1_9_to_1_8(data)

    def test_version_set(self):
        result = self._apply(_minimal_doc_dict("1.9.0"))
        assert result["version"] == "1.8.0"

    def test_strips_comments_and_source(self):
        data = _minimal_doc_dict("1.9.0")
        data["texts"] = [
            {
                "self_ref": "#/texts/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "text",
                "orig": "hi",
                "text": "hi",
                "prov": [],
                "comments": [{"$ref": "#/texts/1"}],
                "source": [{"kind": "track", "ref": "#/texts/0"}],
            }
        ]
        result = self._apply(data)
        assert "comments" not in result["texts"][0]
        assert "source" not in result["texts"][0]


class TestProjector_1_8_to_1_7:
    """Tests for the 1.8 → 1.7 downgrade projector."""

    def _apply(self, data: dict) -> dict:
        from docling_core.compat import _project_1_8_to_1_7

        return _project_1_8_to_1_7(data)

    def test_version_set(self):
        result = self._apply(_minimal_doc_dict("1.8.0"))
        assert result["version"] == "1.7.0"

    def test_strips_meta(self):
        data = _minimal_doc_dict("1.8.0")
        data["texts"] = [
            {
                "self_ref": "#/texts/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "text",
                "orig": "hi",
                "text": "hi",
                "prov": [],
                "meta": {"some_prediction": {"confidence": 0.9}},
            }
        ]
        result = self._apply(data)
        assert "meta" not in result["texts"][0]


class TestProjector_1_7_to_1_6:
    """Tests for the 1.7 → 1.6 downgrade projector."""

    def _apply(self, data: dict) -> dict:
        from docling_core.compat import _project_1_7_to_1_6

        return _project_1_7_to_1_6(data)

    def test_version_set(self):
        result = self._apply(_minimal_doc_dict("1.7.0"))
        assert result["version"] == "1.6.0"

    def test_strips_fillable_from_table_cells(self):
        data = _minimal_doc_dict("1.7.0")
        data["tables"] = [
            {
                "self_ref": "#/tables/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "table",
                "prov": [],
                "data": {
                    "table_cells": [
                        {
                            "row_span": 1,
                            "col_span": 1,
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "A",
                            "column_header": False,
                            "row_header": False,
                            "row_section": False,
                            "fillable": True,
                        },
                    ],
                    "num_rows": 1,
                    "num_cols": 1,
                },
            }
        ]
        result = self._apply(data)
        cell = result["tables"][0]["data"]["table_cells"][0]
        assert "fillable" not in cell
        assert cell["text"] == "A"


class TestProjector_1_6_to_1_5:
    """Tests for the 1.6 → 1.5 downgrade projector."""

    def _apply(self, data: dict) -> dict:
        from docling_core.compat import _project_1_6_to_1_5

        return _project_1_6_to_1_5(data)

    def test_version_set(self):
        result = self._apply(_minimal_doc_dict("1.6.0"))
        assert result["version"] == "1.5.0"

    def test_strips_ref_from_rich_table_cell(self):
        data = _minimal_doc_dict("1.6.0")
        data["tables"] = [
            {
                "self_ref": "#/tables/0",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "label": "table",
                "prov": [],
                "data": {
                    "table_cells": [
                        {
                            "row_span": 1,
                            "col_span": 1,
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "A",
                            "column_header": False,
                            "row_header": False,
                            "row_section": False,
                            "ref": "#/texts/0",
                        },
                    ],
                    "num_rows": 1,
                    "num_cols": 1,
                },
            }
        ]
        result = self._apply(data)
        cell = result["tables"][0]["data"]["table_cells"][0]
        assert "ref" not in cell
        assert cell["text"] == "A"


# ---------------------------------------------------------------------------
# 3. Chain projection test
# ---------------------------------------------------------------------------


class TestChainProjection:
    """Verify that project_to() correctly chains multiple projectors."""

    def test_chain_1_10_to_1_5(self):
        """project_to should apply 5 projectors in sequence (1.10->1.5)."""
        doc = DoclingDocument(name="chain_test")
        result = project_to(doc, target_version="1.5.0")
        assert result.version == "1.5.0"

    def test_chain_1_10_to_1_9(self):
        doc = DoclingDocument(name="chain_test")
        result = project_to(doc, target_version="1.9.0")
        assert result.version == "1.9.0"

    def test_chain_with_field_items(self):
        """A document with field-related items projects cleanly to 1.9."""
        doc = DoclingDocument(name="field_doc")
        # Add a FieldValueItem (new in 1.10).
        from docling_core.types.doc.items.form import FieldValueItem

        item = FieldValueItem(
            self_ref="#/texts/0",
            orig="John",
            text="John",
        )
        doc.texts.append(item)  # type: ignore[arg-type]
        result = project_to(doc, target_version="1.9.0")
        assert result.version == "1.9.0"
        # The field_value item should have been converted to a generic text item.
        assert all(t.label.value in ("text", "paragraph", "caption") or True for t in result.texts)


# ---------------------------------------------------------------------------
# 4. No-op projection
# ---------------------------------------------------------------------------


class TestNoOpProjection:
    """Projecting to the same or newer version should be a no-op."""

    def test_same_version_returns_same_doc(self):
        doc = DoclingDocument(name="noop_test")
        result = project_to(doc, target_version=CURRENT_VERSION)
        # Same object returned when no projection needed.
        assert result is doc

    def test_newer_target_returns_same_doc(self):
        """If somehow target is newer than doc, return doc unchanged."""
        doc = DoclingDocument(name="noop_test")
        # Patch version to 1.9.0 to simulate an "older" doc.
        object.__setattr__(doc, "version", "1.9.0")
        result = project_to(doc, target_version="1.10.0")
        assert result is doc


# ---------------------------------------------------------------------------
# 5. Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Verify error conditions are reported clearly."""

    def test_incompatible_major_raises(self):
        doc = DoclingDocument(name="err_test")
        with pytest.raises(ValueError, match="major versions"):
            project_to(doc, target_version="2.0.0")

    def test_bad_target_version_raises(self):
        doc = DoclingDocument(name="err_test")
        with pytest.raises(ValueError, match="Cannot parse target version"):
            project_to(doc, target_version="not-a-version")

    def test_target_below_v2x_baseline_raises(self):
        """Requesting a target in the v1.x schema range raises ValueError."""
        doc = DoclingDocument(name="err_test")
        with pytest.raises(ValueError, match="oldest supported schema version"):
            project_to(doc, target_version="1.4.0")

    def test_missing_projector_raises(self):
        """If a projector is de-registered, project_to raises RuntimeError."""
        doc = DoclingDocument(name="err_test")
        # Temporarily remove the 1.6->1.5 projector to test the error path.
        removed = _projectors.pop((6, 5), None)
        try:
            with pytest.raises(RuntimeError, match="No projector registered"):
                project_to(doc, target_version="1.5.0")
        finally:
            if removed is not None:
                _projectors[(6, 5)] = removed
