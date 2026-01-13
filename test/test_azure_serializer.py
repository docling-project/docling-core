"""Tests for AzureDocSerializer."""

import json
from pathlib import Path
from typing import Any

from docling_core.transforms.serializer.azure import AzureDocSerializer, AzureParams
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)

from .test_data_gen_flag import GEN_TEST_DATA


def _assert_json_like_equal(a: Any, b: Any, eps: float = 1e-3, path: str = "$") -> None:
    """Recursively assert two JSON-like structures are equal with float tolerance.

    Rules:
    - Dicts: same keys; compare values recursively
    - Lists: same length; compare elements in order
    - Numbers: ints compare by ==; floats (or int-vs-float) compare within eps
    - Strings/bools/null: compare by ==

    The `path` argument is used to provide helpful assertion locations.
    """
    # Handle dicts
    if isinstance(a, dict) and isinstance(b, dict):
        a_keys, b_keys = set(a.keys()), set(b.keys())
        assert a_keys == b_keys, f"Key mismatch at {path}: {a_keys ^ b_keys}"
        for k in sorted(a_keys):
            _assert_json_like_equal(a[k], b[k], eps=eps, path=f"{path}.{k}")
        return

    # Handle lists
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"List length mismatch at {path}: {len(a)} != {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            _assert_json_like_equal(ai, bi, eps=eps, path=f"{path}[{i}]")
        return

    # Handle numbers (int/float)
    num_types = (int, float)
    if isinstance(a, num_types) and isinstance(b, num_types):
        # If either is float, compare with tolerance; if both int, exact match
        if isinstance(a, float) or isinstance(b, float):
            diff = abs(float(a) - float(b))
            assert diff <= eps, f"Float mismatch at {path}: {a} != {b} (diff={diff}, eps={eps})"
        else:
            assert a == b, f"Int mismatch at {path}: {a} != {b}"
        return

    # Handle strings, bools, and None
    if isinstance(a, (str, bool)) or a is None:
        assert a == b, f"Value mismatch at {path}: {a!r} != {b!r}"
        return

    # Fallback: types must match and then equality
    assert type(a) is type(b), f"Type mismatch at {path}: {type(a)} != {type(b)}"
    assert a == b, f"Mismatch at {path}: {a!r} != {b!r}"


def _verify_json(exp_file: Path, actual_json: str) -> None:
    """Verify Azure JSON string against ground-truth file with generation support.

    Compares parsed JSON structures and tolerates tiny float differences.
    """
    if GEN_TEST_DATA:
        # Keep writing the canonical serialized string for determinism
        exp_file.write_text(actual_json + "\n", encoding="utf-8")
    else:
        expected_text = exp_file.read_text(encoding="utf-8").rstrip()
        expected_obj = json.loads(expected_text)
        actual_obj = json.loads(actual_json)
        _assert_json_like_equal(expected_obj, actual_obj, eps=1e-3)


def test_azure_serialize_activities_doc():
    """Serialize a GT document (activities.json) and verify Azure JSON output."""
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = AzureDocSerializer(doc=doc, params=AzureParams(indent=2))
    actual_json = ser.serialize().text

    # Sanity-check the JSON structure
    data = json.loads(actual_json)
    assert isinstance(data, dict)
    assert "pages" in data and isinstance(data["pages"], list)
    assert "tables" in data and isinstance(data["tables"], list)
    assert "figures" in data and isinstance(data["figures"], list)
    assert "paragraphs" in data and isinstance(data["paragraphs"], list)

    _verify_json(exp_file=src.with_suffix(".gt.azure.json"), actual_json=actual_json)


def test_azure_serialize_construct_doc_minimal_prov(sample_doc: DoclingDocument):
    """Serialize a constructed document with minimal provenance to Azure JSON.

    The sample_doc fixture does not attach provenance or pages; here we add a
    single page and minimal bounding boxes to a subset of items to allow Azure JSON
    output to include paragraphs/tables/pictures with boundingRegions.
    """

    # Ensure at least one page is present
    if not sample_doc.pages:
        sample_doc.add_page(page_no=1, size=Size(width=600.0, height=800.0), image=None)

    # Helper to add a simple TOPLEFT bbox provenance if missing
    def _ensure_prov(item, l=10.0, t=10.0, r=200.0, b=40.0):
        if not item.prov:
            item.prov = [
                ProvenanceItem(
                    page_no=min(sample_doc.pages.keys()),
                    bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT),
                    charspan=(0, 0),
                )
            ]

    # Add provenance for the title and a couple of paragraphs if present
    for it in sample_doc.texts[:3]:
        if it.label in {
            DocItemLabel.TITLE,
            DocItemLabel.TEXT,
            DocItemLabel.SECTION_HEADER,
        }:
            _ensure_prov(it)

    # Add provenance for the first table if present
    if sample_doc.tables:
        _ensure_prov(sample_doc.tables[0], l=20.0, t=80.0, r=300.0, b=200.0)

    # Add provenance for the first picture if present
    if sample_doc.pictures:
        _ensure_prov(sample_doc.pictures[0], l=320.0, t=80.0, r=500.0, b=220.0)

    ser = AzureDocSerializer(doc=sample_doc, params=AzureParams(indent=2))
    actual_json = ser.serialize().text

    # Basic structure check
    data = json.loads(actual_json)
    assert isinstance(data, dict)
    assert "pages" in data and isinstance(data["pages"], list) and len(data["pages"]) >= 1
    assert "paragraphs" in data and isinstance(data["paragraphs"], list)

    exp_file = Path("./test/data/doc/constructed.gt.azure.json")
    _verify_json(exp_file=exp_file, actual_json=actual_json)
