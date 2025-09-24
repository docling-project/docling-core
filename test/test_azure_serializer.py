"""Tests for AzureDocSerializer."""

import json
import os
from pathlib import Path

from docling_core.transforms.serializer.azure import AzureDocSerializer
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import DocItemLabel, DoclingDocument, ProvenanceItem

from .test_data_gen_flag import GEN_TEST_DATA
from .test_docling_doc import _construct_doc


def _verify_json(exp_file: Path, actual_json: str) -> None:
    """Verify Azure JSON string against ground-truth file with generation support."""
    if GEN_TEST_DATA or not exp_file.exists():
        exp_file.write_text(actual_json + "\n", encoding="utf-8")
    else:
        expected = exp_file.read_text(encoding="utf-8").rstrip()
        assert expected == actual_json


def test_azure_serialize_activities_doc():
    """Serialize a GT document (activities.json) and verify Azure JSON output."""
    src = Path("./test/data/doc/activities.json")
    doc = DoclingDocument.load_from_json(src)

    ser = AzureDocSerializer(doc=doc)
    actual_json = ser.serialize().text

    # Sanity-check the JSON structure
    data = json.loads(actual_json)
    assert isinstance(data, dict)
    assert "pages" in data and isinstance(data["pages"], list)
    assert "tables" in data and isinstance(data["tables"], list)
    assert "figures" in data and isinstance(data["figures"], list)
    assert "paragraphs" in data and isinstance(data["paragraphs"], list)

    _verify_json(exp_file=src.with_suffix(".gt.azure.json"), actual_json=actual_json)


def test_azure_serialize_construct_doc_minimal_prov():
    """Serialize a constructed document with minimal provenance to Azure JSON.

    The _construct_doc() builder does not attach provenance or pages; here we add a
    single page and minimal bounding boxes to a subset of items to allow Azure JSON
    output to include paragraphs/tables/pictures with boundingRegions.
    """
    doc = _construct_doc()

    # Ensure at least one page is present
    if not doc.pages:
        doc.add_page(page_no=1, size=Size(width=600.0, height=800.0), image=None)

    # Helper to add a simple TOPLEFT bbox provenance if missing
    def _ensure_prov(item, l=10.0, t=10.0, r=200.0, b=40.0):
        if not item.prov:
            item.prov = [
                ProvenanceItem(
                    page_no=min(doc.pages.keys()),
                    bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT),
                    charspan=(0, 0),
                )
            ]

    # Add provenance for the title and a couple of paragraphs if present
    for it in doc.texts[:3]:
        if it.label in {DocItemLabel.TITLE, DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER}:
            _ensure_prov(it)

    # Add provenance for the first table if present
    if doc.tables:
        _ensure_prov(doc.tables[0], l=20.0, t=80.0, r=300.0, b=200.0)

    # Add provenance for the first picture if present
    if doc.pictures:
        _ensure_prov(doc.pictures[0], l=320.0, t=80.0, r=500.0, b=220.0)

    ser = AzureDocSerializer(doc=doc)
    actual_json = ser.serialize().text

    # Basic structure check
    data = json.loads(actual_json)
    assert isinstance(data, dict)
    assert "pages" in data and isinstance(data["pages"], list) and len(data["pages"]) >= 1
    assert "paragraphs" in data and isinstance(data["paragraphs"], list)

    exp_file = Path("./test/data/doc/constructed_doc.gt.azure.json")
    _verify_json(exp_file=exp_file, actual_json=actual_json)

