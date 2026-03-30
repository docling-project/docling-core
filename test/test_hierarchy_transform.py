"""Tests for DocHierarchyTransform."""

import json
from pathlib import Path

import pytest

from docling_core.transforms.hierarchy import DocHierarchyTransform
from docling_core.types.doc.document import (
    DoclingDocument,
    ListGroup,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

DOC_PATH = Path("test/data/doc/2408.09869v5_enriched_summary.json")


@pytest.fixture(scope="module")
def sample_doc() -> DoclingDocument:
    with open(DOC_PATH, encoding="utf-8") as f:
        return DoclingDocument.model_validate(json.load(f))


# ---------------------------------------------------------------------------
# to_flat
# ---------------------------------------------------------------------------


def test_to_flat_returns_docling_document(sample_doc):
    flat = DocHierarchyTransform.to_flat(sample_doc)
    assert isinstance(flat, DoclingDocument)
    assert flat.name == sample_doc.name


def test_to_flat_all_items_are_direct_body_children(sample_doc):
    flat = DocHierarchyTransform.to_flat(sample_doc)
    body_child_refs = set(ref.cref for ref in (flat.body.children or []))

    for item, _ in flat.iterate_items(with_groups=True):
        # Every non-body item must be a direct child of body or a child of an
        # atomic unit (ListGroup, TableItem, PictureItem).
        if item.parent is None:
            continue
        parent = item.parent.resolve(flat)
        assert isinstance(
            parent,
            (type(flat.body), ListGroup, TableItem, PictureItem),
        ), (
            f"Item {item.self_ref} has parent {parent.self_ref!r} of type "
            f"{type(parent).__name__}, expected body or an atomic container"
        )


def test_to_flat_preserves_section_header_levels(sample_doc):
    flat = DocHierarchyTransform.to_flat(sample_doc)

    original_levels = [
        item.level
        for item, _ in sample_doc.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]
    flat_levels = [
        item.level
        for item, _ in flat.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]

    assert original_levels == flat_levels, (
        "SectionHeaderItem levels must be identical after flattening"
    )


def test_to_flat_item_counts_match(sample_doc):
    """All item types should have the same count before and after flattening."""
    flat = DocHierarchyTransform.to_flat(sample_doc)

    def count_by_type(doc, cls):
        # Use exact type match for TextItem so that subclasses (TitleItem,
        # SectionHeaderItem, ListItem, …) are not counted twice.
        match = (lambda item: type(item) is cls) if cls is TextItem else (lambda item: isinstance(item, cls))
        return sum(1 for item, _ in doc.iterate_items(with_groups=True) if match(item))

    for cls in (SectionHeaderItem, TitleItem, TableItem, PictureItem, ListGroup, ListItem, TextItem):
        assert count_by_type(flat, cls) == count_by_type(sample_doc, cls), (
            f"Item count mismatch for {cls.__name__} after to_flat"
        )


def test_to_flat_is_idempotent(sample_doc):
    """Flattening an already-flat document must leave it unchanged."""
    flat_once = DocHierarchyTransform.to_flat(sample_doc)
    flat_twice = DocHierarchyTransform.to_flat(flat_once)

    levels_once = [
        item.level
        for item, _ in flat_once.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]
    levels_twice = [
        item.level
        for item, _ in flat_twice.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]
    assert levels_once == levels_twice

    children_once = len(flat_once.body.children or [])
    children_twice = len(flat_twice.body.children or [])
    assert children_once == children_twice


# ---------------------------------------------------------------------------
# to_hierarchical
# ---------------------------------------------------------------------------


def test_to_hierarchical_returns_docling_document(sample_doc):
    hier = DocHierarchyTransform.to_hierarchical(sample_doc)
    assert isinstance(hier, DoclingDocument)
    assert hier.name == sample_doc.name


def test_to_hierarchical_has_section_headers_with_children(sample_doc):
    hier = DocHierarchyTransform.to_hierarchical(sample_doc)

    total = sum(
        1
        for item, _ in hier.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    )
    with_children = sum(
        1
        for item, _ in hier.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem) and item.children
    )

    assert total > 0, "Document should contain section headers"
    assert with_children > 0, "At least some section headers should have children"


def test_to_hierarchical_body_has_fewer_direct_children_than_flat(sample_doc):
    flat = DocHierarchyTransform.to_flat(sample_doc)
    hier = DocHierarchyTransform.to_hierarchical(sample_doc)

    flat_count = len(flat.body.children or [])
    hier_count = len(hier.body.children or [])

    assert hier_count <= flat_count, (
        "Hierarchical document should have fewer or equal direct body children"
    )


def test_to_hierarchical_preserves_section_header_levels(sample_doc):
    hier = DocHierarchyTransform.to_hierarchical(sample_doc)

    original_levels = [
        item.level
        for item, _ in sample_doc.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]
    hier_levels = [
        item.level
        for item, _ in hier.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]

    assert original_levels == hier_levels


def test_to_hierarchical_section_nesting_respects_levels(sample_doc):
    """A section header's parent section must have a strictly lower level."""
    hier = DocHierarchyTransform.to_hierarchical(sample_doc)

    for item, _ in hier.iterate_items(with_groups=True):
        if not isinstance(item, SectionHeaderItem) or item.parent is None:
            continue
        parent = item.parent.resolve(hier)
        if isinstance(parent, SectionHeaderItem):
            assert parent.level < item.level, (
                f"Parent section (level {parent.level}) should be lower than "
                f"child (level {item.level})"
            )


def test_to_hierarchical_item_counts_match(sample_doc):
    """All item types should have the same count before and after making hierarchical."""
    hier = DocHierarchyTransform.to_hierarchical(sample_doc)

    def count_by_type(doc, cls):
        # Use exact type match for TextItem so that subclasses (TitleItem,
        # SectionHeaderItem, ListItem, …) are not counted twice.
        match = (lambda item: type(item) is cls) if cls is TextItem else (lambda item: isinstance(item, cls))
        return sum(1 for item, _ in doc.iterate_items(with_groups=True) if match(item))

    for cls in (SectionHeaderItem, TitleItem, TableItem, PictureItem, ListGroup, ListItem, TextItem):
        assert count_by_type(hier, cls) == count_by_type(sample_doc, cls), (
            f"Item count mismatch for {cls.__name__} after to_hierarchical"
        )


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_round_trip_flat_then_hierarchical_preserves_levels(sample_doc):
    """flat → hierarchical must produce the same section level sequence as the original."""
    flat = DocHierarchyTransform.to_flat(sample_doc)
    hier = DocHierarchyTransform.to_hierarchical(flat)

    original_levels = [
        item.level
        for item, _ in sample_doc.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]
    round_trip_levels = [
        item.level
        for item, _ in hier.iterate_items(with_groups=True)
        if isinstance(item, SectionHeaderItem)
    ]

    assert original_levels == round_trip_levels
