from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.types.doc import DocItemLabel, DoclingDocument, RichTableCell, TableData
from test.test_serialization import verify
from test.test_serialization_doclang import _verify_doc


@pytest.fixture
def simple_doc() -> DoclingDocument:
    """Create a simple document for testing."""
    return DoclingDocument(name="test_doc")


def _verify_parent_child_relationship(child: Any, parent: Any, parent_name: str) -> None:
    """Helper to verify parent-child relationships."""
    assert child.parent.cref == parent.self_ref, f"{parent_name} parent should not change"
    assert child.get_ref() in parent.children, f"{parent_name} should be in parent's children"


def _validate_doc(doc: DoclingDocument) -> None:
    """Validate document by re-parsing it through Pydantic."""
    DoclingDocument.model_validate(doc.model_dump())


def test_flatten(mixed_hierarchy_doc: DoclingDocument):
    doc: DoclingDocument = mixed_hierarchy_doc
    doc._flatten()

    doc._normalize_references()

    exp_json = Path("./test/data/doc/flattened.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    exp_dclg = Path("./test/data/doc/flattened.dclg.xml")
    actual = DocLangDocSerializer(doc=doc, params=DocLangParams(include_version=False)).serialize().text
    verify(actual=actual, exp_file=exp_dclg)


def test_hierarchize(mixed_hierarchy_doc):
    doc: DoclingDocument = mixed_hierarchy_doc
    doc._hierarchize()

    doc._normalize_references()

    exp_json = Path("./test/data/doc/hierarchized.json")
    _verify_doc(doc=doc, exp_json=exp_json)

    exp_dclg = Path("./test/data/doc/hierarchized.dclg.xml")
    actual = DocLangDocSerializer(doc=doc, params=DocLangParams(include_version=False)).serialize().text
    verify(actual=actual, exp_file=exp_dclg)


def test_hierarchize_preserves_table_structure(simple_doc: DoclingDocument) -> None:
    """Test that _hierarchize does not move table cell children."""
    doc = simple_doc
    table = doc.add_table(data=TableData(num_rows=1, num_cols=2), parent=doc.body)

    # Add groups as table children (simulating rich table cells)
    groups = [doc.add_group(parent=table) for _ in range(2)]
    for i, group in enumerate(groups, 1):
        doc.add_text(label=DocItemLabel.TEXT, text=f"Cell {i}", parent=group)

    # Update table data with rich cells
    table.data.table_cells = [
        RichTableCell(
            text="",
            start_row_offset_idx=0,
            end_row_offset_idx=0,
            start_col_offset_idx=i,
            end_col_offset_idx=i,
            ref=group.get_ref(),
        )
        for i, group in enumerate(groups)
    ]

    doc._hierarchize()
    _validate_doc(doc)

    # Verify that table cell children were not moved
    for i, group in enumerate(groups, 1):
        _verify_parent_child_relationship(group, table, f"Group{i}")


def test_hierarchize_preserves_picture_structure(simple_doc: DoclingDocument) -> None:
    """Test that _hierarchize does not move picture children."""
    doc = simple_doc
    picture = doc.add_picture(parent=doc.body)
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Picture caption", parent=picture)

    doc._hierarchize()
    _validate_doc(doc)

    _verify_parent_child_relationship(caption, picture, "Caption")


def test_hierarchize_preserves_floating_item_descendants(simple_doc: DoclingDocument) -> None:
    """Test that _hierarchize preserves the entire internal structure of FloatingItems, not just direct children."""
    doc = simple_doc
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2), parent=doc.body)
    group = doc.add_group(parent=table)
    header = doc.add_heading(text="Header", level=2, parent=group)
    texts = [doc.add_text(label=DocItemLabel.TEXT, text=f"Text {i}", parent=header) for i in range(1, 4)]

    table.data.table_cells = [
        RichTableCell(
            text="",
            start_row_offset_idx=0,
            end_row_offset_idx=0,
            start_col_offset_idx=0,
            end_col_offset_idx=0,
            ref=group.get_ref(),
        ),
    ]

    _validate_doc(doc)
    header_children_before = [c.cref for c in header.children]

    doc._hierarchize()
    _validate_doc(doc)

    # Verify that the entire structure is preserved
    _verify_parent_child_relationship(group, table, "Group")
    _verify_parent_child_relationship(header, group, "Header")
    for i, text in enumerate(texts, 1):
        _verify_parent_child_relationship(text, header, f"Text{i}")

    # Verify that the header's children list is unchanged
    assert [c.cref for c in header.children] == header_children_before, "Header children should not change"


@pytest.mark.parametrize(
    "json_file",
    list(Path("test/data/doc").glob("*.json")),
)
def test_hierarchize_all_docs(json_file: Path) -> None:
    """Test _hierarchize with real documents to ensure it preserves structural integrity."""
    try:
        doc = DoclingDocument.model_validate_json(json_file.read_text(encoding="utf-8"))
    except ValidationError:
        pytest.skip(f"Invalid doc {json_file.name}")

    # Call _hierarchize
    doc._hierarchize()

    # Validate after hierarchize - this should not raise
    _validate_doc(doc)

    # Verify that all table cells still have correct parents
    for table in doc.tables:
        for cell in table.data.table_cells:
            if hasattr(cell, "ref"):
                cell_item = cell.ref.resolve(doc=doc)
                assert cell_item.parent.resolve(doc=doc) == table, (
                    f"In {json_file.name}: Cell {cell.ref.cref} should have parent {table.self_ref}, not {cell_item.parent.cref}"
                )
                assert cell.ref in table.children, (
                    f"In {json_file.name}: Cell {cell.ref.cref} should be in table's children"
                )
