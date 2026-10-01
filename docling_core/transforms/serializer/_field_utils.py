"""Shared helpers for serializing legacy key-value and form graph items."""

from docling_core.types.doc import GraphCell, GraphData, TextItem
from docling_core.types.doc.document import FieldValueItem
from docling_core.types.doc.labels import DocItemLabel, GraphLinkLabel


def extract_key_value_pairs(graph: GraphData) -> list[tuple[GraphCell, list[GraphCell]]]:
    """Extract key/value cell pairs from a graph, preserving link order.

    Only ``TO_VALUE`` and ``TO_KEY`` links define pairs, and every key/value cell
    combination is emitted once, mirroring the extraction performed by
    ``DoclingDocument._migrate_to_field_regions``.

    Args:
        graph: The graph holding the key and value cells.

    Returns:
        The key cells with their linked value cells, in link order.
    """
    pos_by_cell_id = {cell.cell_id: idx for idx, cell in enumerate(graph.cells)}
    visited: set[str] = set()
    pairs: list[tuple[GraphCell, list[GraphCell]]] = []
    pair_index_by_key_cell_id: dict[int, int] = {}
    for link in graph.links:
        if link.label == GraphLinkLabel.TO_VALUE:
            key_cell = graph.cells[pos_by_cell_id[link.source_cell_id]]
            value_cell = graph.cells[pos_by_cell_id[link.target_cell_id]]
        elif link.label == GraphLinkLabel.TO_KEY:
            value_cell = graph.cells[pos_by_cell_id[link.source_cell_id]]
            key_cell = graph.cells[pos_by_cell_id[link.target_cell_id]]
        else:
            continue
        pair_id = f"{key_cell.cell_id}-{value_cell.cell_id}"
        if pair_id in visited:
            continue
        visited.add(pair_id)
        if key_cell.cell_id not in pair_index_by_key_cell_id:
            pair_index_by_key_cell_id[key_cell.cell_id] = len(pairs)
            pairs.append((key_cell, []))
        pairs[pair_index_by_key_cell_id[key_cell.cell_id]][1].append(value_cell)
    return pairs


def create_key_item(*, cell: GraphCell, idx: int) -> TextItem:
    """Create an in-memory field-key text item for a legacy key cell.

    Args:
        cell: The key cell to convert.
        idx: Index used to build a unique, valid ``self_ref``.

    Returns:
        The field-key text item.
    """
    return TextItem(
        self_ref=f"#/fake/{idx}",
        label=DocItemLabel.FIELD_KEY,
        text=cell.text,
        orig=cell.orig or cell.text,
        prov=[cell.prov] if cell.prov is not None else [],
    )


def create_value_item(*, cell: GraphCell, idx: int) -> FieldValueItem:
    """Create an in-memory field-value text item for a legacy value cell.

    Args:
        cell: The value cell to convert.
        idx: Index used to build a unique, valid ``self_ref``.

    Returns:
        The field-value text item.
    """
    return FieldValueItem(
        self_ref=f"#/fake/{idx}",
        text=cell.text,
        orig=cell.orig or cell.text,
        prov=[cell.prov] if cell.prov is not None else [],
    )
