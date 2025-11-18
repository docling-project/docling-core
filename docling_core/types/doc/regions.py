#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

"""Utils to work with region-defined tables."""

from typing import List, Optional, Protocol, Sequence, Set, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import TableCell, TableData


def bbox_fraction_inside(
    inner: BoundingBox, outer: BoundingBox, *, eps: float = 1.0e-9
) -> float:
    """Return the fraction of ``inner`` area that lies inside ``outer``."""
    area = inner.area()
    if area <= eps:
        return 0.0
    intersection = inner.intersection_area_with(outer)
    return intersection / max(area, eps)


def bbox_contains(
    inner: BoundingBox, outer: BoundingBox, *, threshold: float, eps: float = 1.0e-9
) -> bool:
    """Return ``True`` when ``inner`` is contained in ``outer`` above ``threshold``."""
    return bbox_fraction_inside(inner, outer, eps=eps) >= threshold


def bbox_iou(a: BoundingBox, b: BoundingBox, *, eps: float = 1.0e-6) -> float:
    """Return the intersection over union between two bounding boxes."""
    return a.intersection_over_union(b, eps=eps)


class HasBoundingBox(Protocol):
    """Protocol for objects exposing a bounding box."""

    bbox: BoundingBox


def dedupe_bboxes(
    elements: Sequence[BoundingBox],
    *,
    iou_threshold: float = 0.9,
) -> list[BoundingBox]:
    """Return elements whose bounding boxes are unique within ``iou_threshold``."""
    deduped: list[BoundingBox] = []
    for element in elements:
        if all(bbox_iou(element, kept) < iou_threshold for kept in deduped):
            deduped.append(element)
    return deduped


def is_bbox_within(
    bbox_a: BoundingBox, bbox_b: BoundingBox, threshold: float = 0.5
) -> bool:
    """Return ``True`` when ``bbox_b`` lies within ``bbox_a`` above ``threshold``."""
    return bbox_contains(bbox_b, bbox_a, threshold=threshold)


def _process_table_headers(
    bbox: BoundingBox,
    row_headers: List[BoundingBox] = [],
    col_headers: List[BoundingBox] = [],
    row_sections: List[BoundingBox] = [],
) -> Tuple[bool, bool, bool]:
    c_column_header = False
    c_row_header = False
    c_row_section = False

    for col_header in col_headers:
        if is_bbox_within(col_header, bbox):
            c_column_header = True
    for row_header in row_headers:
        if is_bbox_within(row_header, bbox):
            c_row_header = True
    for row_section in row_sections:
        if is_bbox_within(row_section, bbox):
            c_row_section = True
    return c_column_header, c_row_header, c_row_section


def bbox_intersection(a: BoundingBox, b: BoundingBox) -> Optional[BoundingBox]:
    """Return the intersection of two bounding boxes or ``None`` when disjoint."""
    if a.coord_origin != b.coord_origin:
        raise ValueError("BoundingBoxes have different CoordOrigin")

    left = max(a.l, b.l)
    right = min(a.r, b.r)

    if a.coord_origin == CoordOrigin.TOPLEFT:
        top = max(a.t, b.t)
        bottom = min(a.b, b.b)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(
            l=left, t=top, r=right, b=bottom, coord_origin=a.coord_origin
        )

    top = min(a.t, b.t)
    bottom = max(a.b, b.b)
    if right <= left or top <= bottom:
        return None
    return BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=a.coord_origin)


def compute_cells(
    rows: List[BoundingBox],
    columns: List[BoundingBox],
    merges: List[BoundingBox],
    row_headers: List[BoundingBox] = [],
    col_headers: List[BoundingBox] = [],
    row_sections: List[BoundingBox] = [],
    row_overlap_threshold: float = 0.5,  # how much of a row a merge must cover vertically
    col_overlap_threshold: float = 0.5,  # how much of a column a merge must cover horizontally
) -> List[TableCell]:
    """Returns TableCell. Merged cells are aligned to grid boundaries.

    rows, columns, merges are lists of BoundingBox(l,t,r,b).
    """
    rows.sort(key=lambda r: (r.t + r.b) / 2.0)
    columns.sort(key=lambda c: (c.l + c.r) / 2.0)

    # n_rows, n_cols = len(rows), len(columns)

    def span_from_merge(
        m: BoundingBox, lines: List[BoundingBox], axis: str, frac_threshold: float
    ) -> Optional[Tuple[int, int]]:
        """Map a merge bbox to an inclusive index span over rows or columns.

        axis='row' uses vertical overlap vs row height; axis='col' uses horizontal overlap vs col width.
        If nothing meets threshold, pick the single best-overlapping line if overlap>0; else return None.
        """
        idxs = []
        best_i, best_len = None, 0.0
        for i, elem in enumerate(lines):
            inter = bbox_intersection(m, elem)
            if not inter:
                continue
            if axis == "row":
                overlap_len = inter.height
                base = max(1e-9, elem.height)
            else:
                overlap_len = inter.width
                base = max(1e-9, elem.width)

            frac = overlap_len / base
            if frac >= frac_threshold:
                idxs.append(i)

            if overlap_len > best_len:
                best_len, best_i = overlap_len, i

        if idxs:
            return min(idxs), max(idxs)
        if best_i is not None and best_len > 0.0:
            return best_i, best_i
        return None

    cells: List[TableCell] = []
    covered: Set[Tuple[int, int]] = set()
    seen_merge_rects: Set[Tuple[int, int, int, int]] = set()

    # 1) Add merged cells first (and mark their covered simple cells)
    for m in merges:
        rspan = span_from_merge(
            m, rows, axis="row", frac_threshold=row_overlap_threshold
        )
        cspan = span_from_merge(
            m, columns, axis="col", frac_threshold=col_overlap_threshold
        )
        if rspan is None or cspan is None:
            # Can't confidently map this merge to grid -> skip it
            continue

        sr, er = rspan
        sc, ec = cspan
        rect_key = (sr, er, sc, ec)
        if rect_key in seen_merge_rects:
            continue
        seen_merge_rects.add(rect_key)

        # Grid-aligned bbox for the merged cell
        grid_bbox = BoundingBox(
            l=columns[sc].l,
            t=rows[sr].t,
            r=columns[ec].r,
            b=rows[er].b,
        )
        c_column_header, c_row_header, c_row_section = _process_table_headers(
            grid_bbox, col_headers, row_headers, row_sections
        )

        cells.append(
            TableCell(
                text="",
                row_span=er - sr + 1,
                col_span=ec - sc + 1,
                start_row_offset_idx=sr,
                end_row_offset_idx=er + 1,
                start_col_offset_idx=sc,
                end_col_offset_idx=ec + 1,
                bbox=grid_bbox,
                column_header=c_column_header,
                row_header=c_row_header,
                row_section=c_row_section,
            )
        )
        for ri in range(sr, er + 1):
            for ci in range(sc, ec + 1):
                covered.add((ri, ci))

    # 2) Add simple (1x1) cells where not covered by merges
    for ri, row in enumerate(rows):
        for ci, col in enumerate(columns):
            if (ri, ci) in covered:
                continue
            inter = bbox_intersection(row, col)
            if not inter:
                # In degenerate cases (big gaps), there might be no intersection; skip.
                continue
            c_column_header, c_row_header, c_row_section = _process_table_headers(
                inter, col_headers, row_headers, row_sections
            )
            cells.append(
                TableCell(
                    text="",
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=ri,
                    end_row_offset_idx=ri + 1,
                    start_col_offset_idx=ci,
                    end_col_offset_idx=ci + 1,
                    bbox=inter,
                    column_header=c_column_header,
                    row_header=c_row_header,
                    row_section=c_row_section,
                )
            )
    return cells


def regions_to_table(
    table_bbox: BoundingBox,
    rows: List[BoundingBox],
    cols: List[BoundingBox],
    merges: List[BoundingBox],
    row_headers: List[BoundingBox] = [],
    col_headers: List[BoundingBox] = [],
    row_sections: List[BoundingBox] = [],
) -> Optional[TableData]:
    """Converts regions: rows, columns, merged cells into table_data structure.

    Adds semantics for regions of row_headers, col_headers, row_section
    """
    default_containment_thresh = 0.50
    rows.extend(row_sections)  # use row sections to compensate for missing rows
    rows = dedupe_bboxes(
        [
            e
            for e in rows
            if bbox_contains(e, table_bbox, threshold=default_containment_thresh)
        ]
    )
    cols = dedupe_bboxes(
        [
            e
            for e in cols
            if bbox_contains(e, table_bbox, threshold=default_containment_thresh)
        ]
    )
    merges = dedupe_bboxes(
        [
            e
            for e in merges
            if bbox_contains(e, table_bbox, threshold=default_containment_thresh)
        ]
    )

    col_headers = dedupe_bboxes(
        [
            e
            for e in col_headers
            if bbox_contains(e, table_bbox, threshold=default_containment_thresh)
        ]
    )
    row_headers = dedupe_bboxes(
        [
            e
            for e in row_headers
            if bbox_contains(e, table_bbox, threshold=default_containment_thresh)
        ]
    )
    row_sections = dedupe_bboxes(
        [
            e
            for e in row_sections
            if bbox_contains(e, table_bbox, threshold=default_containment_thresh)
        ]
    )

    # Compute table cells from CVAT elements: rows, cols, merges
    computed_table_cells = compute_cells(
        rows,
        cols,
        merges,
        col_headers,
        row_headers,
        row_sections,
    )

    # If no table structure found, create single fake cell for content
    if not rows or not cols:
        computed_table_cells = [
            TableCell(
                text="",
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                bbox=table_bbox,
                column_header=False,
                row_header=False,
                row_section=False,
            )
        ]
        table_data = TableData(num_rows=1, num_cols=1)
    else:
        table_data = TableData(num_rows=len(rows), num_cols=len(cols))

    table_data.table_cells = computed_table_cells

    return table_data
