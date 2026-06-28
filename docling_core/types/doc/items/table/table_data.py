"""Table cell and table data models."""

import logging
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Optional, Union

from pydantic import BaseModel, Field, computed_field, model_validator
from typing_extensions import Self, override

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.common.reference import RefItem

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_logger = logging.getLogger(__name__)


class TableCell(BaseModel):
    """TableCell."""

    bbox: Optional[BoundingBox] = None
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False
    fillable: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """from_dict_format."""
        if isinstance(data, dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                # "bbox" not in data
                # or data["bbox"] is None
                # or isinstance(data["bbox"], BoundingBox)
                "text" in data
            ):
                return data
            text = data.get("bbox", {}).get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    text = " ".join(el["token"] for el in text_cells)

            data["text"] = text

        return data

    def _get_text(self, doc: Optional["DoclingDocument"] = None, **kwargs: Any) -> str:
        return self.text


class RichTableCell(TableCell):
    """RichTableCell."""

    ref: "RefItem"

    @override
    def _get_text(self, doc: Optional["DoclingDocument"] = None, **kwargs: Any) -> str:
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        if doc is not None:
            doc_serializer = kwargs.pop("doc_serializer", MarkdownDocSerializer(doc=doc))
            ser_res = doc_serializer.serialize(item=self.ref.resolve(doc=doc), **kwargs)
            return ser_res.text
        else:
            return "<!-- rich cell -->"


AnyTableCell = Annotated[
    Union[RichTableCell, TableCell],
    Field(union_mode="left_to_right"),
]


class Orientation(str, Enum):
    """Counter-clockwise rotation of a table on the page, in degrees.

    Follows the convention used by PIL/Pillow's ``Image.rotate``: positive
    angles rotate the table counter-clockwise. ``ROT_0`` / ``ROT_180`` keep
    rows running horizontally on the page; ``ROT_90`` / ``ROT_270`` turn
    rows into vertical stripes.
    """

    ROT_0 = "rot_0"  # no rotation; row 0 at top, rows horizontal
    ROT_90 = "rot_90"  # 90° CCW; row 0 on the left, rows are vertical stripes
    ROT_180 = "rot_180"  # 180°; row 0 at bottom (upside-down), rows horizontal
    ROT_270 = "rot_270"  # 270° CCW (= 90° CW); row 0 on the right, rows are vertical stripes


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: list[AnyTableCell] = []
    num_rows: int = 0
    num_cols: int = 0
    orientation: Orientation = Orientation.ROT_0

    @computed_field  # type: ignore
    @property
    def grid(
        self,
    ) -> list[list[TableCell]]:
        """Grid."""
        # Initialise empty table data grid (only empty cells)
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # Overwrite cells in table data for which there is actual cell content.
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data

    def remove_rows(self, indices: list[int], doc: Optional["DoclingDocument"] = None) -> list[list[TableCell]]:
        """Remove rows from the table by their indices.

        :param indices: list[int]: A list of indices of the rows to remove. (Starting from 0)

        :return: list[list[TableCell]]: A list representation of the removed rows as lists of TableCell objects.
        """
        if not indices:
            return []

        indices = sorted(indices, reverse=True)

        refs_to_remove = []
        all_removed_cells = []
        for row_index in indices:
            if row_index < 0 or row_index >= self.num_rows:
                raise IndexError(
                    f"Row index {row_index} is out of bounds for the current number of rows {self.num_rows}."
                )

            start_idx = row_index * self.num_cols
            end_idx = start_idx + self.num_cols
            removed_cells = self.table_cells[start_idx:end_idx]

            for cell in removed_cells:
                if isinstance(cell, RichTableCell):
                    refs_to_remove.append(cell.ref)

            # Remove the cells from the table
            self.table_cells = self.table_cells[:start_idx] + self.table_cells[end_idx:]

            # Update the number of rows
            self.num_rows -= 1

            # Reassign row offset indices for existing cells
            for index, cell in enumerate(self.table_cells):
                new_index = index // self.num_cols
                cell.start_row_offset_idx = new_index
                cell.end_row_offset_idx = new_index + 1

            all_removed_cells.append(removed_cells)

        if refs_to_remove:
            if doc is None:
                _logger.warning(
                    "When table contains rich cells, `doc` argument must be provided, "
                    "otherwise rich cell content will be left dangling."
                )
            else:
                doc._delete_items(refs_to_remove)

        return all_removed_cells

    def pop_row(self, doc: Optional["DoclingDocument"] = None) -> list[TableCell]:
        """Remove and return the last row from the table.

        :returns: list[TableCell]: A list of TableCell objects representing the popped row.
        """
        if self.num_rows == 0:
            raise IndexError("Cannot pop from an empty table.")

        return self.remove_row(self.num_rows - 1, doc=doc)

    def remove_row(self, row_index: int, doc: Optional["DoclingDocument"] = None) -> list[TableCell]:
        """Remove a row from the table by its index.

        :param row_index: int: The index of the row to remove. (Starting from 0)

        :returns: list[TableCell]: A list of TableCell objects representing the removed row.
        """
        return self.remove_rows([row_index], doc=doc)[0]

    def insert_rows(self, row_index: int, rows: list[list[str]], after: bool = False) -> None:
        """Insert multiple new rows from a list of lists of strings before/after a specific index in the table.

        :param row_index: int: The index at which to insert the new rows. (Starting from 0)
        :param rows: list[list[str]]: A list of lists, where each inner list represents the content of a new row.
        :param after: bool: If True, insert the rows after the specified index, otherwise before it. (Default is False)

        :returns: None
        """
        effective_rows = rows[::-1]

        for row in effective_rows:
            self.insert_row(row_index, row, after)

    def insert_row(self, row_index: int, row: list[str], after: bool = False) -> None:
        """Insert a new row from a list of strings before/after a specific index in the table.

        :param row_index: int: The index at which to insert the new row. (Starting from 0)
        :param row: list[str]: A list of strings representing the content of the new row.
        :param after: bool: If True, insert the row after the specified index, otherwise before it. (Default is False)

        :returns: None
        """
        if len(row) != self.num_cols:
            raise ValueError(f"Row length {len(row)} does not match the number of columns {self.num_cols}.")

        effective_index = row_index + (1 if after else 0)

        if effective_index < 0 or effective_index > self.num_rows:
            raise IndexError(f"Row index {row_index} is out of bounds for the current number of rows {self.num_rows}.")

        new_row_cells = [
            TableCell(
                text=text,
                start_row_offset_idx=effective_index,
                end_row_offset_idx=effective_index + 1,
                start_col_offset_idx=j,
                end_col_offset_idx=j + 1,
            )
            for j, text in enumerate(row)
        ]

        self.table_cells = (
            self.table_cells[: effective_index * self.num_cols]
            + new_row_cells
            + self.table_cells[effective_index * self.num_cols :]
        )

        # Reassign row offset indices for existing cells
        for index, cell in enumerate(self.table_cells):
            new_index = index // self.num_cols
            cell.start_row_offset_idx = new_index
            cell.end_row_offset_idx = new_index + 1

        self.num_rows += 1

    def add_rows(self, rows: list[list[str]]) -> None:
        """Add multiple new rows to the table from a list of lists of strings.

        :param rows: list[list[str]]: A list of lists, where each inner list represents the content of a new row.

        :returns: None
        """
        for row in rows:
            self.add_row(row)

    def add_row(self, row: list[str]) -> None:
        """Add a new row to the table from a list of strings.

        :param row: list[str]: A list of strings representing the content of the new row.

        :returns: None
        """
        self.insert_row(row_index=self.num_rows - 1, row=row, after=True)

    def get_row_bounding_boxes(self, *, minimal: bool = True) -> dict[int, BoundingBox]:
        """Get the bounding box for each row in the table.

        Layout follows the table's ``orientation`` field: ``ROT_0`` / ``ROT_180``
        keep rows running left-to-right on the page; ``ROT_90`` / ``ROT_270``
        turn rows into vertical stripes. This affects both the axis along which
        span cells extend a row's bbox and, when ``minimal=False``, the axis
        equalized across rows.

        Args:
            minimal: If True (default), returns the minimal bounding box for each
                row based on its cells. If False, all rows will have a uniform
                extent perpendicular to the row direction (l/r for ROT_0/ROT_180,
                t/b for ROT_90/ROT_270).

        Returns:
            dict[int, BoundingBox]: A dictionary mapping row indices to their
            bounding boxes. Only rows with cells that have bounding boxes are included.
        """
        horizontal = self.orientation in (Orientation.ROT_0, Orientation.ROT_180)
        coords = []
        for cell in self.table_cells:
            if cell.bbox is not None:
                coords.append(cell.bbox.coord_origin)

        if len(set(coords)) > 1:
            raise ValueError(
                "All bounding boxes must have the same \
                CoordOrigin to compute their union."
            )

        row_bboxes: dict[int, BoundingBox] = {}

        for row_idx in range(self.num_rows):
            row_cells_with_bbox: dict[int, list[BoundingBox]] = {}

            # Collect all cells in this row that have bounding boxes
            for cell in self.table_cells:
                if cell.bbox is not None and cell.start_row_offset_idx <= row_idx < cell.end_row_offset_idx:
                    row_span = cell.end_row_offset_idx - cell.start_row_offset_idx
                    if row_span in row_cells_with_bbox:
                        row_cells_with_bbox[row_span].append(cell.bbox)
                    else:
                        row_cells_with_bbox[row_span] = [cell.bbox]

            # Calculate the enclosing bounding box for this row
            if len(row_cells_with_bbox) > 0:
                min_row_span = min(row_cells_with_bbox.keys())
                row_bbox: BoundingBox = BoundingBox.enclosing_bbox(row_cells_with_bbox[min_row_span])

                # Spanning cells extend along the row's natural axis:
                # horizontal table → row runs l/r; vertical table → row runs t/b.
                for rspan, bboxs in row_cells_with_bbox.items():
                    for bbox in bboxs:
                        if horizontal:
                            row_bbox.l = min(row_bbox.l, bbox.l)
                            row_bbox.r = max(row_bbox.r, bbox.r)
                        else:
                            if bbox.coord_origin == CoordOrigin.TOPLEFT:
                                row_bbox.t = min(row_bbox.t, bbox.t)
                                row_bbox.b = max(row_bbox.b, bbox.b)
                            else:  # BOTTOMLEFT
                                row_bbox.t = max(row_bbox.t, bbox.t)
                                row_bbox.b = min(row_bbox.b, bbox.b)

                row_bboxes[row_idx] = row_bbox

        # If not minimal, make all rows have uniform extent on the axis
        # perpendicular to the row direction.
        if not minimal and row_bboxes:
            if horizontal:
                # Rows run left-to-right; equalize horizontal extent.
                global_l = min(bbox.l for bbox in row_bboxes.values())
                global_r = max(bbox.r for bbox in row_bboxes.values())
                for bbox in row_bboxes.values():
                    bbox.l = global_l
                    bbox.r = global_r
            else:
                # Vertical table: rows are vertical stripes; equalize vertical extent.
                first_bbox = next(iter(row_bboxes.values()))
                if first_bbox.coord_origin == CoordOrigin.TOPLEFT:
                    global_t = min(bbox.t for bbox in row_bboxes.values())
                    global_b = max(bbox.b for bbox in row_bboxes.values())
                else:  # BOTTOMLEFT
                    global_t = max(bbox.t for bbox in row_bboxes.values())
                    global_b = min(bbox.b for bbox in row_bboxes.values())
                for bbox in row_bboxes.values():
                    bbox.t = global_t
                    bbox.b = global_b

        return row_bboxes

    def get_column_bounding_boxes(self, *, minimal: bool = True) -> dict[int, BoundingBox]:
        """Get the bounding box for each column in the table.

        Layout follows the table's ``orientation`` field: ``ROT_0`` / ``ROT_180``
        keep columns running top-to-bottom on the page; ``ROT_90`` / ``ROT_270``
        turn columns into horizontal stripes. This affects both the axis along
        which span cells extend a column's bbox and, when ``minimal=False``, the
        axis equalized across columns.

        Args:
            minimal: If True (default), returns the minimal bounding box for each
                column based on its cells. If False, all columns will have a
                uniform extent perpendicular to the column direction (t/b for
                ROT_0/ROT_180, l/r for ROT_90/ROT_270).

        Returns:
            dict[int, BoundingBox]: A dictionary mapping column indices to their
            bounding boxes. Only columns with cells that have bounding boxes are included.
        """
        horizontal = self.orientation in (Orientation.ROT_0, Orientation.ROT_180)
        coords = []
        for cell in self.table_cells:
            if cell.bbox is not None:
                coords.append(cell.bbox.coord_origin)

        if len(set(coords)) > 1:
            raise ValueError(
                "All bounding boxes must have the same \
                CoordOrigin to compute their union."
            )

        col_bboxes: dict[int, BoundingBox] = {}

        for col_idx in range(self.num_cols):
            col_cells_with_bbox: dict[int, list[BoundingBox]] = {}

            # Collect all cells in this row that have bounding boxes
            for cell in self.table_cells:
                if cell.bbox is not None and cell.start_col_offset_idx <= col_idx < cell.end_col_offset_idx:
                    col_span = cell.end_col_offset_idx - cell.start_col_offset_idx
                    if col_span in col_cells_with_bbox:
                        col_cells_with_bbox[col_span].append(cell.bbox)
                    else:
                        col_cells_with_bbox[col_span] = [cell.bbox]

            # Calculate the enclosing bounding box for this row
            if len(col_cells_with_bbox) > 0:
                min_col_span = min(col_cells_with_bbox.keys())
                col_bbox: BoundingBox = BoundingBox.enclosing_bbox(col_cells_with_bbox[min_col_span])

                # Spanning cells extend along the column's natural axis:
                # horizontal table → column runs t/b; vertical table → column runs l/r.
                for rspan, bboxs in col_cells_with_bbox.items():
                    for bbox in bboxs:
                        if horizontal:
                            if bbox.coord_origin == CoordOrigin.TOPLEFT:
                                col_bbox.b = max(col_bbox.b, bbox.b)
                                col_bbox.t = min(col_bbox.t, bbox.t)
                            elif bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                                col_bbox.b = min(col_bbox.b, bbox.b)
                                col_bbox.t = max(col_bbox.t, bbox.t)
                        else:
                            col_bbox.l = min(col_bbox.l, bbox.l)
                            col_bbox.r = max(col_bbox.r, bbox.r)

                col_bboxes[col_idx] = col_bbox

        # If not minimal, make all columns have uniform extent on the axis
        # perpendicular to the column direction.
        if not minimal and col_bboxes:
            if horizontal:
                # Columns run top-to-bottom; equalize vertical extent.
                # Get the coord_origin from the first bbox (they're all the same)
                first_bbox = next(iter(col_bboxes.values()))
                if first_bbox.coord_origin == CoordOrigin.TOPLEFT:
                    global_t = min(bbox.t for bbox in col_bboxes.values())
                    global_b = max(bbox.b for bbox in col_bboxes.values())
                else:  # BOTTOMLEFT
                    global_t = max(bbox.t for bbox in col_bboxes.values())
                    global_b = min(bbox.b for bbox in col_bboxes.values())
                for bbox in col_bboxes.values():
                    bbox.t = global_t
                    bbox.b = global_b
            else:
                # Vertical table: columns are horizontal stripes; equalize horizontal extent.
                global_l = min(bbox.l for bbox in col_bboxes.values())
                global_r = max(bbox.r for bbox in col_bboxes.values())
                for bbox in col_bboxes.values():
                    bbox.l = global_l
                    bbox.r = global_r

        return col_bboxes

    @classmethod
    def _dedupe_bboxes(
        cls,
        elements: Sequence[BoundingBox],
        *,
        iou_threshold: float = 0.9,
    ) -> list[BoundingBox]:
        """Return elements whose bounding boxes are unique within ``iou_threshold``."""
        deduped: list[BoundingBox] = []
        for element in elements:
            if all(element.intersection_over_union(kept) < iou_threshold for kept in deduped):
                deduped.append(element)
        return deduped

    @classmethod
    def _process_table_headers(
        cls,
        bbox: BoundingBox,
        row_headers: list[BoundingBox] = [],
        col_headers: list[BoundingBox] = [],
        row_sections: list[BoundingBox] = [],
    ) -> tuple[bool, bool, bool]:
        c_column_header = False
        c_row_header = False
        c_row_section = False

        for col_header in col_headers:
            if bbox.intersection_over_self(col_header) >= 0.5:
                c_column_header = True
        for row_header in row_headers:
            if bbox.intersection_over_self(row_header) >= 0.5:
                c_row_header = True
        for row_section in row_sections:
            if bbox.intersection_over_self(row_section) >= 0.5:
                c_row_section = True
        return c_column_header, c_row_header, c_row_section

    @classmethod
    def _compute_cells(
        cls,
        rows: list[BoundingBox],
        columns: list[BoundingBox],
        merges: list[BoundingBox],
        row_headers: list[BoundingBox] = [],
        col_headers: list[BoundingBox] = [],
        row_sections: list[BoundingBox] = [],
        row_overlap_threshold: float = 0.5,  # how much of a row a merge must cover vertically
        col_overlap_threshold: float = 0.5,  # how much of a column a merge must cover horizontally
    ) -> list[TableCell]:
        """Returns TableCell. Merged cells are aligned to grid boundaries.

        rows, columns, merges are lists of BoundingBox(l,t,r,b).
        """
        rows.sort(key=lambda r: (r.t + r.b) / 2.0)
        columns.sort(key=lambda c: (c.l + c.r) / 2.0)

        def span_from_merge(
            m: BoundingBox, lines: list[BoundingBox], axis: str, frac_threshold: float
        ) -> Optional[tuple[int, int]]:
            """Map a merge bbox to an inclusive index span over rows or columns.

            axis='row' uses vertical overlap vs row height; axis='col' uses horizontal overlap vs col width.
            If nothing meets threshold, pick the single best-overlapping line if overlap>0; else return None.
            """
            idxs = []
            best_i, best_len = None, 0.0
            for i, elem in enumerate(lines):
                inter = m.get_intersection_bbox(elem)
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

        cells: list[TableCell] = []
        covered: set[tuple[int, int]] = set()
        seen_merge_rects: set[tuple[int, int, int, int]] = set()

        # 1) Add merged cells first (and mark their covered simple cells)
        for m in merges:
            rspan = span_from_merge(m, rows, axis="row", frac_threshold=row_overlap_threshold)
            cspan = span_from_merge(m, columns, axis="col", frac_threshold=col_overlap_threshold)
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
            c_column_header, c_row_header, c_row_section = cls._process_table_headers(
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
                inter = row.get_intersection_bbox(col)
                if not inter:
                    # In degenerate cases (big gaps), there might be no intersection; skip.
                    continue
                c_column_header, c_row_header, c_row_section = cls._process_table_headers(
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

    @classmethod
    def from_regions(
        cls,
        table_bbox: BoundingBox,
        rows: list[BoundingBox],
        cols: list[BoundingBox],
        merges: list[BoundingBox],
        row_headers: list[BoundingBox] = [],
        col_headers: list[BoundingBox] = [],
        row_sections: list[BoundingBox] = [],
    ) -> Self:
        """Converts regions: rows, columns, merged cells into table_data structure.

        Adds semantics for regions of row_headers, col_headers, row_section
        """
        default_containment_thresh = 0.5
        rows.extend(row_sections)  # use row sections to compensate for missing rows
        rows = cls._dedupe_bboxes(
            [e for e in rows if e.intersection_over_self(table_bbox) >= default_containment_thresh]
        )
        cols = cls._dedupe_bboxes(
            [e for e in cols if e.intersection_over_self(table_bbox) >= default_containment_thresh]
        )
        merges = cls._dedupe_bboxes(
            [e for e in merges if e.intersection_over_self(table_bbox) >= default_containment_thresh]
        )

        col_headers = cls._dedupe_bboxes(
            [e for e in col_headers if e.intersection_over_self(table_bbox) >= default_containment_thresh]
        )
        row_headers = cls._dedupe_bboxes(
            [e for e in row_headers if e.intersection_over_self(table_bbox) >= default_containment_thresh]
        )
        row_sections = cls._dedupe_bboxes(
            [e for e in row_sections if e.intersection_over_self(table_bbox) >= default_containment_thresh]
        )

        # Compute table cells from CVAT elements: rows, cols, merges
        computed_table_cells = cls._compute_cells(
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
            table_data = cls(num_rows=1, num_cols=1)
        else:
            table_data = cls(num_rows=len(rows), num_cols=len(cols))

        table_data.table_cells = computed_table_cells

        return table_data
