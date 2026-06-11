"""JSON table serialization for embedding-friendly output."""

import json
import logging
import re
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseTableSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    create_ser_result,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownFallbackSerializer,
    MarkdownFormSerializer,
    MarkdownInlineSerializer,
    MarkdownKeyValueSerializer,
    MarkdownListSerializer,
    MarkdownMetaSerializer,
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTextSerializer,
)
from docling_core.types.doc.document import DoclingDocument, TableItem

_logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def _clean_text(text: str) -> str:
    """Clean and normalize text.

    Removes extra whitespace, normalizes newlines, and strips leading/trailing spaces.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def _is_valid_table(table_df: pd.DataFrame, min_rows: int = 2, min_cols: int = 2, max_empty_ratio: float = 0.5) -> bool:
    """Validate table quality.

    Checks:
    1. Minimum row count
    2. Minimum column count
    3. Maximum empty cell ratio

    Args:
        table_df: DataFrame to validate
        min_rows: Minimum required rows
        min_cols: Minimum required columns
        max_empty_ratio: Maximum allowed empty cell ratio (0.0 to 1.0)

    Returns:
        True if table is valid, False otherwise
    """
    if table_df is None or table_df.empty:
        return False

    # Check minimum dimensions
    num_rows, num_cols = table_df.shape
    if num_rows < min_rows or num_cols < min_cols:
        return False

    # Check empty cell ratio
    total_cells = num_rows * num_cols
    if total_cells == 0:
        return False

    # Count empty cells. Sum column+row
    empty_cells = table_df.isna().sum().sum() + (table_df == "").sum().sum()
    empty_ratio = empty_cells / total_cells

    if empty_ratio > max_empty_ratio:
        return False

    return True


def _detect_table_metadata(table_df: pd.DataFrame) -> dict[str, Any]:
    """Detect title and description from table structure using a simple heuristic.

    Detection rules:
    - Title: First row with exactly one non-empty cell
    - Description: Last row with exactly one non-empty cell
    - Detected rows are excluded from further header/data processing

    This is a heuristic intended to match the current feature ticket. Since
    ``export_to_dataframe()`` does not preserve merged-cell semantics directly,
    detection is based only on the number of non-empty cells in the first/last row.

    Args:
        table_df: DataFrame to analyze

    Returns:
        Dict with keys: title, description, data_start_row, data_end_row
    """
    metadata: dict[str, Any] = {"title": None, "description": None, "data_start_row": 0, "data_end_row": len(table_df)}

    if table_df.empty:
        return metadata

    def _extract_single_non_empty_value(row: pd.Series) -> Optional[str]:
        values = [_clean_text(str(v)) for v in row if pd.notna(v) and _clean_text(str(v))]
        if len(values) == 1:
            return values[0]
        return None

    first_value = _extract_single_non_empty_value(table_df.iloc[0])
    if first_value is not None:
        metadata["title"] = first_value
        metadata["data_start_row"] = 1

    if len(table_df) > metadata["data_start_row"]:
        last_value = _extract_single_non_empty_value(table_df.iloc[-1])
        if last_value is not None:
            metadata["description"] = last_value
            metadata["data_end_row"] = len(table_df) - 1

    return metadata


def _detect_header_type(table_df: pd.DataFrame) -> str:
    """Detect header configuration.

    Rules:
    - "both": First column name is empty, None, or single character (indicates row headers)
    - "top_only": Otherwise

    Args:
        table_df: DataFrame to analyze

    Returns:
        "top_only" or "both"
    """
    if table_df.empty or len(table_df.columns) == 0:
        return "top_only"

    # Check first column name (top-left header)
    first_col = table_df.columns[0]

    # Convert to string and clean
    first_col_str = _clean_text(str(first_col)) if first_col is not None and pd.notna(first_col) else ""

    # If empty, None, or single character, likely has both headers
    if not first_col_str or len(first_col_str) <= 1 or first_col_str.lower() == "none":
        return "both"

    return "top_only"


# ============================================================================
# Parameters
# ============================================================================


class JsonTableParams(CommonParams):
    """Configuration for JSON table serialization.

    Attributes:
        output_format: Format type - "structured_json" or "simple_json"
        include_table_index: Whether to include table index in output
        include_metadata: Whether to detect and include title/description
        min_rows: Minimum rows for valid table (default: 2)
        min_cols: Minimum columns for valid table (default: 2)
        max_empty_ratio: Maximum ratio of empty cells (default: 0.5)
        convert_to_natural_language: Convert JSON to text for embeddings
        indent: JSON indentation spaces (default: 2, None for compact)
        validate_tables: Whether to validate tables before serialization
    """

    output_format: Literal["structured_json", "simple_json"] = "structured_json"
    include_table_index: bool = True
    include_metadata: bool = True
    min_rows: int = 2
    min_cols: int = 2
    max_empty_ratio: float = 0.5
    convert_to_natural_language: bool = False
    indent: Optional[int] = 2
    validate_tables: bool = True


# ============================================================================
# JSON Table Serializer
# ============================================================================


class JsonTableSerializer(BaseModel, BaseTableSerializer):
    """Serializes tables to JSON format.

    Supports two output formats:
    1. structured_json: Intelligent header detection and structure
    2. simple_json: Basic row-by-row conversion

    Example:
        >>> serializer = JsonTableSerializer()
        >>> result = serializer.serialize(
        ...     item=table_item,
        ...     doc_serializer=doc_ser,
        ...     doc=document
        ... )
        >>> print(result.text)  # JSON string
    """

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize table to JSON format.

        Args:
            item: Table item to serialize
            doc_serializer: Parent document serializer
            doc: Document containing the table
            **kwargs: Additional parameters (merged with JsonTableParams)

        Returns:
            SerializationResult with JSON text and spans
        """
        params = JsonTableParams(**kwargs)

        # Handle excluded items
        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result(text="", span_source=item)

        try:
            # Export table to DataFrame
            table_df = item.export_to_dataframe(doc=doc)

            # Validate table if required
            if params.validate_tables:
                if not _is_valid_table(
                    table_df, min_rows=params.min_rows, min_cols=params.min_cols, max_empty_ratio=params.max_empty_ratio
                ):
                    _logger.info(f"Table {item.self_ref} failed validation, skipping")
                    return create_ser_result(text='{"error": "Table validation failed"}', span_source=item)

            # Get table index (count of tables before this one)
            table_index = self._get_table_index(item, doc) if params.include_table_index else 0

            # Convert to JSON
            if params.output_format == "structured_json":
                table_json = self._table_to_structured_json(
                    table_df, table_index=table_index, include_metadata=params.include_metadata
                )
            else:  # simple_json
                table_json = self._table_to_simple_json(table_df, table_index=table_index)

            # Convert to natural language if requested
            if params.convert_to_natural_language:
                json_text = self._convert_to_natural_language(table_json)
            else:
                json_text = json.dumps(table_json, indent=params.indent, ensure_ascii=False)

            return create_ser_result(text=json_text, span_source=item)

        except Exception as e:
            _logger.error(f"Error serializing table {item.self_ref}: {e}")
            return create_ser_result(text=f'{{"error": "Serialization failed: {e!s}"}}', span_source=item)

    def _get_table_index(self, item: TableItem, doc: DoclingDocument) -> int:
        """Get the index of this table in the document.

        Args:
            item: Table item
            doc: Document

        Returns:
            Zero-based index of the table
        """
        table_count = 0
        for doc_item, _ in doc.iterate_items():
            if isinstance(doc_item, TableItem):
                if doc_item.self_ref == item.self_ref:
                    return table_count
                table_count += 1
        return 0

    def _table_to_structured_json(
        self, table_df: pd.DataFrame, table_index: int = 0, include_metadata: bool = True
    ) -> dict:
        """Convert table to structured JSON format with header detection.

        Args:
            table_df: DataFrame to convert
            table_index: Index of the table in document
            include_metadata: Whether to detect and include metadata

        Returns:
            Dictionary in smart JSON format
        """
        # Detect metadata
        metadata = (
            _detect_table_metadata(table_df)
            if include_metadata
            else {"title": None, "description": None, "data_start_row": 0, "data_end_row": len(table_df)}
        )

        # Extract data rows (excluding title/description)
        data_df = table_df.iloc[metadata["data_start_row"] : metadata["data_end_row"]]

        if data_df.empty:
            return {"table_index": table_index, "error": "No data rows found"}

        # Detect header type
        header_type = _detect_header_type(data_df)

        # Format based on header type
        if header_type == "both":
            result = self._format_both_headers(data_df, table_index)
        else:
            result = self._format_top_only_headers(data_df, table_index)

        # Add metadata
        if metadata["title"]:
            result["title"] = metadata["title"]
        if metadata["description"]:
            result["description"] = metadata["description"]

        return result

    def _format_top_only_headers(self, table_df: pd.DataFrame, table_index: int) -> dict:
        """Format table with top-only headers.

        Args:
            table_df: DataFrame to format
            table_index: Table index

        Returns:
            Dictionary in top-only header format
        """
        result = {"table_index": table_index, "header_type": "top_only"}

        # Extract headers from columns
        headers = [_clean_text(str(col)) for col in table_df.columns]
        result["first_row_headers"] = headers

        # Extract data rows
        data_rows = []
        for _, row in table_df.iterrows():
            row_dict = {_clean_text(str(col)): _clean_text(str(val)) for col, val in row.items()}
            data_rows.append(row_dict)

        result["data_rows_count"] = len(data_rows)
        result["data"] = data_rows

        return result

    def _format_both_headers(self, table_df: pd.DataFrame, table_index: int) -> dict:
        """Format table with top and left headers.

        Args:
            table_df: DataFrame to format
            table_index: Table index

        Returns:
            Dictionary in both headers format
        """
        result = {"table_index": table_index, "header_type": "both"}

        # Extract column headers (skip first column)
        column_headers = [_clean_text(str(col)) for col in table_df.columns[1:]]

        # Extract row headers (first column values)
        row_headers = [_clean_text(str(val)) for val in table_df.iloc[:, 0]]

        result["first_column_and_first_row_headers"] = {"column_headers": column_headers, "row_headers": row_headers}

        # Extract data rows
        data_rows = []
        for idx, row in table_df.iterrows():
            row_header = _clean_text(str(row.iloc[0]))
            values = {
                _clean_text(str(col)): _clean_text(str(val)) for col, val in zip(table_df.columns[1:], row.iloc[1:])
            }
            data_rows.append({"row_header": row_header, "values": values})

        result["data_rows_count"] = len(data_rows)
        result["data"] = data_rows

        return result

    def _table_to_simple_json(self, table_df: pd.DataFrame, table_index: int = 0) -> dict:
        """Convert table to simple JSON format (basic row-by-row).

        Args:
            table_df: DataFrame to convert
            table_index: Index of the table

        Returns:
            Dictionary in simple JSON format
        """
        return {
            "table_index": table_index,
            "columns": [_clean_text(str(col)) for col in table_df.columns],
            "rows": [[_clean_text(str(val)) for val in row] for _, row in table_df.iterrows()],
        }

    def _convert_to_natural_language(self, table_json: dict) -> str:
        """Convert JSON table to natural language for embeddings.

        Args:
            table_json: Table in JSON format

        Returns:
            Natural language text representation
        """
        lines = []

        # Add title
        if "title" in table_json:
            table_idx = table_json.get("table_index", "")
            lines.append(f"Table {table_idx}: {table_json['title']}")
        elif "table_index" in table_json:
            lines.append(f"Table {table_json['table_index']}")

        # Add data rows
        if "data" in table_json:
            if table_json.get("header_type") == "both":
                # Both headers format
                for row in table_json["data"]:
                    row_header = row.get("row_header", "")
                    values_text = ", ".join(f"{k}: {v}" for k, v in row.get("values", {}).items())
                    lines.append(f"{row_header}: {values_text}")
            else:
                # Top-only headers format
                for row in table_json["data"]:
                    row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
                    lines.append(row_text)
        elif "rows" in table_json:
            # Simple JSON format
            columns = table_json.get("columns", [])
            for row in table_json["rows"]:
                row_text = ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
                lines.append(row_text)

        # Add description
        if "description" in table_json:
            lines.append(table_json["description"])

        return "\n".join(lines)

    @override
    def get_header_and_body_lines(
        self,
        *,
        table_text: str,
        **kwargs: Any,
    ) -> tuple[list[str], list[str]]:
        """Split JSON table into header and body for chunking.

        For JSON tables, the "header" is the metadata section
        (table_index, header_type, title, etc.) and the "body"
        is the data array.

        Args:
            table_text: JSON string of the table
            **kwargs: Additional parameters

        Returns:
            Tuple of (header_lines, body_lines)
        """
        # If it's natural language format, treat as single body
        if not table_text.strip().startswith("{"):
            return [], [table_text]

        try:
            table_json = json.loads(table_text)
        except json.JSONDecodeError:
            # Fallback: treat entire text as body
            return [], [table_text]

        header_lines = []
        body_lines = []

        # Build header section (metadata)
        header_parts = []
        for key in ["table_index", "header_type", "title", "first_row_headers", "first_column_and_first_row_headers"]:
            if key in table_json:
                value = table_json[key]
                if isinstance(value, str):
                    header_parts.append(f'  "{key}": "{value}"')
                else:
                    header_parts.append(f'  "{key}": {json.dumps(value)}')

        if header_parts:
            header_lines.append("{")
            header_lines.extend([f"{part}," for part in header_parts])
            header_lines.append('  "data": [')

        # Build body section (data rows)
        if "data" in table_json:
            data_json = json.dumps(table_json["data"], indent=2)
            body_lines = data_json.split("\n")
        elif "rows" in table_json:
            rows_json = json.dumps(table_json["rows"], indent=2)
            body_lines = rows_json.split("\n")

        return header_lines, body_lines
