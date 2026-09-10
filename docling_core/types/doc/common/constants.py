"""Document-level constants: schema version and default export label sets."""

import re
from typing import Final

from docling_core.types.doc.labels import DocItemLabel

CURRENT_VERSION: Final = "1.10.0"
"""Current DoclingDocument schema version.

Bump this (minor) whenever a change to the serialised format would cause an
older docling-core client to fail ``DoclingDocument.model_validate()``.
See CONTRIBUTING.md for the full checklist.
"""

SCHEMA_VERSION_HISTORY: Final[list[dict]] = [
    {"schema": "1.5.0", "library_version": "v2.38.2", "note": "ListGroup, remodelled lists"},
    {"schema": "1.6.0", "library_version": "v2.45.0", "note": "RichTableCell, AnyTableCell union"},
    {"schema": "1.7.0", "library_version": "v2.47.0", "note": "TableCell.fillable"},
    {"schema": "1.8.0", "library_version": "v2.49.0", "note": "BasePrediction, BaseMeta, meta field"},
    {"schema": "1.9.0", "library_version": "v2.57.0", "note": "FineRef, DocItem.comments"},
    {"schema": "1.10.0", "library_version": "v2.69.0", "note": "FieldItem types, 7 new DocItemLabel values"},
]
"""Ordered history of DoclingDocument schema versions in docling-core v2.x.

Each entry records the schema version string, the first library release that
introduced it, and a short description of what changed.  This list is the
authoritative record consumed by ``scripts/check_compat_projectors.py``.

Note:
    Only versions first shipped with docling-core v2.x are included.
    Schema versions 1.0-1.4 belong to the v1.x era and are outside the
    supported downgrade range.
    When ``CURRENT_VERSION`` is bumped to ``"1.N.0"``, append a new entry.
"""

_CURRENT_MINOR: Final[int] = int(re.match(r"^\d+\.(\d+)\.", CURRENT_VERSION).group(1))  # type: ignore[union-attr]
"""Minor component of ``CURRENT_VERSION``, derived automatically. Do not edit."""

FIRST_SUPPORTED_MINOR: Final[int] = int(
    re.match(r"^\d+\.(\d+)\.", SCHEMA_VERSION_HISTORY[0]["schema"]).group(1)  # type: ignore[union-attr]
)
"""Minor version of the oldest schema supported by the downgrade-projector system.

This is the minor component of the first entry in ``SCHEMA_VERSION_HISTORY``
(currently ``5``, corresponding to schema ``1.5.0``).  A projector exists for
every step from ``_CURRENT_MINOR`` down to ``FIRST_SUPPORTED_MINOR + 1``.
"""


DEFAULT_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.KEY_VALUE_REGION,
    DocItemLabel.EMPTY_VALUE,
    DocItemLabel.FIELD_KEY,
    DocItemLabel.FIELD_VALUE,
    DocItemLabel.FIELD_HEADING,
    DocItemLabel.FIELD_HINT,
    DocItemLabel.MARKER,
    DocItemLabel.HANDWRITTEN_TEXT,
}


DOCUMENT_TOKENS_EXPORT_LABELS = DEFAULT_EXPORT_LABELS.copy()
DOCUMENT_TOKENS_EXPORT_LABELS.update(
    [
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.FORM,
    ]
)
