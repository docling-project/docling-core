"""Document-level constants: schema version and default export label sets."""

from typing import Final

from docling_core.types.doc.labels import DocItemLabel

CURRENT_VERSION: Final = "1.10.0"


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
