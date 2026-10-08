"""Field-region form items."""

import typing
from enum import Enum

from docling_core.types.doc.common.scalars import LevelNumber
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.text import TextItem
from docling_core.types.doc.labels import DocItemLabel


class FieldControl(str, Enum):
    """The kind of control a form field presents to whoever fills it in.

    This is the document-level reading of the control, not the raw PDF entry:
    an AcroForm ``/Btn`` field resolves to ``CHECKBOX``, ``RADIO`` or
    ``BUTTON`` depending on its flags, and the untouched ``/FT`` name stays
    available on ``PdfWidget.widget_field_type``.
    """

    TEXT = "text"  # a free-text entry, e.g. AcroForm /Tx
    CHECKBOX = "checkbox"  # an independently togglable box
    RADIO = "radio"  # one choice among a mutually exclusive set
    CHOICE = "choice"  # a dropdown or list selection, e.g. AcroForm /Ch
    SIGNATURE = "signature"  # a signature field, e.g. AcroForm /Sig
    BUTTON = "button"  # an action button that holds no value
    OTHER = "other"  # a control this vocabulary does not name


class FieldRegionItem(DocItem):
    label: typing.Literal[DocItemLabel.FIELD_REGION] = DocItemLabel.FIELD_REGION


class FieldHeadingItem(TextItem):
    label: typing.Literal[DocItemLabel.FIELD_HEADING] = DocItemLabel.FIELD_HEADING  # type: ignore[assignment]
    level: LevelNumber = 1


class FieldItem(DocItem):
    """One form field: the control, its metadata, and its key/value children.

    Note:
        Whether the field is editable is carried by its value, as
        ``FieldValueItem.kind``, and is deliberately not repeated here.
    """

    label: typing.Literal[DocItemLabel.FIELD_ITEM] = DocItemLabel.FIELD_ITEM

    control: FieldControl | None = None
    """The kind of control, when known."""

    description: str | None = None
    """A human-readable description of what to enter, from the producer rather
    than from the page -- an AcroForm ``/TU`` tooltip, or an HTML ``title``.
    Distinct from ``FieldHintItem``, which is text drawn on the page."""

    required: bool = False
    """Whether a value must be supplied, e.g. the AcroForm ``Required`` flag."""

    options: list[str] = []
    """The values this field accepts, for ``CHOICE`` and ``RADIO`` controls.
    Empty when the control is unconstrained or the options are unknown."""


class FieldValueItem(TextItem):
    label: typing.Literal[DocItemLabel.FIELD_VALUE] = DocItemLabel.FIELD_VALUE  # type: ignore[assignment]
    kind: typing.Literal["read_only", "fillable"] = "read_only"
