"""Field-region form items."""

import typing

from docling_core.types.doc.common.scalars import LevelNumber
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.text import TextItem
from docling_core.types.doc.labels import DocItemLabel


class FieldRegionItem(DocItem):
    label: typing.Literal[DocItemLabel.FIELD_REGION] = DocItemLabel.FIELD_REGION


class FieldHeadingItem(TextItem):
    label: typing.Literal[DocItemLabel.FIELD_HEADING] = DocItemLabel.FIELD_HEADING  # type: ignore[assignment]
    level: LevelNumber = 1


class FieldItem(DocItem):
    label: typing.Literal[DocItemLabel.FIELD_ITEM] = DocItemLabel.FIELD_ITEM


class FieldValueItem(TextItem):
    label: typing.Literal[DocItemLabel.FIELD_VALUE] = DocItemLabel.FIELD_VALUE  # type: ignore[assignment]
    kind: typing.Literal["read_only", "fillable"] = "read_only"
