"""Discriminated union of all leaf content item types."""

from typing import Annotated, Union

from pydantic import Field

from docling_core.types.doc.items.code import CodeItem
from docling_core.types.doc.items.form import FieldItem, FieldRegionItem
from docling_core.types.doc.items.key_value import KeyValueItem
from docling_core.types.doc.items.picture.picture import PictureItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import FormulaItem, ListItem, SectionHeaderItem, TextItem, TitleItem

ContentItem = Annotated[
    Union[
        TextItem,
        TitleItem,
        SectionHeaderItem,
        ListItem,
        CodeItem,
        FormulaItem,
        PictureItem,
        TableItem,
        KeyValueItem,
        FieldRegionItem,
        FieldItem,
    ],
    Field(discriminator="label"),
]
