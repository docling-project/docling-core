"""Text-bearing document items: TextItem and subclasses."""

import typing
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import AnyUrl, Field
from typing_extensions import deprecated

from docling_core.types.doc.common.formatting import Formatting
from docling_core.types.doc.common.scalars import LevelNumber
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.labels import DocItemLabel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class TextItem(DocItem):
    """TextItem."""

    label: typing.Literal[
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.REFERENCE,
        DocItemLabel.TEXT,
        DocItemLabel.EMPTY_VALUE,
        DocItemLabel.FIELD_KEY,
        DocItemLabel.FIELD_HINT,
        DocItemLabel.MARKER,
        DocItemLabel.HANDWRITTEN_TEXT,
    ]

    orig: str  # untreated representation
    text: str  # sanitized representation

    formatting: Optional[Formatting] = None
    hyperlink: Optional[Union[AnyUrl, Path]] = Field(union_mode="left_to_right", default=None)

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class TitleItem(TextItem):
    """TitleItem."""

    label: typing.Literal[DocItemLabel.TITLE] = DocItemLabel.TITLE  # type: ignore[assignment]


class SectionHeaderItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.SECTION_HEADER] = DocItemLabel.SECTION_HEADER  # type: ignore[assignment]
    level: LevelNumber = 1

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        r"""Export text element to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
            ),
        )
        text = serializer.serialize(item=self).text
        return text


class ListItem(TextItem):
    """SectionItem."""

    label: typing.Literal[DocItemLabel.LIST_ITEM] = DocItemLabel.LIST_ITEM  # type: ignore[assignment]
    enumerated: bool = False
    marker: str = "-"  # The bullet or number symbol that prefixes this list item


class FormulaItem(TextItem):
    """FormulaItem."""

    label: typing.Literal[DocItemLabel.FORMULA] = DocItemLabel.FORMULA  # type: ignore[assignment]
