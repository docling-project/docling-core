"""Code document item."""

import typing
from typing import TYPE_CHECKING

from typing_extensions import deprecated

from docling_core.types.doc.items.node import FloatingItem
from docling_core.types.doc.items.text import TextItem
from docling_core.types.doc.labels import CodeLanguageLabel, DocItemLabel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class CodeItem(FloatingItem, TextItem):
    """CodeItem."""

    label: typing.Literal[DocItemLabel.CODE] = DocItemLabel.CODE  # type: ignore[assignment]
    code_language: CodeLanguageLabel = CodeLanguageLabel.UNKNOWN

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
