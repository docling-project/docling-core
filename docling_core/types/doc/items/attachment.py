"""Attachment document item."""

import typing
from typing import Optional, Union

from pydantic import AnyUrl, Field

from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.labels import DocItemLabel

if typing.TYPE_CHECKING:
    from pathlib import Path

    from docling_core.types.doc.document import DoclingDocument


AttachmentStatus = typing.Literal["converted", "failed", "unsupported", "depth_limited"]


class AttachmentItem(DocItem):
    """An embedded file attachment referenced by a document."""

    label: typing.Literal[DocItemLabel.ATTACHMENT] = DocItemLabel.ATTACHMENT  # type: ignore[assignment]

    name: str = Field(description="Original attachment filename.")
    mime_type: Optional[str] = Field(
        default=None, description="MIME type of the attachment payload."
    )
    size: Optional[int] = Field(
        default=None, description="Attachment payload size in bytes."
    )
    target: Optional[Union[str, AnyUrl]] = Field(
        default=None,
        description="Relative path/URL to the converted attachment output, or None if not converted.",
    )
    status: AttachmentStatus = Field(
        default="converted",
        description="Conversion status of the attachment.",
    )

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_content: bool = True,
    ):
        """Export to document tokens format."""
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
        return serializer.serialize(item=self).text
