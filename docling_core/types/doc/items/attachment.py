"""Attachment document item."""

import typing
from typing import Optional, Union

from pydantic import AnyUrl, Field

from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.labels import DocItemLabel


class AttachmentItem(DocItem):
    """An embedded file attachment referenced by a document."""

    label: typing.Literal[DocItemLabel.ATTACHMENT] = DocItemLabel.ATTACHMENT  # type: ignore[assignment]

    name: str = Field(description="Original attachment filename.")
    mime_type: Optional[str] = Field(default=None, description="MIME type of the attachment payload.")
    size: Optional[int] = Field(default=None, description="Attachment payload size in bytes.")
    target: Optional[Union[str, AnyUrl]] = Field(
        default=None,
        description="Relative path/URL to the converted attachment output, or None if not converted.",
    )
    data: Optional[bytes] = Field(
        default=None,
        description="Raw binary payload of the attachment. Stored as base64 in JSON.",
    )
    doc_data: Optional[bytes] = Field(
        default=None,
        description=(
            "Serialized DoclingDocument (e.g. JSON/DCLG/DCLX) of the recursively parsed attachment. "
            "When present, the attachment is implicitly considered converted."
        ),
    )
