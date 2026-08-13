"""Attachment document item."""

import typing
import warnings
from typing import Optional, Union

from pydantic import AnyUrl, Field, model_validator

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
    mime_type: Optional[str] = Field(default=None, description="MIME type of the attachment payload.")
    size: Optional[int] = Field(default=None, description="Attachment payload size in bytes.")
    target: Optional[Union[str, AnyUrl]] = Field(
        default=None,
        description="Relative path/URL to the converted attachment output, or None if not converted.",
    )
    status: AttachmentStatus = Field(
        default="converted",
        description=(
            "Conversion status of the attachment. Deprecated when `doc_data` is set: "
            "a present `doc_data` implies successful conversion and `status` is ignored."
        ),
    )
    data: Optional[bytes] = Field(
        default=None,
        description="Raw binary payload of the attachment. Stored as base64 in JSON.",
    )
    doc_data: Optional[bytes] = Field(
        default=None,
        description=(
            "Serialized DoclingDocument (e.g. JSON/DCLG/DCLX) of the recursively parsed attachment. "
            "When present, the attachment is implicitly considered converted and `status` is ignored."
        ),
    )

    @model_validator(mode="after")
    def _validate_status_with_doc_data(self) -> "AttachmentItem":
        if self.doc_data is not None and self.status != "converted":
            warnings.warn(
                f"Attachment '{self.name}' has doc_data set but status='{self.status}'; "
                "doc_data implies converted, status will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self

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
        # Simple fallback without requiring DocTagsAttachmentSerializer (deferred per #713).
        # Serializer support will be added in a follow-up PR.
        parts: list[str] = []
        if add_location and self.prov:
            try:
                loc = self.get_location_tokens(doc=doc, xsize=xsize, ysize=ysize)
                if loc:
                    parts.append(loc)
            except Exception:
                pass
        if add_content:
            if self.status == "converted" and self.target:
                parts.append(f"{self.name} ({self.target})")
            elif self.doc_data is not None:
                parts.append(f"{self.name} (converted, embedded document)")
            else:
                reason = self.status.replace("_", " ")
                parts.append(f"{self.name} (not converted: {reason})")
        text = "".join(parts)
        if text:
            text = f"<attachment>{text}</attachment>"
        return text
