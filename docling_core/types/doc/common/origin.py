"""Document-origin model."""

import mimetypes
import typing
from typing import Optional

from pydantic import AnyUrl, BaseModel, field_validator

from docling_core.types.doc.common.scalars import Uint64


class DocumentOrigin(BaseModel):
    """FileSource."""

    mimetype: str  # the mimetype of the original file
    binary_hash: Uint64  # the binary hash of the original file.
    # TODO: Change to be Uint64 and provide utility method to generate

    filename: str  # The name of the original file, including extension, without path.
    # Could stem from filesystem, source URI, Content-Disposition header, ...

    uri: Optional[AnyUrl] = (
        None  # any possible reference to a source file,
        # from any file handler protocol (e.g. https://, file://, s3://)
    )

    _extra_mimetypes: typing.ClassVar[list[str]] = [
        "application/vnd.box.boxnote",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "text/asciidoc",
        "text/csv",
        "text/markdown",
        "text/vtt",
    ]

    @field_validator("binary_hash", mode="before")
    @classmethod
    def parse_hex_string(cls, value):
        """parse_hex_string."""
        if isinstance(value, str):
            try:
                # Convert hex string to an integer
                hash_int = Uint64(value, 16)
                # Mask to fit within 64 bits (unsigned)
                return hash_int & 0xFFFFFFFFFFFFFFFF  # TODO be sure it doesn't clip uint64 max
            except ValueError:
                raise ValueError(f"Invalid sha256 hexdigest: {value}")
        return value  # If already an int, return it as is.

    @field_validator("mimetype")
    @classmethod
    def validate_mimetype(cls, v):
        """validate_mimetype."""
        # Check if the provided MIME type is valid using mimetypes module
        if v not in mimetypes.types_map.values() and v not in cls._extra_mimetypes:
            raise ValueError(f"'{v}' is not a valid MIME type")
        return v
