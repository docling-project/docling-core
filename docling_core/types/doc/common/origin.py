"""Document-origin and source models."""

import mimetypes
import typing
from typing import Annotated, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

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
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "text/asciidoc",
        "text/markdown",
        "text/csv",
        "text/vtt",
        "audio/x-wav",
        "audio/wav",
        "audio/mp3",
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


class BaseSource(BaseModel):
    """Base class for source information.

    Represents the source of an extracted component within a digital asset.
    """

    kind: Annotated[str, Field(description="Kind of source. It is used as a discriminator for the source type.")]


class TrackSource(BaseSource):
    """Source metadata for a cue extracted from a media track.

    A `TrackSource` instance identifies a cue in a media track (audio, video, subtitles, screen-recording captions,
    etc.). A *cue* here refers to any discrete segment that was pulled out of the original asset, e.g., a subtitle
    block, an audio clip, or a timed marker in a screen-recording.
    """

    model_config = ConfigDict(regex_engine="python-re")
    kind: Annotated[Literal["track"], Field(description="Identifies this type of source.")] = "track"
    start_time: Annotated[
        float,
        Field(
            examples=[11.0, 6.5, 5370.0],
            description="Start time offset of the track cue in seconds",
        ),
    ]
    end_time: Annotated[
        float,
        Field(
            examples=[12.0, 8.2, 5370.1],
            description="End time offset of the track cue in seconds",
        ),
    ]
    identifier: Annotated[
        str | None, Field(description="An identifier of the cue", examples=["test", "123", "b72d946"])
    ] = None
    voice: Annotated[
        str | None,
        Field(description="The name of the voice in this track (the speaker)", examples=["John", "Mary", "Speaker 1"]),
    ] = None

    @model_validator(mode="after")
    def check_order(self) -> Self:
        """Ensure start time is less than the end time."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        return self


SourceType = Annotated[Union[TrackSource], Field(discriminator="kind")]
"""Union type for all source types.

This type alias represents a discriminated union of all available source types that can be associated with
extracted elements in a document. The `kind` field is used as a discriminator to determine the specific
source type at runtime.

Currently supported source types:
    - `TrackSource`: For elements extracted from media assets (audio, video, subtitles)

Notes:
    - Additional source types may be added to this union in the future to support other content sources.
    - For documents with an implicit or explicity layout, such as PDF, HTML, docx, pptx, or markdown files, the
        `ProvenanceItem` should still be used.
"""
