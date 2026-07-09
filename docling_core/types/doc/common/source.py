"""Source models describing where extracted media cues originate."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


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
    - For documents with an implicit or explicit layout, such as PDF, HTML, docx, pptx, or markdown files, the
        `ProvenanceItem` should still be used.
"""
