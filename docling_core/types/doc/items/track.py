"""Track item: a time-aligned media transcript."""

import typing

from pydantic import field_validator

from docling_core.types.doc.common.reference import MediaRef
from docling_core.types.doc.items.node import FloatingItem
from docling_core.types.doc.labels import DocItemLabel


class TrackItem(FloatingItem):
    """A time-aligned media transcript, e.g. the speech of an audio or video recording.

    A track maps to the DocLang ``<track>`` element. Its children are the cues: items
    whose ``source`` is a ``TrackSource`` with the start and end time and an optional
    voice, such as the ``TextItem`` of each transcript turn or the ``PictureItem`` of a
    sampled video frame. The ``image`` is the cover, one picture for the whole track
    (e.g. the poster frame of a video). Like other floating items, a track can carry a
    location (``prov``) and captions. A formatted turn is a ``TextItem`` with empty text
    and an ``InlineGroup`` child.

    The ``audio`` or ``video`` is the recording the times refer to, measured from its
    start. A video covers its own sound; both are set only for separate files.
    """

    label: typing.Literal[DocItemLabel.TRACK] = DocItemLabel.TRACK
    audio: MediaRef | None = None
    video: MediaRef | None = None

    @field_validator("audio")
    @classmethod
    def _check_audio(cls, v: MediaRef | None) -> MediaRef | None:
        if v is not None and not v.mimetype.startswith("audio/"):
            raise ValueError(f"Not an audio media type: {v.mimetype!r}")
        return v

    @field_validator("video")
    @classmethod
    def _check_video(cls, v: MediaRef | None) -> MediaRef | None:
        if v is not None and not v.mimetype.startswith("video/"):
            raise ValueError(f"Not a video media type: {v.mimetype!r}")
        return v
