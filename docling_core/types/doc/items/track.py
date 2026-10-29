"""Track item: a time-aligned media transcript."""

import typing

from docling_core.types.doc.items.node import FloatingItem
from docling_core.types.doc.labels import DocItemLabel


class TrackItem(FloatingItem):
    """A time-aligned media transcript, e.g. the speech of an audio or video recording.

    A track maps to the DocLang ``<track>`` element. Its children are the cues: items
    whose ``source`` is a ``TrackSource`` with the start and end time and an optional
    voice, such as the ``TextItem`` of each transcript turn or the ``PictureItem`` of a
    sampled video frame. The ``image`` is the cover, one picture for the whole track
    (e.g. the poster frame of a video). Like other floating items, a track can carry a
    location (``prov``) and captions.
    """

    label: typing.Literal[DocItemLabel.TRACK] = DocItemLabel.TRACK
