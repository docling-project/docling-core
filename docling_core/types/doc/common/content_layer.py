"""Content-layer enumeration for document nodes."""

from enum import Enum


class ContentLayer(str, Enum):
    """ContentLayer."""

    BODY = "body"  # main content of the document
    FURNITURE = "furniture"  # eg page-headers/footers
    BACKGROUND = "background"  # eg watermarks
    INVISIBLE = "invisible"  # hidden or invisible text
    NOTES = "notes"  # author/speaker notes, corrections, etc


DEFAULT_CONTENT_LAYERS = {ContentLayer.BODY}
