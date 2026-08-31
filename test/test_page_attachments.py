"""Tests for page-level PDF attachment models."""

from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.page import (
    BoundingRectangle,
    FileAttachmentAnnotation,
    ParsedPdfDocument,
    PdfAttachment,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
)


def _box(l: float = 0, b: float = 0, r: float = 612, t: float = 792) -> BoundingBox:
    return BoundingBox(l=l, b=b, r=r, t=t, coord_origin=CoordOrigin.BOTTOMLEFT)


def _geometry() -> PdfPageGeometry:
    rect = BoundingRectangle(
        r_x0=0,
        r_y0=0,
        r_x1=612,
        r_y1=0,
        r_x2=612,
        r_y2=792,
        r_x3=0,
        r_y3=792,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    return PdfPageGeometry(
        angle=0.0,
        rect=rect,
        boundary_type=PdfPageBoundaryType.MEDIA_BOX,
        art_bbox=_box(),
        bleed_bbox=_box(),
        crop_bbox=_box(),
        media_bbox=_box(),
        trim_bbox=_box(),
    )


def _annotation() -> FileAttachmentAnnotation:
    return FileAttachmentAnnotation(
        page_no=1,
        bbox=BoundingRectangle(
            r_x0=10,
            r_y0=10,
            r_x1=110,
            r_y1=10,
            r_x2=110,
            r_y2=30,
            r_x3=10,
            r_y3=30,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        ),
    )


def _attachment(name: str = "notes.txt", data: bytes | None = b"hello world") -> PdfAttachment:
    return PdfAttachment(
        name=name,
        mime_type="text/plain",
        size=len(data) if data else 0,
        annotations=[_annotation()],
        data=data,
    )


def test_pdf_attachment_json_round_trip():
    """A PdfAttachment carrying binary data survives a JSON round-trip."""
    att = _attachment()
    restored = PdfAttachment.model_validate_json(att.model_dump_json())
    assert restored == att
    assert restored.name == "notes.txt"
    assert restored.data == b"hello world"


def test_pdf_attachment_defaults():
    """Unset optional fields get sane defaults."""
    att = PdfAttachment(name="empty.bin")
    assert att.mime_type is None
    assert att.size == 0
    assert att.annotations == []
    assert att.data is None


def _page(attachments: list[PdfAttachment] | None = None) -> SegmentedPdfPage:
    return SegmentedPdfPage(
        dimension=_geometry(),
        char_cells=[],
        word_cells=[],
        textline_cells=[],
        attachments=attachments or [],
    )


def test_segmented_pdf_page_carries_attachments():
    """A SegmentedPdfPage exposes the attachments anchored to it, defaulting to empty."""
    page = _page(attachments=[_attachment()])
    assert len(page.attachments) == 1
    assert page.attachments[0].name == "notes.txt"
    assert page.attachments[0].data == b"hello world"

    empty = _page()
    assert empty.attachments == []


def test_parsed_pdf_document_round_trip_with_attachments():
    """Document-level attachments survive a JSON round-trip alongside page-level ones."""
    doc = ParsedPdfDocument(pages={1: _page(attachments=[_attachment()])})
    doc.attachments.append(_attachment(name="archive.zip", data=b"\x00\x01\x02"))

    restored = ParsedPdfDocument.model_validate_json(doc.model_dump_json())

    assert restored.attachments == doc.attachments
    assert restored.attachments[0].name == "archive.zip"
    assert restored.attachments[0].data == b"\x00\x01\x02"
    assert restored.pages[1].attachments[0].name == "notes.txt"
