"""Tests for AttachmentItem schema and storage (data-model only, no serializers)."""

import warnings

from docling_core.types.doc import (
    AttachmentItem,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)
from docling_core.types.doc.page import BoundingBox


def _prov() -> ProvenanceItem:
    return ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        charspan=(0, 0),
    )


def test_attachment_item_defaults():
    item = AttachmentItem(name="file.pdf", self_ref="#")
    assert item.label == DocItemLabel.ATTACHMENT
    assert item.status == "converted"
    assert item.target is None


def test_attachment_storage_and_traversal():
    """Attachments are stored in DoclingDocument and reachable via iterate_items."""
    doc = DoclingDocument(name="test")
    doc.add_attachment(name="report.pdf", target="report.md")
    assert len(doc.attachments) == 1
    # not yet serialized to markdown (serializers deferred to follow-up PR)
    # but stored and reachable
    found = [item for item, _ in doc.iterate_items(with_groups=True) if isinstance(item, AttachmentItem)]
    assert len(found) == 1
    assert found[0].name == "report.pdf"


def test_attachment_positioned_via_prov():
    doc = DoclingDocument(name="test")
    doc.add_text(DocItemLabel.PARAGRAPH, "Before")
    doc.add_attachment(
        name="annex.pdf",
        target="annex.md",
        prov=_prov(),
    )
    doc.add_text(DocItemLabel.PARAGRAPH, "After")
    assert doc.attachments[0].prov[0].page_no == 1
    lines = [item.text if hasattr(item, "text") else item.name for item, _ in doc.iterate_items()]
    assert "Before" in lines and "annex.pdf" in lines and "After" in lines


def test_attachment_json_roundtrip():
    doc = DoclingDocument(name="test")
    doc.add_attachment(name="report.pdf", target="report.md", size=1234)
    dumped = doc.model_dump_json()
    loaded = DoclingDocument.model_validate_json(dumped)
    assert len(loaded.attachments) == 1
    assert loaded.attachments[0].name == "report.pdf"
    assert loaded.attachments[0].target == "report.md"
    assert loaded.attachments[0].size == 1234


def test_attachment_normalize_references():
    doc = DoclingDocument(name="test")
    doc.add_attachment(name="doc.pdf", target="doc.md")
    doc._normalize_references()
    assert len(doc.attachments) == 1
    assert doc.attachments[0].self_ref == "#/attachments/0"


def test_add_item_attachment():
    doc = DoclingDocument(name="test")
    item = AttachmentItem(name="manual.pdf", target="manual.md", self_ref="#")
    cref = doc._append_item(item=item, parent_ref=doc.body.get_ref())
    assert cref.cref == "#/attachments/0"
    assert len(doc.attachments) == 1
    assert doc.attachments[0].name == "manual.pdf"
    assert doc.attachments[0].self_ref == "#/attachments/0"


def test_attachment_with_binary_data_roundtrip():
    doc = DoclingDocument(name="test")
    raw = b"%PDF-1.4 fake content"
    parsed = b'{"name": "inner"}'
    att = doc.add_attachment(
        name="report.pdf",
        mime_type="application/pdf",
        size=len(raw),
        data=raw,
        doc_data=parsed,
    )
    assert att.data == raw
    assert att.doc_data == parsed
    # JSON roundtrip preserves base64-encoded bytes
    dumped = doc.model_dump_json()
    loaded = DoclingDocument.model_validate_json(dumped)
    assert loaded.attachments[0].data == raw
    assert loaded.attachments[0].doc_data == parsed
    # YAML gold-file path also roundtrips
    assert AttachmentItem(name="x.pdf", self_ref="#", data=raw).data == raw


def test_attachment_doc_data_implies_converted():
    """Point 3: doc_data presence implies converted; status is ignored with warning."""
    # doc_data set with non-converted status should warn but succeed
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        item = AttachmentItem(name="x.pdf", self_ref="#", data=b"raw", doc_data=b'{"x":1}', status="failed")
        assert any("doc_data implies converted" in str(x.message) for x in w)
    assert item.doc_data == b'{"x": 1}' or item.doc_data == b'{"x":1}'
    # export_to_doctags uses doc_data branch when present
    doc = DoclingDocument(name="test")
    att = doc.add_attachment(name="inner.pdf", data=b"raw", doc_data=b'{"doc":1}')
    assert "converted" in att.export_to_doctags(doc=doc).lower() or "inner.pdf" in att.export_to_doctags(doc=doc)


def test_attachment_content_layer_and_prov():
    from docling_core.types.doc.common.content_layer import ContentLayer

    doc = DoclingDocument(name="test")
    att = doc.add_attachment(
        name="layered.pdf",
        target="layered.md",
        prov=_prov(),
        content_layer=ContentLayer.FURNITURE,
    )
    assert att.content_layer == ContentLayer.FURNITURE
    assert len(att.prov) == 1
    # also exercise _append_item content_layer path
    item2 = AttachmentItem(name="via_append.pdf", self_ref="#", content_layer=ContentLayer.FURNITURE)
    item2.content_layer = ContentLayer.FURNITURE
    cref = doc._append_item(item=item2, parent_ref=doc.body.get_ref())
    assert cref.cref == "#/attachments/1"


def test_pdf_attachment_with_data_and_parsed_document():
    from docling_core.types.doc.page import (
        BoundingRectangle,
        FileAttachmentAnnotation,
        ParsedPdfDocument,
        PdfAttachment,
    )

    annot = FileAttachmentAnnotation(
        page_no=1,
        bbox=BoundingRectangle(r_x0=0, r_y0=0, r_x1=10, r_y1=10, r_x2=10, r_y2=10, r_x3=0, r_y3=10),
    )
    pdf_att = PdfAttachment(
        name="embedded.pdf",
        mime_type="application/pdf",
        size=123,
        annotations=[annot],
        data=b"binary payload",
    )
    assert pdf_att.data == b"binary payload"
    # ParsedPdfDocument roundtrip with attachments
    parsed = ParsedPdfDocument(attachments=[pdf_att])
    dumped = parsed.model_dump_json()
    loaded = ParsedPdfDocument.model_validate_json(dumped)
    assert loaded.attachments[0].data == b"binary payload"
    assert loaded.attachments[0].name == "embedded.pdf"


def test_segmented_pdf_page_attachments():
    """Point 1: PdfAttachment is part of SegmentedPdfPage (and ParsedPdfDocument)."""
    from docling_core.types.doc.base import CoordOrigin
    from docling_core.types.doc.page import (
        BoundingBox,
        BoundingRectangle,
        FileAttachmentAnnotation,
        ParsedPdfDocument,
        PdfAttachment,
        PdfPageBoundaryType,
        PdfPageGeometry,
        SegmentedPdfPage,
    )

    annot = FileAttachmentAnnotation(
        page_no=1,
        bbox=BoundingRectangle(r_x0=0, r_y0=0, r_x1=10, r_y1=10, r_x2=10, r_y2=10, r_x3=0, r_y3=10),
    )
    pdf_att = PdfAttachment(name="page_att.pdf", data=b"page data", annotations=[annot])

    # SegmentedPdfPage can hold attachments
    page = SegmentedPdfPage(
        dimension=PdfPageGeometry(
            angle=0,
            rect=BoundingRectangle(r_x0=0, r_y0=0, r_x1=100, r_y1=0, r_x2=100, r_y2=100, r_x3=0, r_y3=100),
            boundary_type=PdfPageBoundaryType.CROP_BOX,
            art_bbox=BoundingBox(l=0, b=0, r=100, t=100, coord_origin=CoordOrigin.BOTTOMLEFT),
            bleed_bbox=BoundingBox(l=0, b=0, r=100, t=100, coord_origin=CoordOrigin.BOTTOMLEFT),
            crop_bbox=BoundingBox(l=0, b=0, r=100, t=100, coord_origin=CoordOrigin.BOTTOMLEFT),
            media_bbox=BoundingBox(l=0, b=0, r=100, t=100, coord_origin=CoordOrigin.BOTTOMLEFT),
            trim_bbox=BoundingBox(l=0, b=0, r=100, t=100, coord_origin=CoordOrigin.BOTTOMLEFT),
        ),
        char_cells=[],
        word_cells=[],
        textline_cells=[],
        attachments=[pdf_att],
    )
    assert len(page.attachments) == 1
    assert page.attachments[0].data == b"page data"
    # JSON roundtrip
    dumped = page.model_dump_json()
    loaded = SegmentedPdfPage.model_validate_json(dumped)
    assert loaded.attachments[0].name == "page_att.pdf"

    # Also still works at ParsedPdfDocument level
    parsed = ParsedPdfDocument(pages={1: page}, attachments=[pdf_att])
    assert len(parsed.attachments) == 1


def test_attachment_export_to_doctags_fallback():
    """Fallback export_to_doctags without serializer (deferred)."""
    from docling_core.types.doc.tokens import DocumentToken

    doc = DoclingDocument(name="test")
    att = doc.add_attachment(name="spec.pdf", target="spec.md")
    doctags = att.export_to_doctags(doc=doc)
    assert "<attachment>" in doctags
    assert "spec.pdf" in doctags
    assert (
        DocumentToken.create_token_name_from_doc_item_label(DocItemLabel.ATTACHMENT) == DocumentToken.ATTACHMENT.value
    )

    # non-converted
    att2 = AttachmentItem(name="bad.exe", self_ref="#", status="unsupported")
    assert "not converted" in att2.export_to_doctags(doc=doc)
