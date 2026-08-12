"""Tests for AttachmentItem schema and serialization."""

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


def test_add_attachment_unpositioned_markdown():
    doc = DoclingDocument(name="test")
    doc.add_attachment(name="report.pdf", target="report.md")
    md = doc.export_to_markdown()
    assert "## Attachments" in md
    assert "[report.pdf](report.md)" in md


def test_add_attachment_positioned_markdown():
    doc = DoclingDocument(name="test")
    doc.add_text(DocItemLabel.PARAGRAPH, "Before")
    doc.add_attachment(
        name="annex.pdf",
        target="annex.md",
        prov=_prov(),
    )
    doc.add_text(DocItemLabel.PARAGRAPH, "After")
    md = doc.export_to_markdown()
    assert "[annex.pdf](annex.md)" in md
    assert "## Attachments" not in md
    lines = [line for line in md.splitlines() if line.strip()]
    assert lines == ["Before", "[annex.pdf](annex.md)", "After"]


def test_attachment_non_converted_statuses():
    doc = DoclingDocument(name="test")
    doc.add_attachment(name="bad.exe", status="unsupported")
    doc.add_attachment(name="deep.pdf", status="depth_limited")
    doc.add_attachment(name="fail.pdf", status="failed")
    md = doc.export_to_markdown()
    assert "bad.exe (not converted: unsupported)" in md
    assert "deep.pdf (not converted: depth limited)" in md
    assert "fail.pdf (not converted: failed)" in md


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


def test_attachment_export_to_doctags():
    from docling_core.types.doc.tokens import DocumentToken

    doc = DoclingDocument(name="test")
    att = doc.add_attachment(name="spec.pdf", target="spec.md")
    doctags = att.export_to_doctags(doc=doc)
    assert "<attachment>spec.pdf (spec.md)</attachment>" in doctags
    assert (
        DocumentToken.create_token_name_from_doc_item_label(DocItemLabel.ATTACHMENT) == DocumentToken.ATTACHMENT.value
    )


def test_attachment_exclusion_filtering():
    doc = DoclingDocument(name="test")
    doc.add_attachment(name="doc.pdf", target="doc.md")
    md_all = doc.export_to_markdown()
    assert "doc.pdf" in md_all

    md_none = doc.export_to_markdown(labels={DocItemLabel.PARAGRAPH})
    assert "doc.pdf" not in md_none
