# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

import pytest

from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, Size
from docling_core.types.doc.document import ProvenanceItem


@pytest.mark.parametrize("target_kind", ["text", "picture"])
def test_concatenate_preserves_comment_targets_and_ranges(target_kind: str):
    first = DoclingDocument(name="first")
    first_target = first.add_text(label=DocItemLabel.PARAGRAPH, text="First target")
    first.add_comment(text="First comment", targets=[first_target])

    second = DoclingDocument(name="second")
    second_target = (
        second.add_text(label=DocItemLabel.PARAGRAPH, text="Second target")
        if target_kind == "text"
        else second.add_picture()
    )
    if target_kind == "text":
        second.add_comment(text="Second comment", targets=[(second_target, (1, 4))])
    else:
        second.add_comment(text="Second comment", targets=[second_target])
    originals = [first.model_dump(), second.model_dump()]

    merged = DoclingDocument.concatenate([first, second])
    # The wrong reference also survives JSON validation, so check its meaning
    # after a public serialization roundtrip, not just that it resolves.
    restored = DoclingDocument.model_validate_json(merged.model_dump_json())
    restored_target = restored.texts[2] if target_kind == "text" else restored.pictures[0]
    assert restored_target.comments[0].resolve(restored).text == "Second comment"
    assert restored_target.comments[0].range == ((1, 4) if target_kind == "text" else None)
    assert restored.texts[0].comments[0].resolve(restored).text == "First comment"
    assert restored.texts[0].comments[0].range is None
    assert [first.model_dump(), second.model_dump()] == originals


def _provenance(page_no: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=0, t=0, r=50, b=10),
        charspan=(0, 6),
    )


def test_filter_remaps_retained_comments_and_drops_excluded_comments():
    doc = DoclingDocument(name="pages")
    for page_no in [1, 2]:
        doc.add_page(page_no=page_no, size=Size(width=100, height=100))
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Excluded target", prov=_provenance(1))
    target = doc.add_text(label=DocItemLabel.PARAGRAPH, text="Retained target", prov=_provenance(2))
    doc.add_comment(text="Retained comment", prov=_provenance(2), targets=[(target, (1, 4))])
    doc.add_comment(text="Excluded comment", prov=_provenance(1), targets=[target])
    original = doc.model_dump()

    filtered = doc.filter(page_nrs={2})
    restored = DoclingDocument.model_validate_json(filtered.model_dump_json())
    retained = restored.texts[0]
    assert retained.text == "Retained target"
    assert len(retained.comments) == 1
    assert retained.comments[0].resolve(restored).text == "Retained comment"
    assert retained.comments[0].range == (1, 4)
    assert set(restored.pages) == {2}
    assert doc.model_dump() == original


def test_filter_preserves_an_unchanged_comment_reference():
    doc = DoclingDocument(name="one page")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    target = doc.add_text(label=DocItemLabel.PARAGRAPH, text="Target", prov=_provenance(1))
    doc.add_comment(text="Comment", prov=_provenance(1), targets=[target])

    filtered = doc.filter(page_nrs={1})
    assert filtered.texts[0].comments[0].resolve(filtered).text == "Comment"
    assert filtered.texts[0].comments[0].range is None


def test_concatenate_unannotated_documents():
    docs = [DoclingDocument(name=name) for name in ["first", "second"]]
    for doc in docs:
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=doc.name)
    merged = DoclingDocument.concatenate(docs)
    assert [item.text for item in merged.texts] == ["first", "second"]
    assert all(not item.comments for item in merged.texts)
