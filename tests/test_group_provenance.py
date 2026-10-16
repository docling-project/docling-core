"""Optional group geometry, without inherited provenance or page membership."""

import warnings

import pytest

from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    InlineGroup,
    ListGroup,
    ProvenanceItem,
    Size,
)


def _prov(page_no: int = 1) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=10, t=20, r=80, b=90, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, 0),
    )


def _doc(*page_nos: int) -> DoclingDocument:
    doc = DoclingDocument(name="group provenance")
    for page_no in page_nos:
        doc.add_page(page_no=page_no, size=Size(width=100, height=100))
    return doc


@pytest.mark.parametrize(
    "method,label,group_type",
    [
        ("add_group", None, GroupItem),
        ("add_list_group", None, ListGroup),
        ("add_inline_group", None, InlineGroup),
        ("add_ordered_list", None, ListGroup),
        ("add_unordered_list", None, ListGroup),
        ("add_group", GroupLabel.LIST, ListGroup),
        ("add_group", GroupLabel.ORDERED_LIST, ListGroup),
        ("add_group", GroupLabel.INLINE, InlineGroup),
    ],
)
def test_group_creation_with_optional_provenance(method: str, label: GroupLabel | None, group_type: type[GroupItem]):
    doc = _doc(1)
    kwargs = {"label": label} if label is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        unlocated = getattr(doc, method)(**kwargs)
        located = getattr(doc, method)(**kwargs, prov=_prov())

    assert type(located) is group_type
    assert located.prov == [_prov()]
    assert unlocated.prov == []
    assert "prov" not in unlocated.model_dump()
    assert located.parent == doc.body.get_ref()
    assert located.get_ref() in doc.body.children


def test_group_provenance_roundtrip_and_ownership():
    doc = _doc(1, 2)
    group = doc.add_group(prov=_prov())
    nested = doc.add_list_group(parent=group, prov=_prov(2))
    child = doc.add_list_item(text="Unlocated", parent=nested)
    group.prov.append(_prov(2))

    loaded = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert loaded == doc
    assert loaded.groups[0].prov == [_prov(), _prov(2)]
    assert loaded.groups[1].prov == [_prov(2)]
    assert child.prov == loaded.texts[0].prov == []
    assert "prov" not in loaded.body.model_dump()
    assert loaded.texts[0].model_dump()["prov"] == []

    legacy_data = doc.model_dump(mode="json")
    legacy_data["version"] = "1.10.0"
    for item in legacy_data["groups"]:
        item.pop("prov")
    assert all(not item.prov for item in DoclingDocument.model_validate(legacy_data).groups)


@pytest.mark.parametrize("owner", ["body", "furniture", "group", "list_group", "inline_group"])
def test_document_validation_clamps_group_provenance(owner: str):
    doc = _doc(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        group = getattr(doc, owner) if owner in {"body", "furniture"} else getattr(doc, f"add_{owner}")()
    prov = _prov()
    prov.bbox.l = -5
    group.prov.append(prov)

    with pytest.warns(UserWarning, match="coordinate l on page 1 is outside page bounds"):
        validated = DoclingDocument.model_validate(doc.model_dump())
    assert group.get_ref().resolve(validated).prov[0].bbox.l == 0


def test_filter_trims_group_geometry_without_changing_child_page_membership():
    doc = _doc(1, 2, 3)
    doc.body.prov = [_prov(), _prov(2)]
    group = doc.add_group(prov=_prov())
    group.prov.append(_prov(3))
    doc.add_text(label=DocItemLabel.TEXT, text="Page two", parent=group, prov=_prov(2))
    doc.add_text(label=DocItemLabel.TEXT, text="Unlocated", parent=group)

    page_one = doc.filter(page_nrs={1})
    assert page_one.groups[0].prov == [_prov()]
    assert page_one.groups[0].children == []
    assert page_one.texts == []

    page_two = doc.filter(page_nrs={2})
    assert page_two.body.prov == [_prov(2)]
    assert page_two.groups[0].prov == []
    assert [item.text for item in page_two.texts] == ["Page two"]
    assert page_two.texts[0].parent == page_two.groups[0].get_ref()
    assert set(page_two.pages) == {2}
    page_two.validate_document()

    assert group.prov == [_prov(), _prov(3)]
    assert doc.body.prov == [_prov(), _prov(2)]


def test_concatenate_offsets_group_and_root_provenance_without_mutating_sources():
    doc = _doc(5, 6)
    doc.body.prov = [_prov(5), _prov(6)]
    group = doc.add_group(prov=_prov(5))
    doc.add_inline_group(parent=group, prov=_prov(6))
    doc.add_text(label=DocItemLabel.TEXT, text="Unlocated", parent=group)

    merged = DoclingDocument.concatenate([doc, doc])
    assert [prov.page_no for prov in merged.body.prov] == [1, 2, 3, 4]
    assert [item.prov[0].page_no for item in merged.groups] == [1, 2, 3, 4]
    assert all(item.prov == [] for item in merged.texts)
    assert set(merged.pages) == {1, 2, 3, 4}
    merged.validate_document()
    assert DoclingDocument.model_validate_json(merged.model_dump_json()) == merged
    assert [prov.page_no for prov in doc.body.prov] == [5, 6]
    assert [item.prov[0].page_no for item in doc.groups] == [5, 6]


def test_group_geometry_does_not_change_content_exports_or_chunks():
    doc = _doc(1)
    group = doc.add_group()
    doc.add_heading(text="Heading", parent=group)
    doc.add_text(label=DocItemLabel.TEXT, text="Paragraph one", parent=group)
    doc.add_text(label=DocItemLabel.TEXT, text="Paragraph two", parent=group)
    doc.add_code(text="print('hello')", parent=group)
    markdown = doc.export_to_markdown()
    chunks = list(HierarchicalChunker().chunk(doc))

    group.prov.append(_prov())
    assert doc.export_to_markdown() == markdown
    assert list(HierarchicalChunker().chunk(doc)) == chunks
    assert doc.export_to_markdown(page_no=1) == ""
    assert doc.filter(page_nrs={1}).texts == []
