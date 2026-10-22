# SPDX-License-Identifier: MIT

"""Exercise the example's metadata handling, not PDF parser accuracy."""

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from examples.serialization_with_provenance import export_markdown_parts


def _document() -> DoclingDocument:
    doc = DoclingDocument(name="自制中文样本")
    for page_no in (1, 2):
        doc.add_page(page_no=page_no, size=Size(width=595, height=842))
    return doc


def _provenance(page_no: int, start: int, end: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=50, t=700, r=500, b=680, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(start, end),
    )


def test_repeated_text_keeps_distinct_source_items() -> None:
    doc = _document()
    text = "测试金额50,000.00元。"
    items = [doc.add_text(label=DocItemLabel.TEXT, text=text, prov=_provenance(page, 0, len(text))) for page in (1, 2)]
    inline = doc.add_inline_group()
    inline_items = [
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=inline, prov=_provenance(2, 0, len(text)))
        for _ in range(2)
    ]

    parts = export_markdown_parts(doc)

    assert len(parts) == 3
    assert [part["text"] for part in parts[:2]] == [text, text]
    assert [part["items"][0]["ref"] for part in parts[:2]] == [item.self_ref for item in items]
    assert items[0].self_ref != items[1].self_ref
    assert [part["items"][0]["provenance"][0]["page_no"] for part in parts[:2]] == [1, 2]
    # An inline part combines two distinct source items, even with equal text.
    assert parts[2]["text"].count(text) == 2
    assert [item["ref"] for item in parts[2]["items"]] == [item.self_ref for item in inline_items]


def test_one_source_item_keeps_every_page_region() -> None:
    doc = _document()
    first, second = "前页说明。", "次页续行。"
    text = first + second
    item = doc.add_text(label=DocItemLabel.TEXT, text=text, prov=_provenance(1, 0, len(first)))
    second_prov = _provenance(2, len(first), len(text))
    second_prov.bbox = BoundingBox(l=80, t=100, r=480, b=124, coord_origin=CoordOrigin.TOPLEFT)
    item.prov.append(second_prov)

    parts = export_markdown_parts(doc)

    assert len(parts) == 1
    assert parts[0]["text"] == text
    assert len(parts[0]["items"]) == 1
    assert parts[0]["items"][0]["ref"] == item.self_ref
    provenance = parts[0]["items"][0]["provenance"]
    assert [prov["page_no"] for prov in provenance] == [1, 2]
    assert [prov["charspan"] for prov in provenance] == [[0, len(first)], [len(first), len(text)]]
    assert [prov["bbox"] for prov in provenance] == [prov.bbox.model_dump(mode="json") for prov in item.prov]
    assert provenance[0]["bbox"]["coord_origin"] == "BOTTOMLEFT"
    assert provenance[1]["bbox"] == {"l": 80, "t": 100, "r": 480, "b": 124, "coord_origin": "TOPLEFT"}


def test_explicit_layers_keep_footer_with_its_source() -> None:
    doc = _document()
    text = "第 21 页"
    footer = doc.add_text(
        label=DocItemLabel.PAGE_FOOTER,
        text=text,
        content_layer=ContentLayer.FURNITURE,
        prov=_provenance(2, 0, len(text)),
    )

    parts = export_markdown_parts(doc)

    assert len(parts) == 1
    assert parts[0]["text"] == text
    assert parts[0]["items"][0]["ref"] == footer.self_ref
    assert parts[0]["items"][0]["provenance"][0]["page_no"] == 2
