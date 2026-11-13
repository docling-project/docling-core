# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

"""Regression tests for floating-item footnotes in LaTeX and DocTags exports."""

import pytest
from PIL import Image as PILImage

from docling_core.transforms.serializer.doctags import DocTagsDocSerializer, DocTagsParams
from docling_core.transforms.serializer.latex import LaTeXDocSerializer, LaTeXParams
from docling_core.types.doc import (
    BoundingBox,
    CodeItem,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocTagsDocument,
    FloatingItem,
    ProvenanceItem,
    Size,
    TableData,
)


def _provenance(page_no: int = 1) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=10, t=20, r=30, b=40, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, 0),
    )


def _floating_doc(kind: str) -> tuple[DoclingDocument, FloatingItem]:
    doc = DoclingDocument(name="footnotes")
    for page_no in (1, 2):
        doc.add_page(page_no=page_no, size=Size(width=100, height=100))
    if kind == "table":
        data = TableData(num_rows=0, num_cols=1)
        data.add_row(["Cell content"])
        item = doc.add_table(data=data, prov=_provenance())
    elif kind == "picture":
        item = doc.add_picture(prov=_provenance())
    else:
        item = doc.add_code(text="print('code content')", prov=_provenance())
    return doc, item


@pytest.mark.parametrize("serializer_cls", [LaTeXDocSerializer, DocTagsDocSerializer])
@pytest.mark.parametrize("kind", ["table", "picture", "code"])
@pytest.mark.parametrize("as_children", [False, True])
def test_floating_footnotes_preserve_order_and_sources(serializer_cls, kind, as_children):
    """Referenced notes appear once with their owner, regardless of tree placement."""
    doc, item = _floating_doc(kind)
    parent = item if as_children else None
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Owner caption", parent=parent)
    item.captions.append(caption.get_ref())
    # Reference order can differ from the order of the text items in the body.
    second = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Second note", parent=parent)
    first = doc.add_text(label=DocItemLabel.FOOTNOTE, text="First note", parent=parent)
    item.footnotes.extend([first.get_ref(), second.get_ref()])

    result = serializer_cls(doc=doc).serialize()
    text = result.text
    assert text.count("Owner caption") == 1
    assert text.count("First note") == 1
    assert text.count("Second note") == 1
    assert text.index("Owner caption") < text.index("First note") < text.index("Second note")
    source_refs = {source.self_ref for source in result.get_unique_doc_items()}
    assert {first.self_ref, second.self_ref} <= source_refs

    if serializer_cls is DocTagsDocSerializer:
        tag = {"table": "otsl", "picture": "picture", "code": "code"}[kind]
        assert text.index(f"<{tag}>") < text.index("<footnote>First note</footnote>")
        assert text.index("<footnote>Second note</footnote>") < text.index(f"</{tag}>")
    elif kind == "code":
        assert text.index(r"\end{verbatim}") < text.index("First note")
    else:
        environment = "table" if kind == "table" else "figure"
        assert text.index(r"\begin{" + environment + "}") < text.index("First note")
        assert text.index("Second note") < text.index(r"\end{" + environment + "}")


@pytest.mark.parametrize("serializer_cls", [LaTeXDocSerializer, DocTagsDocSerializer])
def test_table_with_only_footnotes(serializer_cls):
    """Even an empty table retains its associated footnote."""
    doc = DoclingDocument(name="empty table")
    table = doc.add_table(data=TableData(num_rows=0, num_cols=0))
    note = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Empty table note")
    table.footnotes.append(note.get_ref())

    result = serializer_cls(doc=doc).serialize()
    assert result.text.count("Empty table note") == 1
    assert note.self_ref in {source.self_ref for source in result.get_unique_doc_items()}


@pytest.mark.parametrize("kind", ["table", "picture", "code"])
@pytest.mark.parametrize("escape_latex", [False, True])
def test_latex_footnotes_escape_text_outside_verbatim(kind, escape_latex):
    doc, item = _floating_doc(kind)
    raw = r"Cost $5 & 10%_discount #1 {x} \path ~ ^"
    escaped = (
        r"Cost \$5 \& 10\%\_discount \#1 \{x\} \textbackslash{}path "
        r"\textasciitilde{} \textasciicircum{}"
    )
    note = doc.add_text(label=DocItemLabel.FOOTNOTE, text=raw)
    item.footnotes.append(note.get_ref())

    text = LaTeXDocSerializer(doc=doc, params=LaTeXParams(escape_latex=escape_latex)).serialize().text
    expected = escaped if escape_latex else raw
    assert text.count(expected) == 1
    assert r"{\footnotesize " + expected + r"\par}" in text
    if kind == "code":
        assert text.index(r"\end{verbatim}") < text.index(expected)


@pytest.mark.parametrize("serializer_cls", [LaTeXDocSerializer, DocTagsDocSerializer])
@pytest.mark.parametrize("kind", ["table", "picture", "code"])
@pytest.mark.parametrize("filter_by", ["labels", "pages", "layers"])
def test_footnotes_respect_serializer_filters(serializer_cls, kind, filter_by):
    doc, item = _floating_doc(kind)
    included = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Included note", prov=_provenance())
    excluded = doc.add_text(
        label=DocItemLabel.FOOTNOTE,
        text="Excluded note",
        prov=_provenance(page_no=2 if filter_by == "pages" else 1),
        content_layer=ContentLayer.FURNITURE if filter_by == "layers" else ContentLayer.BODY,
    )
    item.footnotes.extend([included.get_ref(), excluded.get_ref()])
    params_cls = LaTeXParams if serializer_cls is LaTeXDocSerializer else DocTagsParams
    if filter_by == "labels":
        params = params_cls(labels=set(DocItemLabel) - {DocItemLabel.FOOTNOTE})
    elif filter_by == "pages":
        params = params_cls(pages={1})
    else:
        params = params_cls(layers={ContentLayer.BODY})

    result = serializer_cls(doc=doc, params=params).serialize()
    source_refs = {source.self_ref for source in result.get_unique_doc_items()}
    assert "Excluded note" not in result.text
    assert excluded.self_ref not in source_refs
    if filter_by == "labels":
        assert "Included note" not in result.text
        assert included.self_ref not in source_refs
    else:
        assert result.text.count("Included note") == 1
        assert included.self_ref in source_refs


@pytest.mark.parametrize("kind", ["table", "picture", "code"])
@pytest.mark.parametrize("add_content,add_location", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("self_closing", [False, True])
def test_doctags_footnotes_respect_content_and_location_options(kind, add_content, add_location, self_closing):
    doc, item = _floating_doc(kind)
    note = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Located note", prov=_provenance())
    item.footnotes.append(note.get_ref())
    params = DocTagsParams(
        add_content=add_content,
        add_location=add_location,
        do_self_closing=self_closing,
        xsize=1000,
        ysize=1000,
    )

    result = DocTagsDocSerializer(doc=doc, params=params).serialize()
    if not add_content and not add_location:
        assert "Located note" not in result.text
        assert "<loc_" not in result.text
        return
    locations = "".join(f"<loc_{coordinate}{'/' if self_closing else ''}>" for coordinate in (100, 200, 300, 400))
    expected = (locations if add_location else "") + ("Located note" if add_content else "")
    assert result.text.count(f"<footnote>{expected}</footnote>") == 1
    assert note.self_ref in {source.self_ref for source in result.get_unique_doc_items()}
    if not add_content:
        assert "Located note" not in result.text


@pytest.mark.parametrize("kind", ["table", "picture", "code"])
def test_doctags_footnotes_are_independent_of_caption_option(kind):
    doc, item = _floating_doc(kind)
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Hidden caption")
    note = doc.add_text(label=DocItemLabel.FOOTNOTE, text="Visible note")
    item.captions.append(caption.get_ref())
    item.footnotes.append(note.get_ref())

    text = DocTagsDocSerializer(doc=doc, params=DocTagsParams(add_caption=False)).serialize().text
    assert "Hidden caption" not in text
    assert text.count("<footnote>Visible note</footnote>") == 1


@pytest.mark.parametrize("kind", ["table", "picture", "code"])
@pytest.mark.parametrize("with_image", [False, True])
def test_doctags_footnotes_roundtrip_with_owner(kind, with_image):
    """Nested footnotes reload as references instead of becoming owner content."""
    doc, item = _floating_doc(kind)
    note_texts = ["First note\ncontinued", "Second note"]
    note_boxes = [
        BoundingBox(l=40, t=50, r=60, b=70, coord_origin=CoordOrigin.TOPLEFT),
        BoundingBox(l=70, t=80, r=90, b=90, coord_origin=CoordOrigin.TOPLEFT),
    ]
    for text, bbox in zip(note_texts, note_boxes):
        note = doc.add_text(
            label=DocItemLabel.FOOTNOTE,
            text=text,
            prov=ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text))),
        )
        item.footnotes.append(note.get_ref())
    doctags = DocTagsDocSerializer(doc=doc).serialize().text
    images = [PILImage.new("RGB", (100, 100))] if with_image else None
    restored = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([doctags], images))

    if kind == "table":
        restored_item = restored.tables[0]
        assert [cell.text for cell in restored_item.data.table_cells] == ["Cell content"]
    elif kind == "picture":
        restored_item = restored.pictures[0]
    else:
        restored_item = next(text for text in restored.texts if isinstance(text, CodeItem))
        assert restored_item.text == "print('code content')"

    restored_notes = [ref.resolve(restored) for ref in restored_item.footnotes]
    assert [note.text for note in restored_notes] == note_texts
    assert [note.label for note in restored_notes] == [DocItemLabel.FOOTNOTE] * 2
    assert len([text for text in restored.texts if text.label == DocItemLabel.FOOTNOTE]) == 2
    scale = 1 if with_image else 0.01
    assert restored_item.prov[0].bbox == _provenance().bbox.resize_by_scale(scale, scale)
    for note, bbox in zip(restored_notes, note_boxes):
        assert len(note.prov) == 1
        assert note.prov[0].page_no == 1
        assert note.prov[0].charspan == (0, len(note.text))
        expected_bbox = bbox.resize_by_scale(scale, scale)
        assert note.prov[0].bbox.as_tuple() == pytest.approx(expected_bbox.as_tuple())
        assert note.prov[0].bbox.coord_origin == expected_bbox.coord_origin

    reserialized = DocTagsDocSerializer(doc=restored).serialize().text
    for text in note_texts:
        assert reserialized.count(text) == 1


@pytest.mark.parametrize("kind", ["table", "picture", "code"])
@pytest.mark.parametrize("located_owner", [False, True])
def test_doctags_footnote_and_owner_locations_are_independent(kind, located_owner):
    doc, item = _floating_doc(kind)
    if not located_owner:
        item.prov.clear()
    note = doc.add_text(
        label=DocItemLabel.FOOTNOTE,
        text="Independent location",
        prov=None if located_owner else _provenance(),
    )
    item.footnotes.append(note.get_ref())
    doctags = DocTagsDocSerializer(doc=doc).serialize().text
    restored = DoclingDocument.load_from_doctags(
        DocTagsDocument.from_doctags_and_image_pairs([doctags], [PILImage.new("RGB", (100, 100))])
    )

    if kind == "table":
        restored_item = restored.tables[0]
    elif kind == "picture":
        restored_item = restored.pictures[0]
    else:
        restored_item = next(text for text in restored.texts if isinstance(text, CodeItem))
    restored_note = restored_item.footnotes[0].resolve(restored)
    assert bool(restored_item.prov) == located_owner
    assert bool(restored_note.prov) != located_owner
    located_item = restored_item if located_owner else restored_note
    assert located_item.prov[0].bbox == _provenance().bbox
