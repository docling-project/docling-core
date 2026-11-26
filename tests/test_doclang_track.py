"""Tests for tracks: ``TrackItem`` and its DocLang ``<track>`` serialization."""

import warnings
from pathlib import Path

import pytest
from pydantic import AnyUrl

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer._doclang_utils import DocLangVocabulary
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    GroupLabel,
    ImageRef,
    PictureItem,
    ProvenanceItem,
    Size,
    TextItem,
    TrackItem,
    TrackSource,
)
from docling_core.types.doc.base import ImageRefMode
from tests.doclang_validation import validate_dclg_xml
from tests.test_serialization_doclang import verify_doclang

DATA = Path("./tests/data/doc")


def _image(name: str) -> ImageRef:
    return ImageRef(
        mimetype="image/png", dpi=72, size=Size(width=16, height=9), uri=AnyUrl(f"https://example.com/{name}.png")
    )


def _slide_with_video() -> DoclingDocument:
    """A slide with a title and a video: its poster frame, a sampled frame and the transcript."""
    doc = DoclingDocument(name="slide_with_video")
    doc.add_page(page_no=1, size=Size(width=960, height=540))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Quarterly review",
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=50, t=30, r=550, b=70), charspan=(0, 16)),
    )
    track = doc.add_track(
        image=_image("poster"),
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=50, t=100, r=370, b=280), charspan=(0, 0)),
    )
    doc.add_picture(parent=track, image=_image("frame_0"), source=TrackSource(start_time=0.0, end_time=0.0))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Welcome to the quarterly review.",
        parent=track,
        source=TrackSource(start_time=0.0, end_time=2.12, voice="Alice"),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Thanks, Alice & team.",
        parent=track,
        source=TrackSource(start_time=0.0, end_time=2.12, voice="Bob"),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=" Revenue grew by 12% this quarter.",
        parent=track,
        source=TrackSource(start_time=3.1, end_time=5.36),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Costs stayed flat.",
        parent=track,
        source=TrackSource(start_time=3.1, end_time=5.36),
    )
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Thank you.",
        parent=track,
        source=TrackSource(start_time=3725.5, end_time=3726.0),
    )
    return doc


def _cues(doc: DoclingDocument, track: TrackItem) -> list[tuple[str, str, float, float, str | None]]:
    cues = []
    for ref in track.children:
        item = ref.resolve(doc)
        source = item.source[0]
        text = item.text if isinstance(item, TextItem) else ""
        cues.append((type(item).__name__, text, source.start_time, source.end_time, source.voice))
    return cues


def test_track_serialization():
    """A track is a ``<track>`` with its location, cover and one cue block per time.

    Turns with the same times share a block, a frame joins the block that starts at its
    time and comes first in it, a voice is written before its turn, an empty voice ends
    the previous unattributed turn, and an instant is written without end time.
    """
    doc = _slide_with_video()
    actual = doc.export_to_doclang(image_mode=ImageRefMode.REFERENCED)
    verify_doclang(exp_file=DATA / "track.gt.dclg.xml", actual=actual)


@pytest.mark.parametrize("pretty_indentation", ["  ", None])
def test_track_round_trip(pretty_indentation: str | None):
    """Deserializing the serialized track gives back the same cues, and the same DocLang."""
    doc = _slide_with_video()
    params = DocLangParams(pretty_indentation=pretty_indentation)
    xml = DocLangDocSerializer(doc=doc, params=params).serialize().text

    back = DocLangDocDeserializer().deserialize_str(xml)

    assert len(back.tracks) == 1
    track = back.tracks[0]
    assert track.prov
    assert _cues(back, track) == _cues(doc, doc.tracks[0])
    assert DocLangDocSerializer(doc=back, params=params).serialize().text == xml


def test_track_reads_doclang_cue_blocks():
    """Timestamp runs, instants, voices and an empty voice, as in the DocLang spec examples."""
    xml = (
        "<doclang>"
        "<track>"
        "<bdiv/><seconds value='2'/>Hello, and welcome."
        "<bdiv/><minutes value='0'/><seconds value='2'/><msecs value='500'/><minutes value='0'/><seconds value='7'/>"
        "<voice>John</voice>Hi Mary!<voice><italic>Mary</italic></voice><bold>Hello</bold> John!"
        "<bdiv/><minutes value='59'/><seconds value='58'/><msecs value='450'/>"
        "<hours value='1'/><minutes value='0'/><seconds value='1'/><msecs value='210'/>"
        "<voice>Narrator</voice>The results.<voice/>laughter"
        "</track>"
        "</doclang>"
    )
    doc = DocLangDocDeserializer().deserialize_str(xml)

    assert _cues(doc, doc.tracks[0]) == [
        ("TextItem", "Hello, and welcome.", 2.0, 2.0, None),
        ("TextItem", "Hi Mary!", 2.5, 7.0, "John"),
        ("TextItem", "Hello John!", 2.5, 7.0, "Mary"),
        ("TextItem", "The results.", 3598.45, 3601.21, "Narrator"),
        ("TextItem", "laughter", 3598.45, 3601.21, None),
    ]


def test_track_skips_chapter_and_audio_with_a_warning():
    """``<chapter>`` and per-cue ``<audio>`` have no counterpart yet: skipped, the cue is kept."""
    xml = (
        "<doclang><track>"
        "<bdiv/><seconds value='0'/><seconds value='5'/><chapter>Introduction</chapter>"
        "<frame/><audio/>Welcome."
        "</track></doclang>"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        doc = DocLangDocDeserializer().deserialize_str(xml)

    assert sorted(str(w.message) for w in caught) == [
        "<audio> in <track> is not supported yet; skipped",
        "<chapter> in <track> is not supported yet; skipped",
    ]
    assert _cues(doc, doc.tracks[0]) == [
        ("PictureItem", "", 0.0, 0.0, None),
        ("TextItem", "Welcome.", 0.0, 5.0, None),
    ]


def test_track_child_without_track_source_is_not_serialized():
    """A child of a track that is not a cue cannot be written in ``<track>``: it is skipped with a warning."""
    doc = DoclingDocument(name="doc")
    track = doc.add_track()
    doc.add_text(label=DocItemLabel.TEXT, text="no time", parent=track)
    doc.add_text(label=DocItemLabel.TEXT, text="cue", parent=track, source=TrackSource(start_time=1, end_time=2))

    with pytest.warns(UserWarning, match="is not a text or picture with a track source"):
        xml = DocLangDocSerializer(doc=doc, params=DocLangParams(pretty_indentation=None)).serialize().text

    assert "no time" not in xml
    assert '<track><bdiv/><seconds value="1"/><seconds value="2"/>cue</track>' in xml


def test_track_other_formats_show_the_cues():
    """Markdown, text and HTML have no track element: they show the cues in order."""
    doc = _slide_with_video()

    markdown = doc.export_to_markdown()
    assert "missing-text" not in markdown
    # the cover, then the frame and the transcript
    assert markdown.startswith(
        "Quarterly review\n\n<!-- image -->\n\n<!-- image -->\n\nWelcome to the quarterly review."
    )
    assert "Unhandled item type" not in doc.export_to_html()


def test_track_cover_is_shown_like_a_picture():
    """Markdown and HTML show the cover of a track the way they show the image of a picture."""
    doc = DoclingDocument(name="slide")
    doc.add_track(image=_image("poster"))  # e.g. a slide video that is not transcribed
    doc.add_track()  # e.g. a recording: no cover, nothing to show

    assert doc.export_to_markdown() == "<!-- image -->"
    assert doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED) == "![Image](https://example.com/poster.png)"
    assert '<figure>\n<img src="https://example.com/poster.png">\n\n</figure>' in doc.export_to_html(
        image_mode=ImageRefMode.REFERENCED
    )
    assert "<figure>" not in doc.export_to_html()  # no image in placeholder mode, as for a picture


def test_add_and_insert_track():
    """Tracks get their own list and references, like the other item types."""
    doc = DoclingDocument(name="doc")
    first = doc.add_text(label=DocItemLabel.TEXT, text="first")
    appended = doc.add_track()
    inserted = doc.insert_track(sibling=first, after=True)

    assert [t.self_ref for t in doc.tracks] == ["#/tracks/0", "#/tracks/1"]
    assert [ref.cref for ref in doc.body.children] == ["#/texts/0", inserted.self_ref, appended.self_ref]
    assert isinstance(doc.tracks[0], TrackItem) and doc.tracks[0].label == DocItemLabel.TRACK

    restored = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert [type(t) for t in restored.tracks] == [TrackItem, TrackItem]

    doc.delete_items(node_items=[appended])
    assert [ref.cref for ref in doc.body.children] == ["#/texts/0", "#/tracks/0"]


def test_temporal_special_tokens_follow_the_doclang_vocabulary():
    """The timestamp tokens are those of DocLang 0.8, with the values of its token vocabulary."""
    tokens = DocLangVocabulary.get_special_tokens()

    assert '<hours value="23"/>' in tokens
    assert '<hours value="24"/>' not in tokens
    assert '<msecs value="990"/>' in tokens
    assert '<msecs value="5"/>' not in tokens
    assert "<bdiv/>" in tokens
    assert not any("centisecond" in token for token in tokens)
    assert '<msecs value="10"/>' not in DocLangVocabulary.get_special_tokens(include_temporal_tokens=False)


def test_picture_children_of_a_track_are_frames():
    """A sampled frame is a picture cue: it keeps its image as the ``<frame>`` source."""
    doc = _slide_with_video()
    frame = doc.tracks[0].children[0].resolve(doc)
    assert isinstance(frame, PictureItem)

    params = DocLangParams(image_mode=ImageRefMode.REFERENCED, pretty_indentation=None)
    xml = DocLangDocSerializer(doc=doc, params=params).serialize().text
    assert '<frame><src uri="https://example.com/frame_0.png"/></frame>' in xml
    assert '<cover><src uri="https://example.com/poster.png"/></cover>' in xml


def test_frame_joins_the_block_that_starts_at_its_time():
    """A frame is the still at its start time: its own end time does not split the block."""
    doc = DoclingDocument(name="doc")
    track = doc.add_track()
    doc.add_text(label=DocItemLabel.TEXT, text="Hello", parent=track, source=TrackSource(start_time=1, end_time=4))
    doc.add_picture(parent=track, source=TrackSource(start_time=1, end_time=1.001))  # e.g. a sampled video frame
    doc.add_picture(parent=track, source=TrackSource(start_time=2, end_time=2))

    xml = DocLangDocSerializer(doc=doc, params=DocLangParams(pretty_indentation=None)).serialize().text

    assert (
        '<track><bdiv/><seconds value="1"/><seconds value="4"/><frame/>Hello<bdiv/><seconds value="2"/><frame/></track>'
    ) in xml


def test_turns_without_text_are_kept():
    """A turn is kept even without text: of an empty block, a voice alone, or an empty voice."""
    xml = (
        "<doclang><track>"
        "<bdiv/><seconds value='1'/>"
        "<bdiv/><seconds value='2'/><voice>Host</voice>"
        "<bdiv/><seconds value='3'/>Hi<voice/>"
        "</track></doclang>"
    )
    doc = DocLangDocDeserializer().deserialize_str(xml)

    assert _cues(doc, doc.tracks[0]) == [
        ("TextItem", "", 1.0, 1.0, None),
        ("TextItem", "", 2.0, 2.0, "Host"),
        ("TextItem", "Hi", 3.0, 3.0, None),
        ("TextItem", "", 3.0, 3.0, None),
    ]


def _compact(doc: DoclingDocument) -> str:
    return DocLangDocSerializer(doc=doc, params=DocLangParams(pretty_indentation=None)).serialize().text


def _timed(doc: DoclingDocument, text: str, start: float, end: float, voice: str | None = None, **kwargs) -> TextItem:
    source = TrackSource(start_time=start, end_time=end, voice=voice)
    return doc.add_text(label=DocItemLabel.TEXT, text=text, source=source, **kwargs)


def test_loose_timed_items_form_implicit_tracks():
    """Consecutive timed items outside a track are a track, and any other item ends it."""
    doc = DoclingDocument(name="doc")
    doc.add_text(label=DocItemLabel.TEXT, text="Intro")
    _timed(doc, "A", 1, 2, voice="Ann")
    _timed(doc, "B", 2, 3)
    doc.add_text(label=DocItemLabel.TEXT, text="Break", content_layer=ContentLayer.FURNITURE)
    _timed(doc, "C", 1, 2)

    assert _compact(doc) == (
        '<doclang version="0.10"><text>Intro</text>'
        '<track><bdiv/><seconds value="1"/><seconds value="2"/><voice>Ann</voice>A'
        '<bdiv/><seconds value="2"/><seconds value="3"/>B</track>'
        '<text><layer value="furniture"/>Break</text>'
        '<track><bdiv/><seconds value="1"/><seconds value="2"/>C</track></doclang>'
    )


def test_implicit_track_ends_at_its_group():
    """A track cannot cross an element boundary: timed items in and after a group are two tracks."""
    doc = DoclingDocument(name="doc")
    section = doc.add_group(label=GroupLabel.SECTION)
    _timed(doc, "A", 1, 2, parent=section)
    _timed(doc, "B", 3, 4)

    assert _compact(doc) == (
        '<doclang version="0.10"><track><bdiv/><seconds value="1"/><seconds value="2"/>A</track>'
        '<track><bdiv/><seconds value="3"/><seconds value="4"/>B</track></doclang>'
    )


def test_implicit_track_round_trips_into_a_track_item():
    """Read back, an implicit track is an explicit TrackItem with the same cues."""
    doc = DoclingDocument(name="doc")
    _timed(doc, "Hello", 0, 1.5, voice="Ann")
    doc.add_picture(source=TrackSource(start_time=2, end_time=2))
    _timed(doc, "World", 2, 3.25)

    back = DocLangDocDeserializer().deserialize_str(_compact(doc))

    assert _cues(back, back.tracks[0]) == [
        ("TextItem", "Hello", 0.0, 1.5, "Ann"),
        ("PictureItem", "", 2.0, 2.0, None),
        ("TextItem", "World", 2.0, 3.25, None),
    ]


def test_inline_runs_of_a_cue_are_one_turn():
    """Inline runs with the same time and voice are one turn, and a run with timed runs does not end a track."""
    doc = DoclingDocument(name="doc")
    _timed(doc, "First", 0, 1)
    cue = doc.add_inline_group()
    _timed(doc, "Hee! ", 1, 2, voice="Esme", parent=cue)
    _timed(doc, "laughter", 1, 2, parent=cue, formatting=Formatting(italic=True))
    turn = doc.add_text(label=DocItemLabel.TEXT, text="", source=TrackSource(start_time=2, end_time=3, voice="Ann"))
    runs = doc.add_inline_group(parent=turn)
    doc.add_text(label=DocItemLabel.TEXT, text="Formatted ", parent=runs)
    doc.add_text(label=DocItemLabel.TEXT, text="turn", parent=runs, formatting=Formatting(bold=True))

    assert _compact(doc) == (
        '<doclang version="0.10"><track><bdiv/><seconds value="0"/><seconds value="1"/>First'
        '<bdiv/><seconds value="1"/><seconds value="2"/><voice>Esme</voice><content>Hee! </content><voice/>laughter'
        '<bdiv/><seconds value="2"/><seconds value="3"/><voice>Ann</voice>Formatted turn</track></doclang>'
    )


@pytest.mark.parametrize("example", [1, 2, 3, 4, 5])
def test_webvtt_documents_export_to_one_valid_track(example: int):
    """Documents converted from WebVTT keep their times and voices in DocLang."""
    doc = DoclingDocument.load_from_json(DATA / f"webvtt_example_{example:02d}.json")
    xml = _compact(doc)

    validate_dclg_xml(xml)
    assert xml.count("<track>") == 1
    assert xml.count("<bdiv/>") == len({(s.start_time, s.end_time) for t in doc.texts for s in t.source})


def _track_with_cue(doc: DoclingDocument, **track_kwargs) -> TrackItem:
    track = doc.add_track(**track_kwargs)
    doc.add_text(
        label=DocItemLabel.TEXT, text="cue", parent=track, source=TrackSource(start_time=1, end_time=2, voice="A")
    )
    return track


def test_cues_take_the_layer_of_their_track():
    doc = DocLangDocDeserializer().deserialize_str(
        '<doclang version="0.10"><track><layer value="furniture"/><bdiv/><seconds value="0"/>hi</track></doclang>'
    )
    assert doc.tracks[0].content_layer == ContentLayer.FURNITURE
    assert doc.texts[0].content_layer == ContentLayer.FURNITURE
    assert doc.export_to_markdown() == ""


@pytest.mark.parametrize("value", ["x", "-5", "1.5"])
def test_invalid_timestamp_skips_only_its_cue_block(value):
    with pytest.warns(UserWarning, match="invalid <seconds> value"):
        doc = DocLangDocDeserializer().deserialize_str(
            f'<doclang version="0.10"><track><bdiv/><seconds value="{value}"/>bad'
            '<bdiv/><seconds value="2"/>good</track></doclang>'
        )
    assert [text.text for text in doc.texts] == ["good"]


def test_voice_is_not_written_without_its_text():
    doc = DoclingDocument(name="doc")
    _track_with_cue(doc)
    text = DocLangDocSerializer(doc=doc, params=DocLangParams(content_types=set())).serialize().text
    assert "<voice>" not in text
    assert "<seconds" not in text


def test_page_filter_keeps_the_cues_of_a_track_on_the_page():
    doc = DoclingDocument(name="doc")
    for page_no in (1, 2):
        doc.add_page(page_no=page_no, size=Size(width=10, height=10))
    _track_with_cue(doc, prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 0)))
    assert doc.export_to_markdown(page_no=1) == "cue"
    assert doc.export_to_markdown(page_no=2) == ""
    assert [item.text for item, _ in doc.iterate_items(page_no=1) if isinstance(item, TextItem)] == ["cue"]
    assert "cue" in DocLangDocSerializer(doc=doc, params=DocLangParams(pages={1})).serialize().text


def test_track_captions_in_markdown_html_and_latex():
    from docling_core.transforms.serializer.latex import LaTeXDocSerializer

    doc = DoclingDocument(name="doc")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Talk title")
    _track_with_cue(doc, caption=caption)
    assert doc.export_to_markdown() == "Talk title\n\ncue"
    html = doc.export_to_html()
    assert html.index("<figure>") < html.index("Talk title") < html.index("cue") < html.index("</figure>")
    assert "Talk title\n\ncue" in LaTeXDocSerializer(doc=doc).serialize().text
