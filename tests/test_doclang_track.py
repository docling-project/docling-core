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
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    PictureItem,
    ProvenanceItem,
    Size,
    TextItem,
    TrackItem,
    TrackSource,
)
from docling_core.types.doc.base import ImageRefMode
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
    doc.add_picture(parent=track, image=_image("frame_0"), source=TrackSource(start_time=0.0, end_time=0.001))
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

    Cues with the same times share a block, the frame comes first, a voice is written
    before its turn, an empty voice ends the previous unattributed turn, and a point
    cue (1 ms) is written without end time.
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
    """Timestamp runs, point cues, voices and an empty voice, as in the DocLang spec examples."""
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
        ("TextItem", "Hello, and welcome.", 2.0, 2.001, None),
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
        ("PictureItem", "", 0.0, 5.0, None),
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
    assert markdown.startswith("Quarterly review\n\n<!-- image -->\n\nWelcome to the quarterly review.")
    assert "Unhandled item type" not in doc.export_to_html()


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
