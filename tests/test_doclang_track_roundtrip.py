"""DocLang round trip of ``<track>``: DocLang -> ``DoclingDocument`` -> DocLang.

The fixtures in ``tests/data/doc/doclang_track`` are the track examples of the DocLang 0.9.1
reference repository. Each case is compared by what it means, not byte by byte: see
``_canonical``. A known gap is an ``xfail`` with its reason, and a strict one, so that
closing a gap has to remove its marker.
"""

import itertools
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.types.doc import DocItemLabel, DoclingDocument, ImageRef, TrackSource
from docling_core.types.doc.base import ImageRefMode
from tests.doclang_validation import validate_dclg_xml

FIXTURES = Path("./tests/data/doc/doclang_track")

_TIME_UNITS = {"hours": 3_600_000, "minutes": 60_000, "seconds": 1000, "msecs": 1}
_LITERAL_SPACE = "\ue000"  # a space inside <content>, which edge trimming must not touch
_ELEMENT_HEAD = {"label", "location", "caption", "description", "summary", "href", "layer", "custom", "thread", "xref"}


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _spaces(text: str | None) -> str:
    return re.sub(r"[ \t\r\n]+", " ", text or "")


def _inner(el: ET.Element) -> str:
    """Render the body of an element: inline markup kept, whitespace as DocLang reads it."""
    children = list(el)
    # whitespace next to an element head is not text
    parts = [
        _spaces(el.text)
        if el.text and (el.text.strip() or not children or _tag(children[0]) not in _ELEMENT_HEAD)
        else ""
    ]
    for child in children:
        parts.append(_markup(child))
        tail = child.tail or ""
        parts.append("" if _tag(child) in _ELEMENT_HEAD and not tail.strip() else _spaces(tail))
    return "".join(parts)


def _markup(el: ET.Element) -> str:
    """Render an element. ``<content>`` is only its literal text: the tag does not change the reading."""
    if _tag(el) == "content":
        return (el.text or "").replace(" ", _LITERAL_SPACE)
    attrs = "".join(f' {k}="{v}"' for k, v in sorted(el.attrib.items()))
    return f"<{_tag(el)}{attrs}>{_inner(el)}</{_tag(el)}>"


def _text_block(parts: list[str]) -> str:
    """Join the pieces of a text block and drop whitespace at its edges, outside ``<content>``."""
    return re.sub(r"^ +| +$", "", "".join(parts)).replace(_LITERAL_SPACE, " ")


def _track_lines(track: ET.Element) -> list[str]:
    """Describe a track as one line per primitive, in a form that does not depend on its spelling."""
    lines: list[str] = []
    turn: list[str] = []
    runs: list[int] = []
    run: dict[str, int] = {}
    in_block = False
    has_times = False
    voice: str | None = None
    voice_used = True

    def end_run() -> None:
        nonlocal run
        if run:
            runs.append(sum(_TIME_UNITS[k] * v for k, v in run.items()))
            run = {}

    def end_times() -> None:
        nonlocal has_times, runs
        end_run()
        if has_times:
            return
        start = runs[0]
        end = runs[1] if len(runs) > 1 and runs[1] != start else None
        lines.append(f"block {start}" if end is None else f"block {start}-{end}")
        has_times = True
        runs = []

    def end_turn() -> None:
        nonlocal turn, voice_used
        if text := _text_block(turn):
            lines.append(f"turn [{voice or ''}] {text}")
            voice_used = True
        turn = []

    def end_voice() -> None:
        nonlocal voice, voice_used
        if voice and not voice_used:
            lines.append(f"voice {voice}")  # a voice without a turn
        voice, voice_used = None, True

    def add_tail(el: ET.Element) -> None:
        if in_block and _spaces(el.tail).strip():
            end_times()
        if has_times:
            turn.append(_spaces(el.tail))

    for el in track:
        name = _tag(el)
        if name == "bdiv":
            if in_block:
                end_times()
                end_turn()
                end_voice()
            in_block, has_times = True, False
        elif name in _TIME_UNITS:
            if "seconds" in run and name != "msecs":
                end_run()
            run[name] = int(el.get("value", "0"))
        elif in_block:
            end_times()
            if name == "voice":
                end_turn()
                end_voice()
                voice, voice_used = _text_block([_inner(el)]) or None, False
            elif name in {"chapter", "frame", "audio"}:
                end_turn()
                lines.append(f"{name} {_text_block([_inner(el)])}")
            else:
                turn.append(_markup(el))
        else:
            lines.append(f"{'head' if name in _ELEMENT_HEAD else name} {name} {sorted(el.attrib.items())} {_inner(el)}")
        add_tail(el)
    if in_block:
        end_times()
        end_turn()
        end_voice()
    return _merge_same_time_blocks(lines)


def _merge_same_time_blocks(lines: list[str]) -> list[str]:
    """Merge a block into the previous one when both have the same times and it holds only turns.

    ``DoclingDocument`` has no cue block, so this is the canonical form a round trip gives.
    """
    merged: list[str] = []
    header: str | None = None
    for i, line in enumerate(lines):
        if line.startswith("block "):
            body = list(itertools.takewhile(lambda next_line: not next_line.startswith("block "), lines[i + 1 :]))
            if line == header and all(next_line.startswith(("turn ", "voice ")) for next_line in body):
                continue
            header = line
        merged.append(line)
    return merged


def _canonical(xml: str) -> list[str]:
    """Describe a DocLang document by its meaning.

    Comments, CDATA, zero time units, an end time equal to the start time and indentation
    all read the same in DocLang, so they do not count as differences. Nor does splitting
    turns with the same times over several blocks.
    """

    def walk(el: ET.Element) -> list[str]:
        lines: list[str] = []
        for child in el:
            name = _tag(child)
            if name == "track":
                lines += ["track {", *_track_lines(child), "}"]
            else:
                lines += [f"{name} {{", *walk(child)]
                if text := _text_block([_spaces(child.text)]):
                    lines.append(f"text {text}")
                lines.append("}")
        return lines

    return walk(ET.fromstring(xml))


def _round_trip(xml: str, *, indentation: str | None = "  ") -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # what a skipped element warns about is not under test here
        doc = DocLangDocDeserializer().deserialize_str(xml)
    params = DocLangParams(pretty_indentation=indentation, image_mode=ImageRefMode.REFERENCED)
    return DocLangDocSerializer(doc=doc, params=params).serialize().text


def _check(xml: str, indentation: str | None) -> None:
    out = _round_trip(xml, indentation=indentation)
    assert _canonical(out) == _canonical(xml)
    validate_dclg_xml(out)
    assert _round_trip(out, indentation=indentation) == out


def _doc(body: str) -> str:
    return f'<doclang version="0.9">{body}</doclang>'


def _track(body: str) -> str:
    return _doc(f"<track>{body}</track>")


def _sec(value: int, msecs: int | None = None) -> str:
    return f'<seconds value="{value}"/>' + (f'<msecs value="{msecs}"/>' if msecs is not None else "")


_LOC = '<location value="1"/><location value="2"/><location value="30"/><location value="40"/>'

FIXTURE_NAMES = sorted(path.stem for path in FIXTURES.glob("*.dclg"))

# Known gaps, most important first.
_FRAME_ONLY_END = "the end time of a block that holds only a frame has no meaning in DocLang 0.9"
_BODY_TEXT = "not specific to tracks: raw text in the body of a picture is dropped too"
_COVER = "a cover without a restored image, and the head of a cover, are dropped"
_LABEL = "the label of a track is dropped"
_CHAPTER = "chapter has no counterpart in DoclingDocument"
_AUDIO = "per-cue audio has no counterpart in DoclingDocument"
_GENERIC_HEAD = "not specific to tracks: href and a lone thread are dropped on other elements too"
_GROUP = "not specific to tracks: the group wrapper is dropped around other elements too"
_RICH = "turns and voices keep their text but not their formatting or head"
_SRC = "not specific to tracks: a <src> is only restored from an archive, also on pictures"

FIXTURE_GAPS: dict[str, str] = {
    "ok_track_frame_point": _SRC,
    "reference_track": _SRC,
    "ok_track_voices": _RICH,
    "ok_track_inline_formatting": _RICH,
    "ok_track_element_head": _LABEL,
    "ok_track_cover": _LABEL,
    "ok_track_chapters": _CHAPTER,
    "ok_track_chapter_boundary_mid_cue": _CHAPTER,
    "ok_track_chapter_element_head": _CHAPTER,
    "ok_track_frame_audio": _AUDIO,
    "ok_track_audio_seconds_only_runs": _AUDIO,
    "ok_track_in_group_and_cell": _GROUP,
}

# Ladder of hand-written cases, from the simplest to the most advanced.
CASES: dict[str, tuple[str, str | None]] = {
    # 1. Core timing
    "start_only": (_track(f"<bdiv/>{_sec(2)}Hello"), None),
    "start_and_end": (_track(f"<bdiv/>{_sec(2)}{_sec(7)}Hello"), None),
    "msecs": (_track(f"<bdiv/>{_sec(2, 5)}{_sec(7, 999)}Hello"), None),
    "zero_units": (
        _track(f'<bdiv/><hours value="0"/><minutes value="0"/>{_sec(2, 0)}<minutes value="0"/>{_sec(3)}Hello'),
        None,
    ),
    "over_a_day": (_track(f'<bdiv/><hours value="100"/>{_sec(0)}Hello'), None),
    "end_equals_start": (_track(f"<bdiv/>{_sec(5)}{_sec(5)}Hello"), None),
    "one_msec_interval": (_track(f"<bdiv/>{_sec(5)}{_sec(5, 1)}Hello"), None),
    "two_msec_interval": (_track(f"<bdiv/>{_sec(5)}{_sec(5, 2)}Hello"), None),
    "several_blocks": (_track(f"<bdiv/>{_sec(1)}{_sec(2)}A<bdiv/>{_sec(3)}B<bdiv/>{_sec(4)}{_sec(5)}C"), None),
    "overlapping_blocks": (_track(f"<bdiv/>{_sec(1)}{_sec(5)}A<bdiv/>{_sec(2)}{_sec(3)}B"), None),
    "tied_start_longer_first": (_track(f"<bdiv/>{_sec(1)}{_sec(9)}A<bdiv/>{_sec(1)}{_sec(2)}B"), None),
    "tied_start_shorter_first": (_track(f"<bdiv/>{_sec(1)}{_sec(2)}A<bdiv/>{_sec(1)}{_sec(9)}B"), None),
    "comment_between_times": (_track(f"<bdiv/>{_sec(1)}<!-- end -->{_sec(9)}Hello"), None),
    "comment_inside_a_time": (
        _track(f'<bdiv/><minutes value="1"/><!-- c -->{_sec(2)}<minutes value="1"/>{_sec(9)}Hello'),
        None,
    ),
    "comment_in_turn": (_track(f"<bdiv/>{_sec(1)}Hel<!-- c -->lo"), None),
    # 2. Cue blocks with little or nothing in them
    "same_time_blocks": (_track(f"<bdiv/>{_sec(2)}A<bdiv/>{_sec(2)}B"), None),
    "same_interval_blocks": (_track(f"<bdiv/>{_sec(2)}{_sec(3)}A<bdiv/>{_sec(2)}{_sec(3)}B"), None),
    "empty_block": (_track(f"<bdiv/>{_sec(2)}<bdiv/>{_sec(4)}B"), None),
    "only_empty_block": (_track(f"<bdiv/>{_sec(2)}"), None),
    "voice_only_block": (_track(f"<bdiv/>{_sec(2)}<voice>Host</voice>"), None),
    "empty_track": (_track(""), None),
    # 3. Turns
    "voice": (_track(f"<bdiv/>{_sec(1)}<voice>Ann</voice>Hello"), None),
    "two_voices": (_track(f"<bdiv/>{_sec(1)}<voice>Ann</voice>Hi<voice>Bob</voice>Hello"), None),
    "same_voice_twice": (_track(f"<bdiv/>{_sec(1)}<voice>Ann</voice>Hi<voice>Ann</voice>Hello"), None),
    "text_then_voice": (_track(f"<bdiv/>{_sec(1)}Noise<voice>Ann</voice>Hello"), None),
    "empty_voice_ends_turn": (_track(f"<bdiv/>{_sec(1)}<voice>Ann</voice>Hi<voice/>laughter"), None),
    "empty_voice_between_unattributed": (_track(f"<bdiv/>{_sec(1)}One<voice/>Two"), None),
    "voice_does_not_carry_over": (_track(f"<bdiv/>{_sec(1)}<voice>Ann</voice>Hi<bdiv/>{_sec(2)}Hello"), None),
    "voice_with_escapes": (_track(f"<bdiv/>{_sec(1)}<voice>Ann &amp; Bob</voice>Hi"), None),
    "text_with_escapes": (_track(f"<bdiv/>{_sec(1)}a &lt; b &amp; c"), None),
    "cdata": (_track(f"<bdiv/>{_sec(1)}<![CDATA[a < b]]>"), None),
    "multi_line_text": (_track(f"<bdiv/>{_sec(1)}\n  One\n  Two\n"), None),
    "content_keeps_whitespace": (_track(f"<bdiv/>{_sec(1)}<content>  a  b  </content>"), None),
    "content_in_text": (_track(f"<bdiv/>{_sec(1)}a <content>  b</content> c"), None),
    "content_with_escapes": (_track(f"<bdiv/>{_sec(1)}<content><![CDATA[a  &  b]]></content>"), None),
    # 4. Frames and cover, without image sources
    "frame_only": (_track(f"<bdiv/>{_sec(1)}<frame/>"), None),
    "frame_and_turn": (_track(f"<bdiv/>{_sec(1)}<frame/>Hello"), None),
    "frame_only_with_end": (_track(f"<bdiv/>{_sec(1)}{_sec(9)}<frame/>"), _FRAME_ONLY_END),
    "frame_in_interval": (_track(f"<bdiv/>{_sec(1)}{_sec(9)}<frame/>Hello"), None),
    "frame_with_location": (_track(f"<bdiv/>{_sec(1)}<frame>{_LOC}</frame>"), None),
    "frame_with_head": (
        _track(
            f'<bdiv/>{_sec(1)}<frame><layer value="furniture"/>{_LOC}<caption>Slide</caption><description>A chart</description></frame>Hello'
        ),
        None,
    ),
    "frame_with_caption": (_track(f"<bdiv/>{_sec(1)}<frame><caption>Slide</caption></frame>"), None),
    "frame_with_text": (_track(f"<bdiv/>{_sec(1)}<frame>On screen</frame>"), _BODY_TEXT),
    "empty_cover": (_track(f"<cover/><bdiv/>{_sec(1)}Hello"), _COVER),
    "cover_with_caption": (_track(f"<cover><caption>Art</caption></cover><bdiv/>{_sec(1)}Hello"), _COVER),
    # 5. Track element head
    "track_location": (_track(f"{_LOC}<bdiv/>{_sec(1)}Hello"), None),
    "track_caption": (_track(f"<caption>Talk</caption><bdiv/>{_sec(1)}Hello"), None),
    "track_layer": (_track(f'<layer value="furniture"/><bdiv/>{_sec(1)}Hello'), None),
    "track_description": (_track(f"<description>About</description><bdiv/>{_sec(1)}Hello"), None),
    "track_summary": (_track(f"<summary>Short</summary><bdiv/>{_sec(1)}Hello"), None),
    "track_label": (_track(f'<label value="video"/><bdiv/>{_sec(1)}Hello'), _LABEL),
    "track_href": (_track(f'<href uri="https://example.com/v.mp4"/><bdiv/>{_sec(1)}Hello'), _GENERIC_HEAD),
    "track_thread": (_track(f'<thread thread_id="1"/><bdiv/>{_sec(1)}Hello'), _GENERIC_HEAD),
    # 6. Elements without a counterpart
    "chapter": (_track(f"<bdiv/>{_sec(1)}<chapter>Intro</chapter>Hello"), _CHAPTER),
    "chapter_only": (_track(f"<bdiv/>{_sec(1)}<chapter>Intro</chapter>"), _CHAPTER),
    "audio": (_track(f"<bdiv/>{_sec(1)}{_sec(2)}<audio/>Hello"), _AUDIO),
    # 7. Nesting
    "between_texts": (_doc(f"<text>Before</text><track><bdiv/>{_sec(1)}Hello</track><text>After</text>"), None),
    "two_tracks": (_doc(f"<track><bdiv/>{_sec(1)}A</track><track><bdiv/>{_sec(1)}B</track>"), None),
    "in_group_with_text": (_doc(f"<group><text>Intro</text><track><bdiv/>{_sec(1)}Hello</track></group>"), _GROUP),
    "in_table_cell": (_doc(f"<table><fcel/><track><bdiv/>{_sec(1)}Hello</track><nl/></table>"), None),
    # 8. Rich transcripts
    "bold_turn": (_track(f"<bdiv/>{_sec(1)}<bold>Hello</bold> there"), _RICH),
    "italic_voice": (_track(f"<bdiv/>{_sec(1)}<voice><italic>Ann</italic></voice>Hello"), _RICH),
    "voice_with_location": (_track(f"<bdiv/>{_sec(1)}<voice>{_LOC}Ann</voice>Hello"), _RICH),
}


def _params(names: list[str], reason_of):
    return [
        pytest.param(name, marks=pytest.mark.xfail(reason=reason, strict=True) if (reason := reason_of(name)) else ())
        for name in names
    ]


@pytest.mark.parametrize("indentation", ["  ", None], ids=["pretty", "compact"])
@pytest.mark.parametrize("name", _params(FIXTURE_NAMES, FIXTURE_GAPS.get))
def test_fixture_round_trip(name: str, indentation: str | None) -> None:
    """A track example of the DocLang repository reads and writes back the same."""
    _check((FIXTURES / f"{name}.dclg").read_text(encoding="utf-8"), indentation)


@pytest.mark.parametrize("indentation", ["  ", None], ids=["pretty", "compact"])
@pytest.mark.parametrize("name", _params(list(CASES), lambda name: CASES[name][1]))
def test_case_round_trip(name: str, indentation: str | None) -> None:
    """A hand-written case reads and writes back the same."""
    _check(CASES[name][0], indentation)


def test_track_images_round_trip_through_an_archive(tmp_path: Path) -> None:
    """Without an archive a ``<src>`` is not restored; with one, cover and frame images survive."""
    from PIL import Image

    doc = DoclingDocument(name="doc")
    track = doc.add_track(image=ImageRef.from_pil(Image.new("RGB", (4, 3), "red"), dpi=72))
    doc.add_picture(
        parent=track,
        image=ImageRef.from_pil(Image.new("RGB", (2, 2), "blue"), dpi=72),
        source=TrackSource(start_time=1, end_time=2),
    )
    doc.add_text(label=DocItemLabel.TEXT, text="Hello", parent=track, source=TrackSource(start_time=1, end_time=2))

    path = tmp_path / "doc.dclx"
    doc.save_as_doclang_archive(path)
    back = DoclingDocument.load_from_doclang_archive(path)

    assert back.tracks[0].image is not None and back.tracks[0].image.size.width == 4
    assert back.pictures[0].image is not None and back.pictures[0].image.size.width == 2
    assert [t.text for t in back.texts] == ["Hello"]
