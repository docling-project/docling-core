"""Test the data model for WebVTT files.

Assisted by watsonx Code Assistant.
Examples extracted from https://www.w3.org/TR/webvtt1/
Copyright © 2019 World Wide Web Consortium.
"""

import warnings

import pytest
from pydantic import ValidationError

from docling_core.types.doc.webvtt import (
    WebVTTCueBlock,
    WebVTTCueComponentWithTerminator,
    WebVTTCueInternalText,
    WebVTTCueItalicSpan,
    WebVTTCueLanguageSpan,
    WebVTTCueLanguageSpanStartTag,
    WebVTTCueSpanStartTagAnnotated,
    WebVTTCueTextSpan,
    WebVTTCueTimings,
    WebVTTCueVoiceSpan,
    WebVTTFile,
    WebVTTTimestamp,
)

from .test_data_gen_flag import GEN_TEST_DATA

GENERATE = GEN_TEST_DATA


def test_vtt_cue_commponents() -> None:
    """Test WebVTT components."""
    valid_timestamps = [
        "00:01:02.345",
        "12:34:56.789",
        "02:34.567",
        "00:00:00.000",
    ]
    valid_total_seconds = [
        1 * 60 + 2.345,
        12 * 3600 + 34 * 60 + 56.789,
        2 * 60 + 34.567,
        0.0,
    ]
    for idx, ts in enumerate(valid_timestamps):
        model = WebVTTTimestamp(raw=ts)
        assert model.seconds == valid_total_seconds[idx]

    """Test invalid WebVTT timestamps."""
    invalid_timestamps = [
        "00:60:02.345",  # minutes > 59
        "00:01:60.345",  # seconds > 59
        "00:01:02.1000",  # milliseconds > 999
        "01:02:03",  # missing milliseconds
        "01:02",  # missing milliseconds
        ":01:02.345",  # extra : for missing hours
        "abc:01:02.345",  # invalid format
    ]
    for ts in invalid_timestamps:
        with pytest.raises(ValidationError):
            WebVTTTimestamp(raw=ts)

    """Test the timestamp __str__ method."""
    model = WebVTTTimestamp(raw="00:01:02.345")
    assert str(model) == "00:01:02.345"

    """Test valid cue timings."""
    start = WebVTTTimestamp(raw="00:10.005")
    end = WebVTTTimestamp(raw="00:14.007")
    cue_timings = WebVTTCueTimings(start=start, end=end)
    assert cue_timings.start == start
    assert cue_timings.end == end
    assert str(cue_timings) == "00:10.005 --> 00:14.007"

    """Test invalid cue timings with end timestamp before start."""
    start = WebVTTTimestamp(raw="00:10.700")
    end = WebVTTTimestamp(raw="00:10.500")
    with pytest.raises(ValidationError) as excinfo:
        WebVTTCueTimings(start=start, end=end)
    assert "End timestamp must be greater than start timestamp" in str(excinfo.value)

    """Test invalid cue timings with missing end."""
    start = WebVTTTimestamp(raw="00:10.500")
    with pytest.raises(ValidationError) as excinfo:
        WebVTTCueTimings(start=start)  # type: ignore[call-arg]
    assert "Field required" in str(excinfo.value)

    """Test invalid cue timings with missing start."""
    end = WebVTTTimestamp(raw="00:10.500")
    with pytest.raises(ValidationError) as excinfo:
        WebVTTCueTimings(end=end)  # type: ignore[call-arg]
    assert "Field required" in str(excinfo.value)

    """Test with valid text."""
    valid_text = "This is a valid cue text span."
    span = WebVTTCueTextSpan(text=valid_text)
    assert span.text == valid_text
    assert str(span) == valid_text

    """Test with text containing newline characters."""
    invalid_text = "This cue text span\ncontains a newline."
    with pytest.raises(ValidationError):
        WebVTTCueTextSpan(text=invalid_text)

    """Test with text containing ampersand."""
    invalid_text = "This cue text span contains &."
    with pytest.raises(ValidationError):
        WebVTTCueTextSpan(text=invalid_text)
    invalid_text = "An invalid &foo; entity"
    with pytest.raises(ValidationError):
        WebVTTCueTextSpan(text=invalid_text)
    valid_text = "My favorite book is Pride &amp; Prejudice"
    span = WebVTTCueTextSpan(text=valid_text)
    assert span.text == valid_text

    """Test with text containing less-than sign."""
    invalid_text = "This cue text span contains <."
    with pytest.raises(ValidationError):
        WebVTTCueTextSpan(text=invalid_text)

    """Test with empty text."""
    with pytest.raises(ValidationError):
        WebVTTCueTextSpan(text="")

    """Test that annotation validation works correctly."""
    valid_annotation = "valid-annotation"
    invalid_annotation = "invalid\nannotation"
    with pytest.raises(ValidationError):
        WebVTTCueSpanStartTagAnnotated(name="v", annotation=invalid_annotation)
    assert WebVTTCueSpanStartTagAnnotated(name="v", annotation=valid_annotation)

    """Test that classes validation works correctly."""
    annotation = "speaker name"
    valid_classes = ["class1", "class2"]
    invalid_classes = ["class\nwith\nnewlines", ""]
    with pytest.raises(ValidationError):
        WebVTTCueSpanStartTagAnnotated(
            name="v", annotation=annotation, classes=invalid_classes
        )
    assert WebVTTCueSpanStartTagAnnotated(
        name="v", annotation=annotation, classes=valid_classes
    )

    """Test that components validation works correctly."""
    annotation = "speaker name"
    valid_components = [
        WebVTTCueComponentWithTerminator(
            component=WebVTTCueTextSpan(text="random text")
        )
    ]
    invalid_components = [123, "not a component"]
    with pytest.raises(ValidationError):
        WebVTTCueInternalText(components=invalid_components)
    assert WebVTTCueInternalText(components=valid_components)

    """Test valid cue voice spans."""
    cue_span = WebVTTCueVoiceSpan(
        start_tag=WebVTTCueSpanStartTagAnnotated(
            name="v", annotation="speaker", classes=["loud", "clear"]
        ),
        internal_text=WebVTTCueInternalText(
            components=[
                WebVTTCueComponentWithTerminator(
                    component=WebVTTCueTextSpan(text="random text")
                )
            ]
        ),
    )
    expected_str = "<v.loud.clear speaker>random text</v>"
    assert str(cue_span) == expected_str

    cue_span = WebVTTCueVoiceSpan(
        start_tag=WebVTTCueSpanStartTagAnnotated(name="v", annotation="speaker"),
        internal_text=WebVTTCueInternalText(
            components=[
                WebVTTCueComponentWithTerminator(
                    component=WebVTTCueTextSpan(text="random text")
                )
            ]
        ),
    )
    expected_str = "<v speaker>random text</v>"
    assert str(cue_span) == expected_str


def test_webvttcueblock_parse() -> None:
    """Test the method parse of _WebVTTCueBlock class."""
    raw: str = (
        "04:02.500 --> 04:05.000\n" "J’ai commencé le basket à l'âge de 13, 14 ans\n"
    )
    block: WebVTTCueBlock = WebVTTCueBlock.parse(raw)
    assert str(block.timings) == "04:02.500 --> 04:05.000"
    assert len(block.payload) == 1
    assert isinstance(block.payload[0], WebVTTCueComponentWithTerminator)
    assert isinstance(block.payload[0].component, WebVTTCueTextSpan)
    assert (
        block.payload[0].component.text
        == "J’ai commencé le basket à l'âge de 13, 14 ans"
    )
    assert raw == str(block)

    raw = (
        "04:05.001 --> 04:07.800\n"
        "Sur les <i.foreignphrase><lang en>playground</lang></i>, ici à Montpellier\n"
    )
    block = WebVTTCueBlock.parse(raw)
    assert str(block.timings) == "04:05.001 --> 04:07.800"
    assert len(block.payload) == 3
    assert isinstance(block.payload[0], WebVTTCueComponentWithTerminator)
    assert isinstance(block.payload[0].component, WebVTTCueTextSpan)
    assert block.payload[0].component.text == "Sur les "
    assert isinstance(block.payload[1], WebVTTCueComponentWithTerminator)
    assert isinstance(block.payload[1].component, WebVTTCueItalicSpan)
    assert len(block.payload[1].component.internal_text.components) == 1
    lang_span = block.payload[1].component.internal_text.components[0].component
    assert isinstance(lang_span, WebVTTCueLanguageSpan)
    assert isinstance(
        lang_span.internal_text.components[0].component, WebVTTCueTextSpan
    )
    assert lang_span.internal_text.components[0].component.text == "playground"
    assert isinstance(block.payload[2], WebVTTCueComponentWithTerminator)
    assert isinstance(block.payload[2].component, WebVTTCueTextSpan)
    assert block.payload[2].component.text == ", ici à Montpellier"
    assert raw == str(block)


def test_webvtt_file() -> None:
    """Test WebVTT files."""
    with open("./test/data/webvtt/webvtt_example_01.vtt", encoding="utf-8") as f:
        content = f.read()
        vtt = WebVTTFile.parse(content)
    assert len(vtt) == 13
    block = vtt.cue_blocks[11]
    assert str(block.timings) == "00:32.500 --> 00:33.500"
    assert len(block.payload) == 1
    cue_span = block.payload[0]
    assert isinstance(cue_span.component, WebVTTCueVoiceSpan)
    assert cue_span.component.start_tag.annotation == "Neil deGrasse Tyson"
    assert not cue_span.component.start_tag.classes
    assert len(cue_span.component.internal_text.components) == 1
    comp = cue_span.component.internal_text.components[0]
    assert isinstance(comp.component, WebVTTCueItalicSpan)
    assert len(comp.component.internal_text.components) == 1
    comp2 = comp.component.internal_text.components[0]
    assert isinstance(comp2.component, WebVTTCueTextSpan)
    assert comp2.component.text == "Laughs"

    with open("./test/data/webvtt/webvtt_example_02.vtt", encoding="utf-8") as f:
        content = f.read()
        vtt = WebVTTFile.parse(content)
    assert len(vtt) == 4
    reverse = (
        "WEBVTT\n\nNOTE Copyright © 2019 World Wide Web Consortium. "
        "https://www.w3.org/TR/webvtt1/\n\n"
    )
    reverse += "\n".join([str(block) for block in vtt.cue_blocks])
    assert content == reverse.rstrip()

    with open("./test/data/webvtt/webvtt_example_03.vtt", encoding="utf-8") as f:
        content = f.read()
        vtt = WebVTTFile.parse(content)
    assert len(vtt) == 13
    for block in vtt:
        assert block.identifier
    block = vtt.cue_blocks[0]
    assert block.identifier == "62357a1d-d250-41d5-a1cf-6cc0eeceffcc/15-0"
    assert str(block.timings) == "00:00:04.963 --> 00:00:08.571"
    assert len(block.payload) == 1
    assert isinstance(block.payload[0].component, WebVTTCueVoiceSpan)
    block = vtt.cue_blocks[2]
    assert block.identifier == "62357a1d-d250-41d5-a1cf-6cc0eeceffcc/16-0"
    assert str(block.timings) == "00:00:10.683 --> 00:00:11.563"
    assert len(block.payload) == 1
    assert isinstance(block.payload[0].component, WebVTTCueTextSpan)
    assert block.payload[0].component.text == "Good."
    assert not vtt.title

    with open("./test/data/webvtt/webvtt_example_04.vtt", encoding="utf-8") as f:
        content = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vtt = WebVTTFile.parse(content)
    assert len(vtt) == 2
    block = vtt.cue_blocks[1]
    assert len(block.payload) == 5
    assert str(block) == (
        "00:05.000 --> 00:09.000\n"
        "— It will perforate your stomach.\n"
        "— You could <b.loud>die</b>.\n"
        "<v John>This is true.</v>\n"
    )
    assert vtt.title == "Danger of Nitrogen"


def test_webvtt_cue_language_span_start_tag():
    WebVTTCueLanguageSpanStartTag.model_validate_json('{"annotation": "en"}')
    WebVTTCueLanguageSpanStartTag.model_validate_json('{"annotation": "en-US"}')
    WebVTTCueLanguageSpanStartTag.model_validate_json('{"annotation": "zh-Hant"}')
    with pytest.raises(ValidationError, match="BCP 47"):
        WebVTTCueLanguageSpanStartTag.model_validate_json('{"annotation": "en_US"}')
    with pytest.raises(ValidationError, match="BCP 47"):
        WebVTTCueLanguageSpanStartTag.model_validate_json('{"annotation": "123-de"}')
