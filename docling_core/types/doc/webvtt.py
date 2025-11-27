"""Models for the Docling's adoption of Web Video Text Tracks format."""

import logging
import re
from enum import Enum
from typing import Annotated, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import StringConstraints
from typing_extensions import Self, override

_log = logging.getLogger(__name__)


_VALID_ENTITIES: set = {"amp", "lt", "gt", "lrm", "rlm", "nbsp"}
_ENTITY_PATTERN: re.Pattern = re.compile(r"&([a-zA-Z0-9]+);")
_START_TAG_NAMES = Literal["c", "b", "i", "u", "v", "lang"]


class _WebVTTLineTerminator(str, Enum):
    CRLF = "\r\n"
    LF = "\n"
    CR = "\r"


_WebVTTCueIdentifier = Annotated[str, StringConstraints(strict=True, pattern=r"^(?!.*-->)[^\n\r]+$")]


class _WebVTTTimestamp(BaseModel):
    """WebVTT timestamp.

    A WebVTT timestamp is always interpreted relative to the current playback position
    of the media data that the WebVTT file is to be synchronized with.
    """

    model_config = ConfigDict(regex_engine="python-re")

    raw: Annotated[
        str,
        Field(description="A representation of the WebVTT Timestamp as a single string"),
    ]

    _pattern: ClassVar[re.Pattern] = re.compile(r"^(?:(\d{2,}):)?([0-5]\d):([0-5]\d)\.(\d{3})$")
    _hours: int
    _minutes: int
    _seconds: int
    _millis: int

    @model_validator(mode="after")
    def validate_raw(self) -> Self:
        m = self._pattern.match(self.raw)
        if not m:
            raise ValueError(f"Invalid WebVTT timestamp format: {self.raw}")
        self._hours = int(m.group(1)) if m.group(1) else 0
        self._minutes = int(m.group(2))
        self._seconds = int(m.group(3))
        self._millis = int(m.group(4))

        if self._minutes < 0 or self._minutes > 59:
            raise ValueError("Minutes must be between 0 and 59")
        if self._seconds < 0 or self._seconds > 59:
            raise ValueError("Seconds must be between 0 and 59")

        return self

    @property
    def seconds(self) -> float:
        """A representation of the WebVTT Timestamp in seconds."""
        return self._hours * 3600 + self._minutes * 60 + self._seconds + self._millis / 1000.0

    @override
    def __str__(self) -> str:
        return self.raw


class _WebVTTCueTimings(BaseModel):
    """WebVTT cue timings."""

    start: Annotated[_WebVTTTimestamp, Field(description="Start time offset of the cue")]
    end: Annotated[_WebVTTTimestamp, Field(description="End time offset of the cue")]

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.start and self.end:
            if self.end.seconds <= self.start.seconds:
                raise ValueError("End timestamp must be greater than start timestamp")
        return self

    @override
    def __str__(self):
        return f"{self.start} --> {self.end}"


class _WebVTTCueTextSpan(BaseModel):
    """WebVTT cue text span."""

    kind: Literal["text"] = "text"
    text: Annotated[str, Field(description="The cue text.")]

    @field_validator("text", mode="after")
    @classmethod
    def is_valid_text(cls, value: str) -> str:
        for match in _ENTITY_PATTERN.finditer(value):
            entity = match.group(1)
            if entity not in _VALID_ENTITIES:
                raise ValueError(f"Cue text contains an invalid HTML entity: &{entity};")
        if "&" in re.sub(_ENTITY_PATTERN, "", value):
            raise ValueError("Found '&' not part of a valid entity in the cue text")
        if any(ch in value for ch in {"\n", "\r", "<"}):
            raise ValueError("Cue text contains invalid characters")
        if len(value) == 0:
            raise ValueError("Cue text cannot be empty")

        return value

    @override
    def __str__(self):
        return self.text


class _WebVTTCueComponentWithTerminator(BaseModel):
    """WebVTT caption or subtitle cue component optionally with a line terminator."""

    component: "_WebVTTCueComponent"
    terminator: Optional[_WebVTTLineTerminator] = None

    @override
    def __str__(self):
        return f"{self.component}{self.terminator.value if self.terminator else ''}"


class _WebVTTCueInternalText(BaseModel):
    """WebVTT cue internal text."""

    terminator: Optional[_WebVTTLineTerminator] = None
    components: Annotated[
        list[_WebVTTCueComponentWithTerminator],
        Field(description=("WebVTT caption or subtitle cue components representing the cue internal text")),
    ] = []

    @override
    def __str__(self):
        cue_str = f"{self.terminator.value if self.terminator else ''}{''.join(str(span) for span in self.components)}"
        return cue_str


class _WebVTTCueSpanStartTag(BaseModel):
    """WebVTT cue span start tag."""

    name: Annotated[_START_TAG_NAMES, Field(description="The tag name")]
    classes: Annotated[
        list[str],
        Field(description="List of classes representing the cue span's significance"),
    ] = []

    @field_validator("classes", mode="after")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        for item in value:
            if any(ch in item for ch in {"\t", "\n", "\r", " ", "&", "<", ">", "."}):
                raise ValueError("A cue span start tag class contains invalid characters")
            if not item:
                raise ValueError("Cue span start tag classes cannot be empty")
        return value

    def _get_name_with_classes(self) -> str:
        return f"{self.name}.{'.'.join(self.classes)}" if self.classes else self.name

    @override
    def __str__(self):
        return f"<{self._get_name_with_classes()}>"


class _WebVTTCueSpanStartTagAnnotated(_WebVTTCueSpanStartTag):
    """WebVTT cue span start tag requiring an annotation."""

    annotation: Annotated[str, Field(description="Cue span start tag annotation")]

    @field_validator("annotation", mode="after")
    @classmethod
    def is_valid_annotation(cls, value: str) -> str:
        for match in _ENTITY_PATTERN.finditer(value):
            entity = match.group(1)
            if entity not in _VALID_ENTITIES:
                raise ValueError(f"Annotation contains an invalid HTML entity: &{entity};")
        if "&" in re.sub(_ENTITY_PATTERN, "", value):
            raise ValueError("Found '&' not part of a valid entity in annotation")
        if any(ch in value for ch in {"\n", "\r", ">"}):
            raise ValueError("Annotation contains invalid characters")
        if len(value) == 0:
            raise ValueError("Annotation cannot be empty")

        return value

    @override
    def __str__(self):
        return f"<{self._get_name_with_classes()} {self.annotation}>"


class _WebVTTCueComponentBase(BaseModel):
    """WebVTT caption or subtitle cue component.

    All the WebVTT caption or subtitle cue components are represented by this class
    except the WebVTT cue text span, which requires different definitions.
    """

    kind: Literal["c", "b", "i", "u", "v", "lang"]
    start_tag: _WebVTTCueSpanStartTag
    internal_text: _WebVTTCueInternalText

    @model_validator(mode="after")
    def check_tag_names_match(self) -> Self:
        if self.kind != self.start_tag.name:
            raise ValueError("The tag name of this cue component should be {self.kind}")
        return self

    @override
    def __str__(self):
        return f"{self.start_tag}{self.internal_text}</{self.start_tag.name}>"


class _WebVTTCueVoiceSpan(_WebVTTCueComponentBase):
    """WebVTT cue voice span associated with a specific voice."""

    kind: Literal["v"] = "v"
    start_tag: _WebVTTCueSpanStartTagAnnotated


class _WebVTTCueClassSpan(_WebVTTCueComponentBase):
    """WebVTT cue class span.

    It represents a span of text and it is used to annotate parts of the cue with
    applicable classes without implying further meaning (such as italics or bold).
    """

    kind: Literal["c"] = "c"
    start_tag: _WebVTTCueSpanStartTag = _WebVTTCueSpanStartTag(name="c")


class _WebVTTCueItalicSpan(_WebVTTCueComponentBase):
    """WebVTT cue italic span representing a span of italic text."""

    kind: Literal["i"] = "i"
    start_tag: _WebVTTCueSpanStartTag = _WebVTTCueSpanStartTag(name="i")


class _WebVTTCueBoldSpan(_WebVTTCueComponentBase):
    """WebVTT cue bold span representing a span of bold text."""

    kind: Literal["b"] = "b"
    start_tag: _WebVTTCueSpanStartTag = _WebVTTCueSpanStartTag(name="b")


class _WebVTTCueUnderlineSpan(_WebVTTCueComponentBase):
    """WebVTT cue underline span representing a span of underline text."""

    kind: Literal["u"] = "u"
    start_tag: _WebVTTCueSpanStartTag = _WebVTTCueSpanStartTag(name="u")


class _WebVTTCueLanguageSpan(_WebVTTCueComponentBase):
    """WebVTT cue language span.

    It represents a span of text and it is used to annotate parts of the cue where the
    applicable language might be different than the surrounding text's, without
    implying further meaning (such as italics or bold).
    """

    kind: Literal["lang"] = "lang"
    start_tag: _WebVTTCueSpanStartTagAnnotated


_WebVTTCueComponent = Annotated[
    Union[
        _WebVTTCueTextSpan,
        _WebVTTCueClassSpan,
        _WebVTTCueItalicSpan,
        _WebVTTCueBoldSpan,
        _WebVTTCueUnderlineSpan,
        _WebVTTCueVoiceSpan,
        _WebVTTCueLanguageSpan,
    ],
    Field(
        discriminator="kind",
        description="The type of WebVTT caption or subtitle cue component.",
    ),
]


class _WebVTTCueBlock(BaseModel):
    """Model representing a WebVTT cue block.

    The optional WebVTT cue settings list is not supported.
    The cue payload is limited to the following spans: text, class, italic, bold,
    underline, and voice.
    """

    model_config = ConfigDict(regex_engine="python-re")

    identifier: Optional[_WebVTTCueIdentifier] = Field(None, description="The WebVTT cue identifier")
    timings: Annotated[_WebVTTCueTimings, Field(description="The WebVTT cue timings")]
    payload: Annotated[
        list[_WebVTTCueComponentWithTerminator],
        Field(description="The WebVTT caption or subtitle cue text"),
    ]

    # pattern of a WebVTT cue span start/end tag
    _pattern_tag: ClassVar[re.Pattern] = re.compile(
        r"<(?P<end>/?)"
        r"(?P<tag>i|b|c|u|v|lang)"
        r"(?P<class>(?:\.[^\t\n\r &<>.]+)*)"
        r"(?:[ \t](?P<annotation>[^\n\r&>]*))?>"
    )

    @field_validator("payload", mode="after")
    @classmethod
    def validate_payload(cls, payload):
        for voice in payload:
            if "-->" in str(voice):
                raise ValueError("Cue payload must not contain '-->'")
        return payload

    @classmethod
    def parse(cls, raw: str) -> "_WebVTTCueBlock":
        lines = raw.strip().splitlines()
        if not lines:
            raise ValueError("Cue block must have at least one line")
        identifier: Optional[_WebVTTCueIdentifier] = None
        timing_line = lines[0]
        if "-->" not in timing_line and len(lines) > 1:
            identifier = timing_line
            timing_line = lines[1]
            cue_lines = lines[2:]
        else:
            cue_lines = lines[1:]

        if "-->" not in timing_line:
            raise ValueError("Cue block must contain WebVTT cue timings")

        start, end = [t.strip() for t in timing_line.split("-->")]
        end = re.split(" |\t", end)[0]  # ignore the cue settings list
        timings: _WebVTTCueTimings = _WebVTTCueTimings(start=_WebVTTTimestamp(raw=start), end=_WebVTTTimestamp(raw=end))
        cue_text = " ".join(cue_lines).strip()
        # adding close tag for cue spans without end tag
        for omm in {"v"}:
            if cue_text.startswith(f"<{omm}") and f"</{omm}>" not in cue_text:
                cue_text += f"</{omm}>"
                break

        stack: list[list[_WebVTTCueComponentWithTerminator]] = [[]]
        tag_stack: list[dict] = []

        pos = 0
        matches = list(cls._pattern_tag.finditer(cue_text))
        i = 0
        while i < len(matches):
            match = matches[i]
            if match.start() > pos:
                stack[-1].append(
                    _WebVTTCueComponentWithTerminator(component=_WebVTTCueTextSpan(text=cue_text[pos : match.start()]))
                )
            gps = {k: (v if v else None) for k, v in match.groupdict().items()}

            if gps["tag"] in {"c", "b", "i", "u", "v", "lang"}:
                if not gps["end"]:
                    tag_stack.append(gps)
                    stack.append([])
                else:
                    children = stack.pop() if stack else []
                    if tag_stack:
                        closed = tag_stack.pop()
                        if (ct := closed["tag"]) != gps["tag"]:
                            raise ValueError(f"Incorrect end tag: {ct}")
                        class_string = closed["class"]
                        annotation = closed["annotation"]
                        classes: list[str] = []
                        if class_string:
                            classes = [c for c in class_string.split(".") if c]
                        st = (
                            _WebVTTCueSpanStartTagAnnotated(name=ct, classes=classes, annotation=annotation.strip())
                            if annotation
                            else _WebVTTCueSpanStartTag(name=ct, classes=classes)
                        )
                        it = _WebVTTCueInternalText(components=children)
                        cp: _WebVTTCueComponent
                        if ct == "c":
                            cp = _WebVTTCueClassSpan(start_tag=st, internal_text=it)
                        elif ct == "b":
                            cp = _WebVTTCueBoldSpan(start_tag=st, internal_text=it)
                        elif ct == "i":
                            cp = _WebVTTCueItalicSpan(start_tag=st, internal_text=it)
                        elif ct == "u":
                            cp = _WebVTTCueUnderlineSpan(start_tag=st, internal_text=it)
                        elif ct == "lang":
                            cp = _WebVTTCueLanguageSpan(start_tag=st, internal_text=it)
                        elif ct == "v":
                            cp = _WebVTTCueVoiceSpan(start_tag=st, internal_text=it)
                        stack[-1].append(_WebVTTCueComponentWithTerminator(component=cp))

            pos = match.end()
            i += 1

        if pos < len(cue_text):
            stack[-1].append(_WebVTTCueComponentWithTerminator(component=_WebVTTCueTextSpan(text=cue_text[pos:])))

        return cls(
            identifier=identifier,
            timings=timings,
            payload=stack[0],
        )

    def __str__(self):
        parts = []
        if self.identifier:
            parts.append(f"{self.identifier}\n")
        timings_line = str(self.timings)
        parts.append(timings_line + "\n")
        for idx, span in enumerate(self.payload):
            if idx == 0 and len(self.payload) == 1 and span.component.kind == "v":
                # the end tag may be omitted for brevity
                parts.append(str(span).removesuffix("</v>"))
            else:
                parts.append(str(span))

        return "".join(parts) + "\n"


class _WebVTTFile(BaseModel):
    """A model representing a WebVTT file."""

    cue_blocks: list[_WebVTTCueBlock]

    @staticmethod
    def verify_signature(content: str) -> bool:
        if not content:
            return False
        elif len(content) == 6:
            return content == "WEBVTT"
        elif len(content) > 6 and content.startswith("WEBVTT"):
            return content[6] in (" ", "\t", "\n")
        else:
            return False

    @classmethod
    def parse(cls, raw: str) -> "_WebVTTFile":
        # Normalize newlines to LF
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")

        # Check WebVTT signature
        if not cls.verify_signature(raw):
            raise ValueError("Invalid WebVTT file signature")

        # Strip "WEBVTT" header line
        lines = raw.split("\n", 1)
        body = lines[1] if len(lines) > 1 else ""

        # Remove NOTE/STYLE/REGION blocks
        body = re.sub(r"^(NOTE[^\n]*\n(?:.+\n)*?)\n", "", body, flags=re.MULTILINE)
        body = re.sub(r"^(STYLE|REGION)(?:.+\n)*?\n", "", body, flags=re.MULTILINE)

        # Split into cue blocks
        raw_blocks = re.split(r"\n\s*\n", body.strip())
        cues: list[_WebVTTCueBlock] = []
        for block in raw_blocks:
            try:
                cues.append(_WebVTTCueBlock.parse(block))
            except ValueError as e:
                _log.warning(f"Failed to parse cue block:\n{block}\n{e}")

        return cls(cue_blocks=cues)

    def __iter__(self):
        return iter(self.cue_blocks)

    def __getitem__(self, idx):
        return self.cue_blocks[idx]

    def __len__(self):
        return len(self.cue_blocks)
