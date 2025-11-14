"""Models for the Docling's adoption of Web Video Text Tracks format."""

import logging
import re
from typing import Annotated, ClassVar, Literal, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import StringConstraints
from typing_extensions import Self, override

_log = logging.getLogger(__name__)


class _WebVTTTimestamp(BaseModel):
    """Model representing a WebVTT timestamp.

    A WebVTT timestamp is always interpreted relative to the current playback position
    of the media data that the WebVTT file is to be synchronized with.
    """

    model_config = ConfigDict(regex_engine="python-re")

    raw: Annotated[
        str,
        Field(
            description="A representation of the WebVTT Timestamp as a single string"
        ),
    ]

    _pattern: ClassVar[re.Pattern] = re.compile(
        r"^(?:(\d{2,}):)?([0-5]\d):([0-5]\d)\.(\d{3})$"
    )
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
        return (
            self._hours * 3600
            + self._minutes * 60
            + self._seconds
            + self._millis / 1000.0
        )

    @override
    def __str__(self) -> str:
        return self.raw


_WebVTTCueIdentifier = Annotated[
    str, StringConstraints(strict=True, pattern=r"^(?!.*-->)[^\n\r]+$")
]


class _WebVTTCueTimings(BaseModel):
    """Model representating WebVTT cue timings."""

    start: Annotated[
        _WebVTTTimestamp, Field(description="Start time offset of the cue")
    ]
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
    """Model representing a WebVTT cue text span."""

    text: str
    span_type: Literal["text"] = "text"

    _valid_entities: ClassVar[set] = {"amp", "lt", "gt", "lrm", "rlm", "nbsp"}
    _entity_pattern: ClassVar[re.Pattern] = re.compile(r"&([a-zA-Z0-9]+);")

    @field_validator("text", mode="after")
    @classmethod
    def validate_text(cls, value: str) -> str:
        for match in cls._entity_pattern.finditer(value):
            entity = match.group(1)
            if entity not in cls._valid_entities:
                raise ValueError(
                    f"Cue text span contains an invalid HTML entity: &{entity};"
                )
        if "&" in re.sub(cls._entity_pattern, "", value):
            raise ValueError(
                "Found '&' not part of a valid entity in the cue text span"
            )
        if any(ch in value for ch in {"\n", "\r", "<"}):
            raise ValueError("Cue text span contains invalid characters")
        if len(value) == 0:
            raise ValueError("Cue text span cannot be empty")
        return value

    @override
    def __str__(self):
        return self.text


class _WebVTTCueVoiceSpan(BaseModel):
    """Model representing a WebVTT cue voice span."""

    annotation: Annotated[
        str,
        Field(
            description=(
                "Cue span start tag annotation text representing the name of thevoice"
            )
        ),
    ]
    classes: Annotated[
        list[str],
        Field(description="List of classes representing the cue span's significance"),
    ] = []
    components: Annotated[
        list["_WebVTTCueComponent"],
        Field(description="The components representing the cue internal text"),
    ] = []
    span_type: Literal["v"] = "v"

    @field_validator("annotation", mode="after")
    @classmethod
    def validate_annotation(cls, value: str) -> str:
        if any(ch in value for ch in {"\n", "\r", "&", ">"}):
            raise ValueError(
                "Cue span start tag annotation contains invalid characters"
            )
        if not value:
            raise ValueError("Cue text span cannot be empty")
        return value

    @field_validator("classes", mode="after")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        for item in value:
            if any(ch in item for ch in {"\t", "\n", "\r", " ", "&", "<", ">", "."}):
                raise ValueError(
                    "A cue span start tag class contains invalid characters"
                )
            if not item:
                raise ValueError("Cue span start tag classes cannot be empty")
        return value

    @override
    def __str__(self):
        tag = f"v.{'.'.join(self.classes)}" if self.classes else "v"
        inner = "".join(str(span) for span in self.components)
        return f"<{tag} {self.annotation}>{inner}</v>"


class _WebVTTCueClassSpan(BaseModel):
    span_type: Literal["c"] = "c"
    components: list["_WebVTTCueComponent"]

    @override
    def __str__(self):
        inner = "".join(str(span) for span in self.components)
        return f"<c>{inner}</c>"


class _WebVTTCueItalicSpan(BaseModel):
    span_type: Literal["i"] = "i"
    components: list["_WebVTTCueComponent"]

    @override
    def __str__(self):
        inner = "".join(str(span) for span in self.components)
        return f"<i>{inner}</i>"


class _WebVTTCueBoldSpan(BaseModel):
    span_type: Literal["b"] = "b"
    components: list["_WebVTTCueComponent"]

    @override
    def __str__(self):
        inner = "".join(str(span) for span in self.components)
        return f"<b>{inner}</b>"


class _WebVTTCueUnderlineSpan(BaseModel):
    span_type: Literal["u"] = "u"
    components: list["_WebVTTCueComponent"]

    @override
    def __str__(self):
        inner = "".join(str(span) for span in self.components)
        return f"<u>{inner}</u>"


_WebVTTCueComponent = Annotated[
    Union[
        _WebVTTCueTextSpan,
        _WebVTTCueClassSpan,
        _WebVTTCueItalicSpan,
        _WebVTTCueBoldSpan,
        _WebVTTCueUnderlineSpan,
        _WebVTTCueVoiceSpan,
    ],
    Field(discriminator="span_type", description="The WebVTT cue component"),
]


class _WebVTTCueBlock(BaseModel):
    """Model representing a WebVTT cue block.

    The optional WebVTT cue settings list is not supported.
    The cue payload is limited to the following spans: text, class, italic, bold,
    underline, and voice.
    """

    model_config = ConfigDict(regex_engine="python-re")

    identifier: Optional[_WebVTTCueIdentifier] = Field(
        None, description="The WebVTT cue identifier"
    )
    timings: Annotated[_WebVTTCueTimings, Field(description="The WebVTT cue timings")]
    payload: Annotated[list[_WebVTTCueComponent], Field(description="The cue payload")]

    _pattern_block: ClassVar[re.Pattern] = re.compile(
        r"<(/?)(i|b|c|u|v(?:\.[^\t\n\r &<>.]+)*)(?:\s+([^>]*))?>"
    )
    _pattern_voice_tag: ClassVar[re.Pattern] = re.compile(
        r"^<v(?P<class>\.[^\t\n\r &<>]+)?"  # zero or more classes
        r"[ \t]+(?P<annotation>[^\n\r&>]+)>"  # required space and annotation
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
        timings: _WebVTTCueTimings = _WebVTTCueTimings(
            start=_WebVTTTimestamp(raw=start), end=_WebVTTTimestamp(raw=end)
        )
        cue_text = " ".join(cue_lines).strip()
        if cue_text.startswith("<v") and "</v>" not in cue_text:
            # adding close tag for cue voice spans without end tag
            cue_text += "</v>"

        stack: list[list[_WebVTTCueComponent]] = [[]]
        tag_stack: list[Union[str, tuple]] = []

        pos = 0
        matches = list(cls._pattern_block.finditer(cue_text))
        i = 0
        while i < len(matches):
            match = matches[i]
            if match.start() > pos:
                stack[-1].append(_WebVTTCueTextSpan(text=cue_text[pos : match.start()]))
            tag = match.group(0)

            if tag.startswith(("<i>", "<b>", "<u>", "<c>")):
                tag_type = tag[1:2]
                tag_stack.append(tag_type)
                stack.append([])
            elif tag == "</i>":
                children = stack.pop()
                stack[-1].append(_WebVTTCueItalicSpan(components=children))
                tag_stack.pop()
            elif tag == "</b>":
                children = stack.pop()
                stack[-1].append(_WebVTTCueBoldSpan(components=children))
                tag_stack.pop()
            elif tag == "</u>":
                children = stack.pop()
                stack[-1].append(_WebVTTCueUnderlineSpan(components=children))
                tag_stack.pop()
            elif tag == "</c>":
                children = stack.pop()
                stack[-1].append(_WebVTTCueClassSpan(components=children))
                tag_stack.pop()
            elif tag.startswith("<v"):
                tag_stack.append(("v", tag))
                stack.append([])
            elif tag.startswith("</v"):
                children = stack.pop() if stack else []
                if (
                    tag_stack
                    and isinstance(tag_stack[-1], tuple)
                    and tag_stack[-1][0] == "v"
                ):
                    _, voice = cast(tuple, tag_stack.pop())
                    voice_match = cls._pattern_voice_tag.match(voice)
                    if voice_match:
                        class_string = voice_match.group("class")
                        annotation = voice_match.group("annotation")
                        if annotation:
                            classes: list[str] = []
                            if class_string:
                                classes = [c for c in class_string.split(".") if c]
                            stack[-1].append(
                                _WebVTTCueVoiceSpan(
                                    annotation=annotation.strip(),
                                    classes=classes,
                                    components=children,
                                )
                            )

            pos = match.end()
            i += 1

        if pos < len(cue_text):
            stack[-1].append(_WebVTTCueTextSpan(text=cue_text[pos:]))

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
            if idx == 0 and len(self.payload) == 1 and span.span_type == "v":
                # the end tag may be omitted for brevity
                parts.append(str(span).removesuffix("</v>"))
            else:
                parts.append(str(span))

        return "".join(parts)


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
