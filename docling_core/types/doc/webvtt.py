"""Models for the Docling's adoption of Web Video Text Tracks format."""

import re
import warnings
from collections.abc import Iterator
from enum import Enum
from functools import total_ordering
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import StringConstraints
from typing_extensions import Self, override

_VALID_ENTITIES: set = {"amp", "lt", "gt", "lrm", "rlm", "nbsp"}
_ENTITY_PATTERN: re.Pattern = re.compile(r"&([a-zA-Z0-9]+);")
START_TAG_NAMES = Literal["c", "b", "i", "u", "v", "lang"]


class WebVTTLineTerminator(str, Enum):
    """WebVTT line terminator."""

    CRLF = "\r\n"
    LF = "\n"
    CR = "\r"


WebVTTCueIdentifier = Annotated[str, StringConstraints(strict=True, pattern=r"^(?!.*-->)[^\n\r]+$")]


@total_ordering
class WebVTTTimestamp(BaseModel):
    """WebVTT timestamp.

    The timestamp is a string consisting of the following components in the given order:

    - hours (optional, required if non-zero): two or more digits
    - minutes: two digits between 0 and 59
    - a colon character (:)
    - seconds: two digits between 0 and 59
    - a full stop character (.)
    - thousandths of a second: three digits

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
        """Validate the WebVTT timestamp as a string."""
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

    @classmethod
    def from_seconds(cls, seconds: float) -> Self:
        """Create a WebVTT timestamp from seconds.

        Args:
            seconds: The time in seconds (can include fractional seconds for milliseconds).

        Returns:
            A WebVTT timestamp instance.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis: int = round((seconds % 1) * 1000)

        return cls(raw=f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}")

    def __eq__(self, other: object) -> bool:
        """Two timestamps are equal if their total number of seconds is equal."""
        if not isinstance(other, WebVTTTimestamp):
            return NotImplemented
        return self.seconds == other.seconds

    def __lt__(self, other: "WebVTTTimestamp") -> bool:
        """Return True if this timestamp occurs before `other`."""
        if not isinstance(other, WebVTTTimestamp):
            return NotImplemented
        return self.seconds < other.seconds

    def format(self, omit_hours_if_zero: bool = False) -> str:
        """Format the timestamp as a string.

        Args:
            omit_hours_if_zero: If True, omit hours when they are 0.

        Returns:
            Formatted timestamp string.
        """
        if omit_hours_if_zero and self._hours == 0:
            return f"{self._minutes:02d}:{self._seconds:02d}.{self._millis:03d}"
        return self.raw

    @override
    def __str__(self) -> str:
        """Return a string representation of a WebVTT timestamp.

        Always returns the full timestamp format including hours (HH:MM:SS.mmm),
        even when hours are zero. Use `format(omit_hours_if_zero=True)` to get
        a shorter representation (MM:SS.mmm) when hours are zero.
        """
        return self.raw


class WebVTTCueTimings(BaseModel):
    """WebVTT cue timings."""

    start: Annotated[WebVTTTimestamp, Field(description="Start time offset of the cue")]
    end: Annotated[WebVTTTimestamp, Field(description="End time offset of the cue")]

    @model_validator(mode="after")
    def check_order(self) -> Self:
        """Ensure start timestamp is less than end timestamp."""
        if self.start and self.end:
            if self.end <= self.start:
                raise ValueError("End timestamp must be greater than start timestamp")
        return self

    def format(self, omit_hours_if_zero: bool = False) -> str:
        """Format the cue timings as a string.

        Args:
            omit_hours_if_zero: If True, omit hours when they are 0 in both timestamps.

        Returns:
            Formatted cue timings string in the format "start --> end".
        """
        start_str = self.start.format(omit_hours_if_zero=omit_hours_if_zero)
        end_str = self.end.format(omit_hours_if_zero=omit_hours_if_zero)
        return f"{start_str} --> {end_str}"

    @override
    def __str__(self) -> str:
        """Return a string representation of the cue timings.

        Always returns the full format including hours (HH:MM:SS.mmm --> HH:MM:SS.mmm),
        even when hours are zero. Use `format(omit_hours_if_zero=True)` to get
        a shorter representation when hours are zero.
        """
        return f"{self.start} --> {self.end}"


class WebVTTCueTextSpan(BaseModel):
    """WebVTT cue text span."""

    kind: Literal["text"] = "text"
    text: Annotated[str, Field(description="The cue text.")]

    @field_validator("text", mode="after")
    @classmethod
    def is_valid_text(cls, value: str) -> str:
        """Ensure cue text contains only permitted characters and HTML entities."""
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
    def __str__(self) -> str:
        """Return a string representation of the cue text span."""
        return self.text


class WebVTTCueComponentWithTerminator(BaseModel):
    """WebVTT caption or subtitle cue component optionally with a line terminator."""

    component: "WebVTTCueComponent"
    terminator: WebVTTLineTerminator | None = None

    @override
    def __str__(self) -> str:
        """Return a string representation of the cue component with terminator."""
        return f"{self.component}{self.terminator.value if self.terminator else ''}"


class WebVTTCueInternalText(BaseModel):
    """WebVTT cue internal text."""

    terminator: WebVTTLineTerminator | None = None
    components: Annotated[
        list[WebVTTCueComponentWithTerminator],
        Field(description=("WebVTT caption or subtitle cue components representing the cue internal text")),
    ] = []

    @override
    def __str__(self) -> str:
        """Return a string representation of the cue internal text."""
        cue_str = f"{self.terminator.value if self.terminator else ''}{''.join(str(span) for span in self.components)}"
        return cue_str


class WebVTTCueSpanStartTag(BaseModel):
    """WebVTT cue span start tag."""

    name: Annotated[START_TAG_NAMES, Field(description="The tag name")]
    classes: Annotated[
        list[str] | None,
        Field(description="List of classes representing the cue span's significance"),
    ] = None

    @field_validator("classes", mode="after")
    @classmethod
    def validate_classes(cls, value: list[str] | None) -> list[str] | None:
        """Validate cue span start tag classes."""
        for item in value or []:
            if any(ch in item for ch in {"\t", "\n", "\r", " ", "&", "<", ">", "."}):
                raise ValueError("A cue span start tag class contains invalid characters")
            if not item:
                raise ValueError("A cue span start tag class cannot be empty")
        return value

    def _get_name_with_classes(self) -> str:
        """Return the name of the cue span start tag with classes."""
        return f"{self.name}.{'.'.join(self.classes)}" if self.classes else self.name

    @override
    def __str__(self) -> str:
        """Return a string representation of the cue span start tag."""
        return f"<{self._get_name_with_classes()}>"


class WebVTTCueSpanStartTagAnnotated(WebVTTCueSpanStartTag):
    """WebVTT cue span start tag requiring an annotation."""

    annotation: Annotated[str, Field(description="Cue span start tag annotation")]

    @field_validator("annotation", mode="after")
    @classmethod
    def is_valid_annotation(cls, value: str) -> str:
        """Ensure annotation contains only permitted characters and HTML entities."""
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
    def __str__(self) -> str:
        """Return a string representation of the cue span start tag."""
        return f"<{self._get_name_with_classes()} {self.annotation}>"


class WebVTTCueLanguageSpanStartTag(WebVTTCueSpanStartTagAnnotated):
    """WebVTT cue language span start tag."""

    _pattern: ClassVar[re.Pattern] = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$", re.IGNORECASE)

    name: Literal["lang"] = Field("lang", description="The tag name")

    @field_validator("annotation", mode="after")
    @classmethod
    @override
    def is_valid_annotation(cls, value: str) -> str:
        """Ensure that the language annotation is in BCP 47 language tag format."""
        if cls._pattern.match(value):
            return value
        else:
            raise ValueError("Annotation should be in BCP 47 language tag format")


class WebVTTCueComponentBase(BaseModel):
    """WebVTT caption or subtitle cue component.

    All the WebVTT caption or subtitle cue components are represented by this class
    except the WebVTT cue text span, which requires different definitions.
    """

    kind: Literal["c", "b", "i", "u", "v", "lang"]
    start_tag: WebVTTCueSpanStartTag
    internal_text: WebVTTCueInternalText

    @model_validator(mode="after")
    def check_tag_names_match(self) -> Self:
        """Ensure that the start tag name matches this cue component type."""
        if self.kind != self.start_tag.name:
            raise ValueError("The tag name of this cue component should be {self.kind}")
        return self

    @override
    def __str__(self) -> str:
        """Return a string representation of the cue component."""
        return f"{self.start_tag}{self.internal_text}</{self.start_tag.name}>"


class WebVTTCueVoiceSpan(WebVTTCueComponentBase):
    """WebVTT cue voice span associated with a specific voice."""

    kind: Literal["v"] = "v"
    start_tag: WebVTTCueSpanStartTagAnnotated


class WebVTTCueClassSpan(WebVTTCueComponentBase):
    """WebVTT cue class span.

    It represents a span of text and it is used to annotate parts of the cue with
    applicable classes without implying further meaning (such as italics or bold).
    """

    kind: Literal["c"] = "c"
    start_tag: WebVTTCueSpanStartTag = WebVTTCueSpanStartTag(name="c")


class WebVTTCueItalicSpan(WebVTTCueComponentBase):
    """WebVTT cue italic span representing a span of italic text."""

    kind: Literal["i"] = "i"
    start_tag: WebVTTCueSpanStartTag = WebVTTCueSpanStartTag(name="i")


class WebVTTCueBoldSpan(WebVTTCueComponentBase):
    """WebVTT cue bold span representing a span of bold text."""

    kind: Literal["b"] = "b"
    start_tag: WebVTTCueSpanStartTag = WebVTTCueSpanStartTag(name="b")


class WebVTTCueUnderlineSpan(WebVTTCueComponentBase):
    """WebVTT cue underline span representing a span of underline text."""

    kind: Literal["u"] = "u"
    start_tag: WebVTTCueSpanStartTag = WebVTTCueSpanStartTag(name="u")


class WebVTTCueLanguageSpan(WebVTTCueComponentBase):
    """WebVTT cue language span.

    It represents a span of text and it is used to annotate parts of the cue where the
    applicable language might be different than the surrounding text's, without
    implying further meaning (such as italics or bold).
    """

    kind: Literal["lang"] = "lang"
    start_tag: WebVTTCueLanguageSpanStartTag


WebVTTCueComponent = Annotated[
    WebVTTCueTextSpan
    | WebVTTCueClassSpan
    | WebVTTCueItalicSpan
    | WebVTTCueBoldSpan
    | WebVTTCueUnderlineSpan
    | WebVTTCueVoiceSpan
    | WebVTTCueLanguageSpan,
    Field(
        discriminator="kind",
        description="The type of WebVTT caption or subtitle cue component.",
    ),
]


class WebVTTCueBlock(BaseModel):
    """Model representing a WebVTT cue block.

    The optional WebVTT cue settings list is not supported.
    The cue payload is limited to the following spans: text, class, italic, bold,
    underline, and voice.
    """

    model_config = ConfigDict(regex_engine="python-re")

    identifier: Annotated[WebVTTCueIdentifier | None, Field(description="The WebVTT cue identifier")] = None
    timings: Annotated[WebVTTCueTimings, Field(description="The WebVTT cue timings")]
    payload: Annotated[
        list[WebVTTCueComponentWithTerminator],
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
        """Ensure that the cue payload contains valid text."""
        for voice in payload:
            if "-->" in str(voice):
                raise ValueError("Cue payload must not contain '-->'")
        return payload

    @staticmethod
    def _create_text_components(
        text: str,
    ) -> Iterator[WebVTTCueComponentWithTerminator]:
        text_list = text.split("\n")
        for idx, line in enumerate(text.split("\n")):
            terminator = WebVTTLineTerminator.LF if idx < len(text_list) - 1 or text.endswith("\n") else None
            if len(line) > 0:
                yield WebVTTCueComponentWithTerminator(
                    component=WebVTTCueTextSpan(text=line),
                    terminator=terminator,
                )

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse a WebVTT cue block from a string.

        Args:
            raw: The raw WebVTT cue block string.

        Returns:
            The parsed WebVTT cue block.
        """
        lines = raw.strip().splitlines()
        if not lines:
            raise ValueError("Cue block must have at least one line")
        identifier: WebVTTCueIdentifier | None = None
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
        timings: WebVTTCueTimings = WebVTTCueTimings(start=WebVTTTimestamp(raw=start), end=WebVTTTimestamp(raw=end))
        cue_text = "\n".join(cue_lines).strip()
        # adding close tag for cue spans without end tag
        for omm in {"v"}:
            if cue_text.startswith(f"<{omm}") and f"</{omm}>" not in cue_text:
                cue_text += f"</{omm}>"
                break

        stack: list[list[WebVTTCueComponentWithTerminator]] = [[]]
        tag_stack: list[dict] = []

        pos = 0
        matches = list(cls._pattern_tag.finditer(cue_text))
        i = 0
        while i < len(matches):
            match = matches[i]
            if match.start() > pos:
                text = cue_text[pos : match.start()]
                stack[-1].extend(cls._create_text_components(text))
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
                        classes: list[str] | None = None
                        if class_string:
                            classes = [c for c in class_string.split(".") if c]
                        st: WebVTTCueSpanStartTag
                        if annotation and ct == "lang":
                            st = WebVTTCueLanguageSpanStartTag(name=ct, classes=classes, annotation=annotation.strip())
                        elif annotation:
                            st = WebVTTCueSpanStartTagAnnotated(name=ct, classes=classes, annotation=annotation.strip())
                        else:
                            st = WebVTTCueSpanStartTag(name=ct, classes=classes)
                        it = WebVTTCueInternalText(components=children)
                        cp: WebVTTCueComponent
                        if ct == "c":
                            cp = WebVTTCueClassSpan(start_tag=st, internal_text=it)
                        elif ct == "b":
                            cp = WebVTTCueBoldSpan(start_tag=st, internal_text=it)
                        elif ct == "i":
                            cp = WebVTTCueItalicSpan(start_tag=st, internal_text=it)
                        elif ct == "u":
                            cp = WebVTTCueUnderlineSpan(start_tag=st, internal_text=it)
                        elif ct == "lang":
                            cp = WebVTTCueLanguageSpan(start_tag=st, internal_text=it)
                        elif ct == "v":
                            cp = WebVTTCueVoiceSpan(start_tag=st, internal_text=it)
                        stack[-1].append(WebVTTCueComponentWithTerminator(component=cp))

            pos = match.end()
            i += 1

        if pos < len(cue_text):
            text = cue_text[pos:]
            stack[-1].extend(cls._create_text_components(text))

        return cls(
            identifier=identifier,
            timings=timings,
            payload=stack[0],
        )

    def format(self, omit_hours_if_zero: bool = False, omit_voice_end: bool = False) -> str:
        """Format the WebVTT cue block as a string.

        Args:
            omit_hours_if_zero: If True, omit hours when they are 0 in the timings.
            omit_voice_end: If True and this cue block has a WebVTT cue voice span as
                its only component, omit the voice end tag for brevity.

        Returns:
            Formatted cue block string.
        """
        parts = []
        if self.identifier:
            parts.append(f"{self.identifier}\n")
        timings_line = self.timings.format(omit_hours_if_zero=omit_hours_if_zero)
        parts.append(timings_line + "\n")
        for idx, span in enumerate(self.payload):
            if omit_voice_end and idx == 0 and len(self.payload) == 1 and span.component.kind == "v":
                parts.append(str(span).removesuffix("</v>"))
            else:
                parts.append(str(span))

        return "".join(parts) + "\n"

    def __str__(self) -> str:
        """Return a string representation of the WebVTT cue block.

        Always returns the full format including hours in timestamps (HH:MM:SS.mmm),
        even when hours are zero. Use `format(omit_hours_if_zero=True)` to get
        a shorter representation when hours are zero.
        Always returns the WebVTT cue voice spans with the voice end tag, even if this
        cue block has a WebVTT cue voice span as a single component in the payload. Use
        `format(omit_voice_end=True)` to get a shorter representation without the voice
        end tag.
        """
        return self.format()


class WebVTTFile(BaseModel):
    """A model representing a WebVTT file."""

    _pattern: ClassVar[re.Pattern] = re.compile(r"(?m)^(STYLE|NOTE|REGION)\b[\s\S]*?(?:\n\s*\n|\Z)")
    cue_blocks: list[WebVTTCueBlock]
    title: str | None = None

    @staticmethod
    def verify_signature(content: str) -> bool:
        """Verify the WebVTT file signature."""
        if not content:
            return False
        elif len(content) == 6:
            return content == "WEBVTT"
        elif len(content) > 6 and content.startswith("WEBVTT"):
            return content[6] in (" ", "\t", "\n")
        else:
            return False

    @model_validator(mode="after")
    def validate_start_time(self) -> Self:
        """Validate cue start times.

        The start time offset of the cue must be greater than or equal to the start
        time offsets of all previous cues.
        """
        idx: int = 0
        while idx < (len(self.cue_blocks) - 1):
            if self.cue_blocks[idx + 1].timings.start < self.cue_blocks[idx].timings.start:
                raise ValueError(
                    f"The start time offset of block {idx + 1} must be greater than or"
                    " equal to the start time offsets of all previous cues in the file"
                )
            idx += 1

        return self

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse a WebVTT file.

        Args:
            raw: The raw WebVTT file content.

        Returns:
            The parsed WebVTT file.
        """
        # Normalize newlines to LF
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")

        # Check WebVTT signature
        if not cls.verify_signature(raw):
            raise ValueError("Invalid WebVTT file signature")

        # Strip "WEBVTT" header line
        lines = raw.split("\n", 1)
        title = lines[0].removeprefix("WEBVTT").strip() or None
        body = lines[1] if len(lines) > 1 else ""

        # Remove NOTE/STYLE/REGION blocks
        body = re.sub(cls._pattern, "", body)

        # Split into cue blocks
        raw_blocks = re.split(r"\n\s*\n", body.strip())
        cues: list[WebVTTCueBlock] = []
        for block in raw_blocks:
            if not block.strip():
                continue
            try:
                cues.append(WebVTTCueBlock.parse(block))
            except ValueError as e:
                warnings.warn(f"Failed to parse cue block:\n{block}\n{e}", RuntimeWarning)

        return cls(title=title, cue_blocks=cues)

    def __iter__(self) -> Iterator[WebVTTCueBlock]:  # type: ignore[override]
        """Return an iterator over the cue blocks."""
        return iter(self.cue_blocks)

    def __getitem__(self, idx) -> WebVTTCueBlock:
        """Return the cue block at the given index."""
        return self.cue_blocks[idx]

    def __len__(self) -> int:
        """Return the number of cue blocks."""
        return len(self.cue_blocks)

    def format(self, omit_hours_if_zero: bool = False, omit_voice_end: bool = False) -> str:
        """Format the WebVTT file as a string.

        Args:
            omit_hours_if_zero: If True, omit hours when they are 0 in the timings.
            omit_voice_end: If True and cue blocks have a WebVTT cue voice span as
                the only component, omit the voice end tag for brevity.

        Returns:
            Formatted WebVTT file string.
        """
        parts: list[str] = []

        if self.title:
            parts.append(f"WEBVTT {self.title}\n")
        else:
            parts.append("WEBVTT\n")

        for cue_block in self.cue_blocks:
            parts.append("\n")
            parts.append(cue_block.format(omit_hours_if_zero=omit_hours_if_zero, omit_voice_end=omit_voice_end))

        # Remove the trailing newline from the last cue block
        return "".join(parts).rstrip("\n")

    def __str__(self) -> str:
        """Return a string representation of the WebVTT file.

        Always returns the full format including hours in timestamps (HH:MM:SS.mmm),
        even when hours are zero. Use `format(omit_hours_if_zero=True)` to get
        a shorter representation when hours are zero.
        """
        return self.format()
