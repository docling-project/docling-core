"""Define classes for WebVTT serialization."""

import logging
import re
from pathlib import Path
from typing import Annotated, Any, get_args

from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BaseMetaSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    create_ser_result,
)
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    NodeItem,
    PictureItem,
    TableItem,
    TextItem,
    TitleItem,
    TrackSource,
)
from docling_core.types.doc.webvtt import (
    START_TAG_NAMES,
    WebVTTCueBlock,
    WebVTTCueSpanStartTag,
    WebVTTCueSpanStartTagAnnotated,
    WebVTTCueTimings,
    WebVTTFile,
    WebVTTLineTerminator,
    WebVTTTimestamp,
)

_logger = logging.getLogger(__name__)


def _remove_consecutive_pairs(text: str) -> str:
    """Remove one pass of consecutive start/end tag pairs.

    This function looks for patterns like </tag><tag> where the tags are identical
    and removes them. It handles two cases:
    1. Direct adjacent tags with content: <tag>content</tag>whitespace<tag>
    2. Tags with other tags in between: </tag><othertag><tag>

    Args:
        text: Input string

    Returns:
        String with one pass of consecutive pairs removed
    """
    # Pattern 1: Direct adjacent tags </tag><tag> with same classes and annotations
    pattern1 = re.compile(
        r"<([bciuv]|lang)((?:\.\w+)*)(?:\s+([^>]+))?>"  # Opening tag: capture tag, classes, annotation
        r"((?:(?!</\1>).)*?)"  # Content (non-greedy, not containing the closing tag)
        r"</\1>"  # Closing tag
        r"(\s*)"  # Capture whitespace between tags (including newlines)
        r"<\1((?:\.\w+)*)(?:\s+([^>]+))?>"  # Next opening tag: capture classes and annotation
    )

    def replacer1(match: re.Match[str]) -> str:
        tag = match.group(1)
        classes1 = match.group(2) or ""
        anno1 = match.group(3) or ""
        content = match.group(4)
        whitespace = match.group(5)  # Whitespace between tags
        classes2 = match.group(6) or ""
        anno2 = match.group(7) or ""

        # Only merge if classes and annotations match
        if classes1 == classes2 and anno1 == anno2:
            # Merge: remove the closing and opening tags, but keep the whitespace
            return f"<{tag}{classes1}{' ' + anno1 if anno1 else ''}>{content}{whitespace}"
        else:
            # Don't merge - return original
            return match.group(0)

    # Pattern 2: Tags with other tags in between </tag><othertag><tag>
    # This removes redundant </tag> and <tag> when there's another tag in between
    pattern2 = re.compile(
        r"</([bciuv]|lang)>"  # Closing tag
        r"(<[^>]+>)"  # Any other tag in between
        r"<\1(?:\.\w+)*(?:\s+[^>]+)?>"  # Same opening tag (with any classes/annotations)
    )

    def replacer2(match: re.Match[str]) -> str:
        # Just keep the middle tag, remove the closing and opening of the same type
        return match.group(2)

    result = pattern1.sub(replacer1, text)
    result = pattern2.sub(replacer2, result)

    return result


class WebVTTParams(CommonParams):
    """Serialization parameters for the Web Video Text Tracks (WebVTT) format."""

    layers: set[ContentLayer] = {ContentLayer.BODY}
    omit_hours_if_zero: Annotated[bool, Field(description="If True, omit hours when they are 0 in the timings.")] = (
        False
    )
    omit_voice_end: Annotated[
        bool,
        Field(
            description=(
                "If True and cue blocks have a WebVTT cue voice span as the only component, omit the voice end tag for"
                " brevity."
            )
        ),
    ] = False


class WebVTTTextSerializer(BaseModel, BaseTextSerializer):
    """Text serializer to Web Video Text Tracks (WebVTT) format."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        visited: set[str] | None = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # Handle TitleItem specially - it doesn't have provenance but we need its text
        if isinstance(item, TitleItem):
            return create_ser_result(text=item.text, span_source=item)

        # Only process items with TrackSource (WebVTT cues)
        if not item.text or not item.source or item.source[0].kind != "track":
            return create_ser_result()

        # Apply post-processing here: formatting and voice.
        # If the TextItem is part of an InlineGroup, we need to further post-process it within the group context.
        source: TrackSource = item.source[0]
        text: str = doc_serializer.post_process(
            text=item.text,
            formatting=item.formatting,
            voice=source.voice,
        )
        if is_inline_scope:
            # Iteratively remove unnecessary consecutive tag pairs until no more changes
            prev_text: str | None = None
            while prev_text != text:
                prev_text = text
                text = _remove_consecutive_pairs(text)

        return create_ser_result(text=text, span_source=item)


class _WebVTTTableSerializer(BaseTableSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _WebVTTPictureSerializer(BasePictureSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _WebVTTKeyValueSerializer(BaseKeyValueSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _WebVTTFormSerializer(BaseFormSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _WebVTTFallbackSerializer(BaseFallbackSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (item, doc_serializer, doc, kwargs)
        return create_ser_result()


class _WebVTTListSerializer(BaseModel, BaseListSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (doc, list_level, is_inline_scope, item, doc_serializer, kwargs)
        return create_ser_result()


class WebVTTInlineSerializer(BaseInlineSerializer):
    """Inline group serializer to Web Video Text Tracks (WebVTT) format."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: set[str] | None = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes an inline group to WebVTT format."""
        _ = doc
        my_visited = visited if visited is not None else set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )
        # Include all parts, even if text is empty or whitespace-only
        # Use 'is not None' instead of truthiness check to preserve whitespace
        text_res = "".join([p.text for p in parts if p.text is not None])

        # Apply tag normalization to the concatenated result
        # Iteratively remove consecutive pairs until no more changes
        prev_text = None
        while prev_text != text_res:
            prev_text = text_res
            text_res = _remove_consecutive_pairs(text_res)

        return create_ser_result(text=text_res, span_source=parts)


class _WebVTTMetaSerializer(BaseModel, BaseMetaSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (doc, item, kwargs)
        return create_ser_result()


class _WebVTTAnnotationSerializer(BaseModel, BaseAnnotationSerializer):
    """No-op for WebVTT output (not represented)."""

    @override
    def serialize(
        self,
        *,
        item: DocItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        _ = (doc, item, kwargs)
        return create_ser_result()


class WebVTTDocSerializer(DocSerializer):
    """Document serializer to Web Video Text Tracks (WebVTT) format."""

    text_serializer: BaseTextSerializer = WebVTTTextSerializer()
    table_serializer: BaseTableSerializer = _WebVTTTableSerializer()
    picture_serializer: BasePictureSerializer = _WebVTTPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = _WebVTTKeyValueSerializer()
    form_serializer: BaseFormSerializer = _WebVTTFormSerializer()
    fallback_serializer: BaseFallbackSerializer = _WebVTTFallbackSerializer()
    list_serializer: BaseListSerializer = _WebVTTListSerializer()
    inline_serializer: BaseInlineSerializer = WebVTTInlineSerializer()
    meta_serializer: BaseMetaSerializer | None = _WebVTTMetaSerializer()
    annotation_serializer: BaseAnnotationSerializer = _WebVTTAnnotationSerializer()

    params: WebVTTParams = WebVTTParams()

    @override
    def requires_page_break(self) -> bool:
        """Whether to add page breaks.

        WebVTT format does not support page breaks.
        """
        return False

    @override
    def serialize_bold(self, text: str, **kwargs) -> str:
        """Apply WebVTT-specific bold serialization."""

        return self.serialize_cue_span(text=text, tag="b")

    @override
    def serialize_italic(self, text: str, **kwargs) -> str:
        """Apply WebVTT-specific italic serialization."""

        return self.serialize_cue_span(text=text, tag="i")

    @override
    def serialize_underline(self, text: str, **kwargs) -> str:
        """Apply WebVTT-specific underline serialization."""

        return self.serialize_cue_span(text=text, tag="u")

    def serialize_cue_span(
        self,
        text: str,
        tag: START_TAG_NAMES,
        anno: str | None = None,
    ) -> str:
        """Apply serialization to a WebVTT cue span.

        Currently, only b, i, u, and v tags are supported.
        """
        start_tag: WebVTTCueSpanStartTag
        if tag in {"b", "i", "u"}:
            start_tag = WebVTTCueSpanStartTag(name=tag)
        elif tag in {"v"}:
            if not anno:
                _logger.warning(f"Invalid {tag} cue span without annotation: {text}")
                return text
            else:
                start_tag = WebVTTCueSpanStartTagAnnotated(name=tag, annotation=anno)
        else:
            return text

        res: str = f"{start_tag}{text}</{tag}>"
        return res

    @staticmethod
    def _extract_classes(classes: list[str]) -> dict[str, list[str]]:
        """Extract tag and values from provenance classes.

        Args:
            classes: The classes from a TrackSource object.

        Returns:
            Map of tag to class values.
        """
        res: dict[str, list[str]] = {}
        for item in classes or []:
            for prefix in get_args(START_TAG_NAMES):
                if item == prefix:
                    res[prefix] = []
                    break
                elif item.startswith(prefix + "."):
                    cls_str: str = item[len(prefix) + 1 :]
                    res[prefix] = cls_str.split(".")
                    break
        return res

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts."""
        title: str | None = None

        timings: WebVTTCueTimings | None = None
        id: str | None = None
        text: str = ""
        cue_blocks: list[WebVTTCueBlock] = []
        for part in parts:
            if not part.text or not part.spans:
                continue

            # Get the doc item from the first span
            doc_item: DocItem = part.spans[0].item

            # Handle title items (check both TitleItem type and label)
            if isinstance(doc_item, TitleItem) or (
                isinstance(doc_item, TextItem) and doc_item.label == DocItemLabel.TITLE
            ):
                title = part.text
                continue
            if isinstance(doc_item, InlineGroup) and doc_item.children:
                doc_item = doc_item.children[0].resolve(doc=self.doc)
            if isinstance(doc_item, TextItem) and doc_item.source and doc_item.source[0].kind == "track":
                prov: TrackSource = doc_item.source[0]
                if (
                    prov.identifier == id
                    and timings
                    and timings.start.seconds == prov.start_time
                    and timings.end.seconds == prov.end_time
                ):
                    # When combining items with same timing, add newline and merge consecutive tags
                    combined = text.rstrip() + WebVTTLineTerminator.LF.value + part.text
                    # Use _remove_consecutive_pairs to merge tags like </v>\n<v Speaker A>
                    # Iteratively remove consecutive pairs until no more changes
                    prev_combined = None
                    while prev_combined != combined:
                        prev_combined = combined
                        combined = _remove_consecutive_pairs(combined)
                    text = combined + WebVTTLineTerminator.LF.value
                else:
                    if text:
                        cue_blocks.append(WebVTTCueBlock.parse(text))
                    timings = WebVTTCueTimings(
                        start=WebVTTTimestamp.from_seconds(prov.start_time),
                        end=WebVTTTimestamp.from_seconds(prov.end_time),
                    )
                    id = prov.identifier
                    text = (
                        f"{id + WebVTTLineTerminator.LF.value if id else ''}{timings}"
                        f"{WebVTTLineTerminator.LF.value}{part.text}"
                        f"{WebVTTLineTerminator.LF.value}"
                    )
        if text:
            cue_blocks.append(WebVTTCueBlock.parse(text))

        webvtt_file = WebVTTFile(title=title, cue_blocks=cue_blocks)
        content = webvtt_file.format(
            omit_hours_if_zero=self.params.omit_hours_if_zero, omit_voice_end=self.params.omit_voice_end
        )
        return create_ser_result(text=content, span_source=parts)

    def post_process(
        self,
        text: str,
        *,
        formatting: Formatting | None = None,
        hyperlink: AnyUrl | Path | None = None,
        **kwargs: Any,
    ) -> str:
        """Apply some text post-processing steps by adding formatting tags.

        The order of the formatting tags is determined by this function and `DocSerializer.post_process`,
        from the innermost to the outermost:
            1. underline (<u>)
            2. italic (<i>)
            3. bold (<b>)
            4. voice (<v>)
        """
        res: str = text

        res = super().post_process(text=res, formatting=formatting)

        voice: str | None = kwargs.get("voice", None)
        if voice:
            res = self.serialize_cue_span(text=res, tag="v", anno=voice)

        return res
